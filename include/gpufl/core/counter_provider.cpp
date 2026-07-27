#include "gpufl/core/counter_provider.hpp"

#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#include <cstdlib>

#include "gpufl/core/counter_registry.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/debug_logger.hpp"

namespace gpufl::detail {
namespace {

#if defined(_WIN32)
constexpr const char* kRuntimeName = "gpufl_counter_runtime.dll";
#elif defined(__APPLE__)
constexpr const char* kRuntimeName = "libgpufl_counter_runtime.dylib";
#else
constexpr const char* kRuntimeName = "libgpufl_counter_runtime.so";
#endif

std::mutex g_mu;
// Only a SUCCESSFUL bind is cached. A failure is not final: under injection the
// runtime appears when the driver loads gpufl_inject, which is after the target
// may already have registered a counter. Caching that failure forever is what
// would leave the application on one registry and the evaluator on another.
bool g_resolved = false;
bool g_failed_once = false;
const gpufl_counter_provider_v1* g_provider = nullptr;
bool g_shared = false;

/**
 * Directory of the module this code is linked into - the injection DLL, the
 * Python extension, or the host executable. NOT the process directory: under
 * CUDA_INJECTION64_PATH the process is the profiled target, which has no
 * reason to sit next to our files.
 */
std::string ThisModuleDir() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&ThisModuleDir), &module)) {
        return {};
    }
    char path[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(module, path, sizeof(path));
    if (n == 0 || n >= sizeof(path)) return {};
    std::string full(path, n);
    const auto slash = full.find_last_of("\\/");
    return slash == std::string::npos ? std::string{} : full.substr(0, slash);
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&ThisModuleDir), &info) == 0 ||
        info.dli_fname == nullptr) {
        return {};
    }
    const std::string full(info.dli_fname);
    const auto slash = full.find_last_of('/');
    return slash == std::string::npos ? std::string{} : full.substr(0, slash);
#endif
}

/**
 * A copy already mapped into this process, or nullptr.
 *
 * Tried before any path, and it is what actually delivers the single-instance
 * property. Deployment colocates the runtime with each consumer - beside the
 * injection DLL, inside the Python wheel - so several copies exist on disk, and
 * loading them by full path would map them as separate modules with separate
 * registries. That is the very split this ABI exists to prevent. Binding to
 * whatever is already loaded makes the number of copies on disk irrelevant.
 */
void* AlreadyLoaded() {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetModuleHandleA(kRuntimeName));
#else
    // NOLOAD returns a handle only if the object is already mapped.
    return dlopen(kRuntimeName, RTLD_NOW | RTLD_NOLOAD);
#endif
}

void* OpenLibrary(const std::string& path) {
#if defined(_WIN32)
    return LoadLibraryA(path.c_str());
#else
    // RTLD_GLOBAL so a second binder resolves to this same object rather than
    // mapping its own copy.
    return dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
}

void* FindSymbol(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

/** Directory part of a path, or empty. */
std::string DirOf(const std::string& path) {
    const auto slash = path.find_last_of("\\/");
    return slash == std::string::npos ? std::string{} : path.substr(0, slash);
}

/** Candidate locations, most specific first. */
std::vector<std::string> BuildCandidates(const char* injection_path,
                                         const char* explicit_runtime_path) {
    std::vector<std::string> out;

    // An explicit override wins. Deployment layouts we do not control need a
    // way to say where the runtime is without guessing.
    if (explicit_runtime_path != nullptr && explicit_runtime_path[0] != '\0') {
        out.emplace_back(explicit_runtime_path);
    }

    // Beside the injection library. This is what makes an ordinary C++ target
    // work: it links gpufl statically, so ThisModuleDir() is the TARGET's
    // directory, which has no runtime in it. If that target calls counter()
    // before its first CUDA call - entirely normal - it would bind to its own
    // local registry and stay there, invisible to the evaluator that loads
    // later. The launcher puts this variable in the child's environment before
    // exec, so it is readable from the very first instruction.
    if (injection_path != nullptr && injection_path[0] != '\0') {
        if (const std::string dir = DirOf(injection_path); !dir.empty()) {
            out.push_back(dir + "/" + kRuntimeName);
        }
    }

    const std::string dir = ThisModuleDir();
    if (!dir.empty()) {
        out.push_back(dir + "/" + kRuntimeName);
        // Build trees put the shared runtime beside the config directory
        // rather than next to every consumer.
        out.push_back(dir + "/../" + kRuntimeName);
        out.push_back(dir + "/../bin/" + kRuntimeName);
    }
    // Last resort: the platform loader's own search. Also the path that finds
    // an already-loaded copy by name, which is the common case for whichever
    // module binds second.
    out.emplace_back(kRuntimeName);
    return out;
}

bool Resolve() {
    // Whatever is already mapped wins, before any path is considered. See
    // AlreadyLoaded(): several copies of this library legitimately exist on
    // disk, and binding by path would defeat the point.
    std::vector<std::string> candidates =
        BuildCandidates(std::getenv(env::kCudaInjection64Path),
                        std::getenv(env::kCounterRuntimePath));
    candidates.insert(candidates.begin(), std::string{});   // "" = already loaded

    for (const std::string& candidate : candidates) {
        void* handle = candidate.empty() ? AlreadyLoaded() : OpenLibrary(candidate);
        if (handle == nullptr) continue;

        using GetProviderFn = const gpufl_counter_provider_v1* (*)();
        const auto entry = reinterpret_cast<GetProviderFn>(
            FindSymbol(handle, "gpufl_get_counter_provider_v1"));
        if (entry == nullptr) continue;

        const gpufl_counter_provider_v1* provider = entry();
        if (provider == nullptr) continue;
        // Check both before calling through: a newer runtime may have appended
        // members this build does not know about, and an older one may lack
        // members this build assumes.
        if (provider->abi_version != GPUFL_COUNTER_ABI_VERSION ||
            provider->struct_size < sizeof(gpufl_counter_provider_v1)) {
            GFL_LOG_ERROR("[CounterProvider] ", candidate,
                          " reports ABI ", provider->abi_version,
                          " (size ", provider->struct_size, "); this build needs ",
                          GPUFL_COUNTER_ABI_VERSION, " (size ",
                          sizeof(gpufl_counter_provider_v1), ")");
            continue;
        }

        g_provider = provider;
        g_shared = true;
        GFL_LOG_DEBUG("[CounterProvider] bound to shared runtime: ",
                      candidate.empty() ? "(already loaded)" : candidate.c_str());
        return true;
    }

    // No shared runtime. Correct for an embedded host, which holds the only
    // copy of gpufl in the process; wrong under injection, where the target and
    // the evaluator are separate modules and would each get their own registry.
    // isShared() is what lets the evaluator refuse a rule it cannot honour,
    // rather than reporting a counter that is being ticked as Missing.
    g_provider = nullptr;
    g_shared = false;
    GFL_LOG_DEBUG("[CounterProvider] ", kRuntimeName,
                  " not found; counters stay local to this module");
    return false;
}

}  // namespace

std::vector<std::string> CounterRuntimeCandidatesForTesting(
    const char* injection_path, const char* explicit_runtime_path) {
    return BuildCandidates(injection_path, explicit_runtime_path);
}

const gpufl_counter_provider_v1* CounterProvider::get() {
    std::lock_guard lk(g_mu);
    if (!g_resolved) {
        g_resolved = Resolve();
        if (!g_resolved) g_failed_once = true;
        return g_provider;
    }
    return g_provider;
}

bool CounterProvider::isShared() {
    get();
    std::lock_guard lk(g_mu);
    return g_shared;
}

void CounterProvider::resetForTesting() {
    std::lock_guard lk(g_mu);
    g_resolved = false;
    g_failed_once = false;
    g_provider = nullptr;
    g_shared = false;
}

}  // namespace gpufl::detail
