#include "gpufl/backends/nvidia/cupti_runtime_support.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/env_vars.hpp"

namespace gpufl {
namespace {

std::atomic<CuptiBackend*> g_activeBackend{nullptr};

}  // namespace

bool IsInsufficientPrivilege(CUptiResult res) {
    if (res == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) return true;
#ifdef CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES
    if (res == CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES)
        return true;
#endif
    return false;
}

void LogCuptiIfUnexpected(const char* scope, const char* op, CUptiResult res) {
    if (res == CUPTI_SUCCESS || res == CUPTI_ERROR_NOT_INITIALIZED ||
        IsInsufficientPrivilege(res)) {
        return;
    }
    LogCuptiErrorIfFailed(scope, op, res);
}

#if defined(__linux__)
// CUPTI loads PerfWorks by soname the first time a profiling feature
// runs - cuptiPCSamplingEnable (PC sampling) and cuptiProfilerInitialize
// (SASS / Range / Deep) both do. PerfWorks is NOT one library: the host
// API (libnvperf_host.so) pulls in companions - libnvperf_target.so (the
// driver-side counterpart that NVPW_CUDA_LoadDriver initializes) and, for
// PC sampling, libpcsamplingutil.so. ALL of them must come from the SAME
// CUDA install as the libcupti we're bound to. If the dynamic loader
// resolves ANY of them from a DIFFERENT install (classic case: a pip
// `nvidia-cu13` CUPTI inside a venv + the system /usr/local/cuda nvperf),
// NVPW_CUDA_LoadDriver SEGFAULTs on the version mismatch - the crash that
// killed BOTH Deep and PcSampling in PyTorch venvs.
//
// Putting the matching directory on LD_LIBRARY_PATH fixes it because that
// redirects the WHOLE set. An earlier version of this preloaded ONLY
// libnvperf_host.so, which was NOT enough: host resolved to the venv copy
// but its companion libnvperf_target.so still resolved to the mismatched
// system copy, so NVPW_CUDA_LoadDriver kept crashing. So we preload the
// ENTIRE PerfWorks set sitting next to our libcupti, RTLD_GLOBAL, in
// dependency order (target before host). The loader tracks shared objects
// by SONAME, so once these are resident CUPTI's later internal dlopen()s
// return THESE regardless of LD_LIBRARY_PATH ordering. Best-effort:
// anything missing is logged and skipped - CUPTI falls back to the
// loader's choice (prior behavior).
void PreloadMatchingPerfWorks() {
    Dl_info info{};
    // &cuptiSubscribe resolves into whichever libcupti this binary is
    // bound to; dladdr hands back that library's on-disk path.
    if (!dladdr(reinterpret_cast<void*>(&cuptiSubscribe), &info) ||
        !info.dli_fname || !info.dli_fname[0]) {
        GFL_LOG_DEBUG("[CuptiBackend] PerfWorks preload: couldn't locate our "
                      "libcupti via dladdr; skipping (CUPTI will use the "
                      "loader's PerfWorks libs).");
        return;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::path(info.dli_fname).parent_path();
    if (dir.empty() || !fs::is_directory(dir, ec)) {
        GFL_LOG_DEBUG("[CuptiBackend] PerfWorks preload: our libcupti's "
                      "directory is not accessible; skipping.");
        return;
    }

    auto tryLoad = [](const fs::path& p) {
        if (dlopen(p.string().c_str(), RTLD_LAZY | RTLD_GLOBAL)) {
            GFL_LOG_DEBUG("[CuptiBackend] Preloaded PerfWorks lib: ",
                          p.string());
        } else {
            const char* err = dlerror();
            GFL_LOG_DEBUG("[CuptiBackend] PerfWorks preload skipped ",
                          p.string(), " (", err ? err : "n/a", ")");
        }
    };

    // Bucket the companion libs in our CUPTI's directory. Load order
    // matters: dependencies before dependents. libnvperf_target.so (driver
    // side) and other helpers go first, libnvperf_host.so last - otherwise
    // host's DT_NEEDED on the target soname resolves to the system copy
    // BEFORE we make the matching one resident, and soname-dedup then locks
    // in the wrong target.
    std::vector<fs::path> targets, hosts, others;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        if (name.find(".so") == std::string::npos) continue;
        if (name.rfind("libnvperf_target", 0) == 0) {
            targets.push_back(entry.path());
        } else if (name.rfind("libnvperf_host", 0) == 0) {
            hosts.push_back(entry.path());
        } else if (name.rfind("libnvperf", 0) == 0 ||
                   name.rfind("libpcsamplingutil", 0) == 0) {
            others.push_back(entry.path());
        }
    }

    if (targets.empty() && hosts.empty() && others.empty()) {
        GFL_LOG_DEBUG("[CuptiBackend] No PerfWorks libs found next to our "
                      "CUPTI in ", dir.string(), " - CUPTI will use the "
                      "loader's choice, which may mismatch and crash in "
                      "NVPW_CUDA_LoadDriver on split CUDA installs.");
        return;
    }

    for (const auto& p : targets) tryLoad(p);  // driver-side first
    for (const auto& p : others) tryLoad(p);   // pcsamplingutil, etc.
    for (const auto& p : hosts) tryLoad(p);     // host API last
}
#elif defined(_WIN32)
// Windows variant of the same fix. PyTorch wheels ship a versionless
// nvperf_host.dll (CUDA 12.x) in torch\lib, so our cupti64's by-name load
// of "nvperf_host.dll" can resolve torch's mismatched copy. Preload the
// PerfWorks set next to OUR cupti64 by absolute path - once resident, later
// by-name loads return it. Load order mirrors Linux (target, util, host).
void PreloadMatchingPerfWorks() {
    HMODULE cuptiMod = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&cuptiSubscribe),
                            &cuptiMod) ||
        !cuptiMod) {
        GFL_LOG_DEBUG("[CuptiBackend] PerfWorks preload: couldn't locate our "
                      "cupti64 module; skipping (CUPTI will use the loader's "
                      "PerfWorks DLLs).");
        return;
    }
    wchar_t buf[MAX_PATH] = {};
    if (!GetModuleFileNameW(cuptiMod, buf, MAX_PATH)) {
        GFL_LOG_DEBUG("[CuptiBackend] PerfWorks preload: couldn't resolve our "
                      "cupti64 module path; skipping.");
        return;
    }
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(buf).parent_path();

    auto tryLoad = [](const fs::path& p) {
        if (LoadLibraryW(p.wstring().c_str())) {
            GFL_LOG_DEBUG("[CuptiBackend] Preloaded PerfWorks lib: ",
                          p.string());
        } else {
            GFL_LOG_DEBUG("[CuptiBackend] PerfWorks preload skipped ",
                          p.string(), " (error ", GetLastError(), ")");
        }
    };

    tryLoad(dir / L"nvperf_target.dll");
    tryLoad(dir / L"pcsamplingutil.dll");
    tryLoad(dir / L"nvperf_host.dll");
}
#else
void PreloadMatchingPerfWorks() {}
#endif

bool WindowsInjectedProcess() {
#if defined(_WIN32)
    const char* injected = std::getenv(gpufl::env::kInject);
    return injected && std::strcmp(injected, "1") == 0;
#else
    return false;
#endif
}

// NVIDIA calls InitializeInjection while the Windows CUDA driver is still
// initializing. Creating a CUDA context from that callback can re-enter the
// driver and deadlock, so the injected path only uses an already-current
// context.
bool TryCurrentCudaContext(CUcontext* ctx) {
    if (!ctx) return false;
    if (*ctx && IsContextValid(*ctx)) return true;

    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current) {
        *ctx = current;
        return true;
    }
    return false;
}

void SetActiveCuptiBackend(CuptiBackend* backend) {
    g_activeBackend.store(backend, std::memory_order_release);
}

CuptiBackend* GetActiveCuptiBackend() {
    return g_activeBackend.load(std::memory_order_acquire);
}

}  // namespace gpufl
