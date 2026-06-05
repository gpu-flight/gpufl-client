#pragma once

// Env-var names and profile constants shared between the launcher binary
// (`gpufl trace`) and the injection shared library (`libgpufl_inject.so`).
// The launcher sets these in the child env before fork+exec; the inject
// lib's constructor reads them to drive gpufl::init().

namespace gpufl::inject {

// Sentinel — the inject lib's constructor returns silently if this is
// not "1" in the environment. Stops accidental LD_PRELOAD into shells
// or system utilities from firing CUPTI.
constexpr const char* kEnvSentinel = "GPUFL_INJECT";

// Session display name. Default: basename of the launched command.
constexpr const char* kEnvAppName = "GPUFL_APP_NAME";

// Local NDJSON output directory (will get an `app.log` appended for
// gpufl::InitOptions.log_path).
constexpr const char* kEnvLogDir = "GPUFL_LOG_DIR";

// Profile preset: "comprehensive" (default) | "light" | "monitoring-only".
// Maps to one of the `*_default_options()` factories in gpufl.hpp.
constexpr const char* kEnvProfile = "GPUFL_INJECT_PROFILE";

// Override the profiling engine. The launcher validates the value and
// forwards it verbatim; gpufl::init() is the single parser for this var
// (see gpufl.cpp), so the inject lib does NOT parse it. Canonical values:
// "Monitor" | "Trace" | "PcSampling" | "SassMetrics" | "PmSampling" |
// "RangeProfiler" | "Deep". Empty = use the profile's default engine.
constexpr const char* kEnvProfilingEngine = "GPUFL_PROFILING_ENGINE";

// Opt-in: when "1", the inject lib runs gpufl::uploadLogs() right after
// gpufl::shutdown() returns (and before signalling completion), shipping
// the session's NDJSON to the backend. Creds come from the standard
// GPUFL_API_KEY / GPUFL_BACKEND_URL / GPUFL_API_PATH env vars (the same
// ones gpufl::init() reads). Set by the launcher's `--upload` flag, which
// pre-checks that the key + url are present before exec.
constexpr const char* kEnvUpload = "GPUFL_INJECT_UPLOAD";

// Phase 2: write end of an anonymous pipe inherited via fork. Inject
// lib writes a single byte after gpufl::shutdown() returns so the
// launcher knows uploads are drained. Empty = no completion signal.
constexpr const char* kEnvCompletionFd = "GPUFL_INJECT_COMPLETION_FD";

// Opt-in: when set to "1", the inject lib's __attribute__((constructor))
// runs gpufl::init() at ld.so time (before main, before cuInit). Default
// off because the Phase 0.1 spike showed CUPTI subscribe before cuInit
// segfaults on no-CUDA targets. With this off, only InitializeInjection
// (called by libcuda after cuInit) drives init.
constexpr const char* kEnvUseConstructor = "GPUFL_INJECT_USE_CONSTRUCTOR";

// Profile-name string values (must stay in sync with the launcher's
// `--profile` flag parsing).
constexpr const char* kProfileComprehensive = "comprehensive";
constexpr const char* kProfileLight         = "light";
constexpr const char* kProfileMonitoringOnly = "monitoring-only";

}  // namespace gpufl::inject
