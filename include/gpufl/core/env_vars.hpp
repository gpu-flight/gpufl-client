#pragma once

// ─────────────────────────────────────────────────────────────────────────
// Central registry of every environment-variable NAME GPUFlight reads or sets.
//
// One place, so each name lives exactly once — no more the same "GPUFL_…"
// string typed as a raw literal in five files and silently drifting. Read or
// set env through these constants, never a bare string literal:
//
//     if (const char* v = std::getenv(gpufl::env::kApiKey)) { ... }
//
// Names only — the VALUES some of these carry (e.g. "EAGER" for
// kCudaModuleLoading, the "comprehensive"/"light"/… profile presets in
// inject/inject_entry.hpp) live with their own logic, not here.
//
// Pure-OS lookups the code also does (PATH, HOME, USERPROFILE, HOMEDRIVE,
// HOMEPATH) are intentionally left as literals at their call sites: they're OS
// contracts, not GPUFlight knobs.
// ─────────────────────────────────────────────────────────────────────────

namespace gpufl::env {

// ── Kill switch / config / credentials ─────────────────────────────────────
// Master off switch — truthy => gpufl::init() returns early (no CUPTI / NVML).
constexpr const char* kDisabled     = "GPUFL_DISABLED";
// Path to a local JSON config file (InitOptions.config_file fallback).
constexpr const char* kConfigFile   = "GPUFL_CONFIG_FILE";
// Backend base URL / bearer token / API path prefix — read by init()'s remote
// config probe, the inject-lib post-run upload, and the `gpufl upload` command.
constexpr const char* kBackendUrl   = "GPUFL_BACKEND_URL";
constexpr const char* kApiKey       = "GPUFL_API_KEY";
constexpr const char* kApiPath      = "GPUFL_API_PATH";
// Fallback URL for the remote named-config fetch when kBackendUrl is unset.
constexpr const char* kRemoteConfig = "GPUFL_REMOTE_CONFIG";

// ── Launcher ↔ inject-lib protocol ──────────────────────────────────────────
// `gpufl trace` sets these in the child env before fork+exec; the inject lib's
// entry point reads them to drive gpufl::init(). (Moved here from
// inject/inject_entry.hpp so every env name lives in one place.)
//
// Sentinel — the inject lib returns silently unless this is "1", so a stray
// LD_PRELOAD into a shell/utility can't accidentally fire CUPTI.
constexpr const char* kInject               = "GPUFL_INJECT";
// Session display name (default: basename of the launched command).
constexpr const char* kAppName              = "GPUFL_APP_NAME";
// Local NDJSON output directory (becomes InitOptions.log_path).
constexpr const char* kLogDir               = "GPUFL_LOG_DIR";
// Profile preset: "comprehensive" | "light" | "monitoring-only" (the matching
// VALUE constants are kProfile* in inject/inject_entry.hpp).
constexpr const char* kInjectProfile        = "GPUFL_INJECT_PROFILE";
// Opt-in ("1"): inject lib runs uploadLogs() right after shutdown(), shipping
// the session NDJSON with the kApiKey / kBackendUrl / kApiPath creds.
constexpr const char* kInjectUpload         = "GPUFL_INJECT_UPLOAD";
// Write end of an inherited pipe; the inject lib writes one byte after
// shutdown() so the launcher knows uploads drained. Empty = no signal.
constexpr const char* kInjectCompletionFd   = "GPUFL_INJECT_COMPLETION_FD";
// Opt-in ("1"): run gpufl::init() from the inject lib's ld.so constructor
// (before main / cuInit). Off by default — pre-cuInit CUPTI subscribe segfaults
// on no-CUDA targets; normally InitializeInjection drives init after cuInit.
constexpr const char* kInjectUseConstructor = "GPUFL_INJECT_USE_CONSTRUCTOR";

// ── Inject-lib timing knobs (tuning / debugging the injection lifecycle) ────
constexpr const char* kInjectShutdownDelayMs = "GPUFL_INJECT_SHUTDOWN_DELAY_MS";
constexpr const char* kInjectSyncWaitMs      = "GPUFL_INJECT_SYNC_WAIT_MS";
constexpr const char* kInjectLaunchWaitMs    = "GPUFL_INJECT_LAUNCH_WAIT_MS";
constexpr const char* kInjectProcessScope    = "GPUFL_INJECT_PROCESS_SCOPE";
constexpr const char* kInjectInitDelayMs     = "GPUFL_INJECT_INIT_DELAY_MS";

// ── Profiling engine selection ──────────────────────────────────────────────
// Override the resolved ProfilingEngine; gpufl::init() is the single string→enum
// parser. Canonical values: Monitor | Trace | PcSampling | SassMetrics |
// PmSampling | RangeProfiler | Deep.
constexpr const char* kProfilingEngine    = "GPUFL_PROFILING_ENGINE";
// Opt-in (1/true/yes/on): force CUDA_MODULE_LOADING=EAGER for SASS / Deep.
constexpr const char* kEagerModuleLoading = "GPUFL_EAGER_MODULE_LOADING";

// ── Multi-pass profiling (set per pass by the launcher's multi-pass driver) ─
// One analysis = N passes sharing kAnalysisId; each pass is labeled with its
// 0-based kPassIndex out of kPassCount. init() reads them into job_start.
constexpr const char* kAnalysisId = "GPUFL_ANALYSIS_ID";
constexpr const char* kPassIndex  = "GPUFL_PASS_INDEX";
constexpr const char* kPassCount  = "GPUFL_PASS_COUNT";

// ── Deep engine knobs ───────────────────────────────────────────────────────
constexpr const char* kDeepPcOnly  = "GPUFL_DEEP_PC_ONLY";
constexpr const char* kDeepTryBoth = "GPUFL_DEEP_TRY_BOTH";

// ── SASS metrics knobs ──────────────────────────────────────────────────────
constexpr const char* kSassMetricsOnly              = "GPUFL_SASS_METRICS_ONLY";
constexpr const char* kSassForceSafeActivity        = "GPUFL_SASS_FORCE_SAFE_ACTIVITY";
constexpr const char* kSassAllowFullActivity        = "GPUFL_SASS_ALLOW_FULL_ACTIVITY";
constexpr const char* kSassAllowKernelActivity      = "GPUFL_SASS_ALLOW_KERNEL_ACTIVITY";
constexpr const char* kSassAllowMarkerActivity      = "GPUFL_SASS_ALLOW_MARKER_ACTIVITY";
constexpr const char* kSassAllowMemTransferActivity = "GPUFL_SASS_ALLOW_MEM_TRANSFER_ACTIVITY";
constexpr const char* kSassAllowMemory2Activity     = "GPUFL_SASS_ALLOW_MEMORY2_ACTIVITY";
constexpr const char* kSassAllowMemoryActivity      = "GPUFL_SASS_ALLOW_MEMORY_ACTIVITY";
constexpr const char* kSassAllowSyncActivity        = "GPUFL_SASS_ALLOW_SYNC_ACTIVITY";
constexpr const char* kSassAllowGraphActivity       = "GPUFL_SASS_ALLOW_GRAPH_ACTIVITY";
constexpr const char* kSassAllowExternalCorrelation = "GPUFL_SASS_ALLOW_EXTERNAL_CORRELATION";
constexpr const char* kDisableCubinCapture          = "GPUFL_DISABLE_CUBIN_CAPTURE";
constexpr const char* kSassDisableCubinCapture      = "GPUFL_SASS_DISABLE_CUBIN_CAPTURE";
constexpr const char* kSassExcludeArchs             = "GPUFL_SASS_EXCLUDE_ARCHS";
constexpr const char* kSassLazyPatching            = "GPUFL_SASS_LAZY_PATCHING";
constexpr const char* kSassDeferScopeFlush         = "GPUFL_SASS_DEFER_SCOPE_FLUSH";

// ── Standalone monitor daemon (daemon/monitor) ──────────────────────────────
constexpr const char* kMonitorApp        = "GPUFL_MONITOR_APP";
constexpr const char* kMonitorLogDir     = "GPUFL_MONITOR_LOG_DIR";
constexpr const char* kMonitorIntervalMs = "GPUFL_MONITOR_INTERVAL_MS";

// ── External / platform names GPUFlight reads or sets ───────────────────────
// Not our knobs (CUDA-driver / dynamic-loader contracts) but referenced from
// specific spots — centralized so the exact spelling lives in one place.
constexpr const char* kCudaModuleLoading   = "CUDA_MODULE_LOADING";
constexpr const char* kCudaInjection64Path = "CUDA_INJECTION64_PATH";
constexpr const char* kNvtxInjection64Path = "NVTX_INJECTION64_PATH";
constexpr const char* kCudaPath            = "CUDA_PATH";
constexpr const char* kLdPreload           = "LD_PRELOAD";

}  // namespace gpufl::env
