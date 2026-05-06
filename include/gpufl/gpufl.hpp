#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "gpufl/core/monitor.hpp"

namespace gpufl {
enum class BackendKind { Auto, Nvidia, Amd, None };

struct InitOptions {
    std::string app_name = "gpufl";
    std::string log_path = "";  // if empty, will default to "<app>.log"
    int system_sample_rate_ms =
        0;  // currently less than 50-100 would not be effective.
    int kernel_sample_rate_ms = 0;
    BackendKind backend = BackendKind::Auto;
    bool sampling_auto_start = false;
    bool enable_kernel_details = false;
    bool enable_debug_output = false;
    bool enable_stack_trace = false;
    bool enable_source_collection = true;  // collect source file content for source/SASS correlation
    bool flush_logs_always = false;
    // Enable CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION so frameworks
    // (PyTorch's torch.profiler, TF's profile.trace, JAX, XLA) that
    // bracket their ops with cuptiActivityPushExternalCorrelationId can
    // tag every kernel with their op id. When the framework isn't
    // pushing, CUPTI emits zero records — no overhead. Default-on
    // because it's both useful and free in the absence of frameworks;
    // disable only if running on a CUPTI version that errors on the
    // kind (logs a soft-warning if so, doesn't crash).
    bool enable_external_correlation = true;
    // Enable CUPTI_ACTIVITY_KIND_SYNCHRONIZATION so that every
    // cudaStreamSynchronize / cudaDeviceSynchronize / cudaEventSynchronize
    // / cuStreamWaitEvent call gets a wall-clock timed record. The
    // primary insight: time spent here = host blocked on GPU = a
    // direct measure of GPU underutilization. Default-on because the
    // overhead is small (one record per sync call, mid volume) and
    // the answer it unlocks ("X% of your wall time is `cudaStreamSync`")
    // is a top-five most-asked question. If a workload performs
    // millions of synchronizations and the volume becomes a problem,
    // disable this flag — the rest of the pipeline keeps working.
    bool enable_synchronization = true;
    // Enable CUPTI_ACTIVITY_KIND_MEMORY2 to capture cudaMalloc /
    // cudaFree / cudaMallocAsync / cudaMallocManaged / cudaMallocHost
    // with timing, address, size, and kind. **Default-off** in v1
    // because TensorFlow eager mode and similar can produce a high
    // record rate, and we want a couple of internal sessions to
    // validate overhead before flipping it on by default. PyTorch
    // workloads with the caching allocator typically generate <1k
    // events per session, so it's safe to enable manually for those
    // (and that's why we surface the toggle prominently rather than
    // burying it in a config file).
    bool enable_memory_tracking = false;
    // Enable CUPTI_ACTIVITY_KIND_GRAPH_TRACE to capture per-launch
    // timing for cudaGraphLaunch calls. **Default-off** in v1 because
    // CUDA Graphs interact with PC Sampling on some Blackwell driver
    // builds (graph launch can reset the sampling buffer). Opt-in
    // until we've validated on a couple of internal sessions; will
    // flip to default-on in a follow-up release.
    //
    // Cost when off: zero — we never call cuptiActivityEnable for
    // the kind. Cost when on: one record per cudaGraphLaunch (very
    // low volume).
    bool enable_cuda_graphs_tracking = false;
    // Default: PC Sampling alone. The cubin-disassembly pipeline
    // (ResourceHandler + nvdisasm) already provides SASS listings and
    // source correlation, so PC Sampling gives users the full SASS view
    // plus stall reasons. Opt into SassMetrics or PcSamplingWithSass only
    // when per-instruction execution/divergence counters are needed —
    // those paths use CUPTI kernel replay and are ~100× more expensive
    // on pre-sm_120 GPUs.
    ProfilingEngine profiling_engine = ProfilingEngine::PcSampling;

    // ── Configuration sources, in precedence order (low → high) ────────────
    //
    //   1. InitOptions defaults (these field initializers)
    //   2. Remote named config (opt-in: requires backend_url + api_key + config_name)
    //   3. Local config file (if config_file path is set)
    //   4. Programmatic overrides (env vars, auto-tuning in gpufl::init)
    //   5. The caller's explicit field sets in InitOptions
    //
    // Source (2) only runs when `config_name` is non-empty — merely
    // setting `backend_url` + `api_key` does NOT trigger a fetch. This
    // keeps the config-merge path predictable: you opt into remote
    // config by name.

    /** Path to a local JSON config file. See ConfigFileLoader. */
    std::string config_file = "";

    /**
     * GPUFlight backend base URL — e.g. "https://api.gpuflight.com" or
     * "http://localhost:8080". Host-only; do NOT include the API
     * prefix (use {@link api_path} for that). Used by:
     *   - log upload (when {@link remote_upload} is true) → POSTs to
     *     `<backend_url><api_path>/events/<type>`.
     *   - remote named-config fetch (when {@link config_name} is set) →
     *     GETs `<backend_url><api_path>/config?config=<name>`.
     *
     * Setting this alone does nothing; you must also opt into at least
     * one of the two capabilities via `remote_upload` or `config_name`.
     */
    std::string backend_url = "";

    /**
     * Override for the URL path prefix the agent uses when calling
     * the backend. Empty (default) → the client uses the version it
     * was compiled against (currently `/api/v1`, see
     * {@code gpufl::kDefaultApiPath} in {@code core/version.hpp}).
     *
     * Set this only if you're running the backend behind a reverse
     * proxy / API gateway that mounts it at a non-root path (e.g.
     * `/profiler/api/v1`). It does NOT let you choose between API
     * versions — the client library can only emit one wire format,
     * and pointing it at a path that speaks a different one will
     * just produce parse errors.
     *
     * Normalization: leading slash is added if missing; trailing
     * slashes are stripped; empty → default. Env-var override:
     * {@code GPUFL_API_PATH}.
     */
    std::string api_path = "";

    /**
     * API key used for BOTH remote config fetch and log upload (for v1).
     * Sent as `X-API-Key` on the config GET and
     * `Authorization: Bearer <key>` on event POSTs — matching the
     * existing backend auth paths. May split later if config and
     * ingestion need independent keys.
     */
    std::string api_key = "";

    /**
     * Name of the remote config to fetch (e.g. "production", "debug").
     * When non-empty AND both {@link backend_url} and {@link api_key}
     * are set, `gpufl::init()` performs a synchronous HTTP GET against
     * `<backend_url><api_path>/config?config=<name>` and applies the
     * returned field overrides to this InitOptions BEFORE the monitor
     * is initialized. Empty means "no remote fetch" — your local
     * config wins without any network round-trip.
     */
    std::string config_name = "";

    /**
     * When true, gpufl::init() attaches an HttpLogSink to the logger so
     * every NDJSON line is POSTed directly to the backend at
     * {@code <backend_url><api_path>/events/<type>} using
     * {@code Authorization: Bearer <api_key>}.
     *
     * Intended for interactive contexts (local dev, SSH, Jupyter) where
     * deploying the monitor daemon is heavy. The file-based NDJSON logs
     * are still written in parallel, so no data is lost if the backend
     * is unreachable — the agent daemon (or a manual upload tool) can
     * back-fill later.
     *
     * Requires both {@link backend_url} and {@link api_key} to be set;
     * ignored otherwise. Env-var override: {@code GPUFL_REMOTE_UPLOAD=1}.
     */
    bool remote_upload = false;
};

struct BackendProbeResult {
    bool available;
    std::string reason;
};

extern std::atomic<int> g_systemSampleRateMs;
extern InitOptions g_opts;

BackendProbeResult probeNvml();
BackendProbeResult probeRocm();

void systemStart(std::string name = "system");
void systemStop(std::string name = "system");

// F1 (External Correlation) — active push/pop for callers that want to
// tag CUDA work with an op id WITHOUT relying on a framework profiler
// being active. Used by `gpufl.torch.attach()` to stamp every aten
// dispatch's kernels with a stable id derived from the op name.
//
// `kind` follows CUPTI's CUpti_ExternalCorrelationKind enum:
//   1 = UNKNOWN, 2 = OPENACC, 3 = CUSTOM0 (used by torch.profiler),
//   4 = CUSTOM1 (used by gpufl.torch.attach), 5 = CUSTOM2.
//
// Calls are no-ops on non-CUPTI platforms (AMD ROCm, CPU-only fallback)
// — safe to call from cross-platform code.
void pushExternalCorrelation(uint32_t kind, uint64_t id);
void popExternalCorrelation(uint32_t kind);

// Start global runtime. Returns true on success.
bool init(const InitOptions& opts);

// Stop runtime, flush and close logs.
void shutdown();

// Generate a text report from the log files written during this session.
// Call after shutdown().
// - No argument: prints the report to console (stdout).
// - With output_path: saves the report to a file.
void generateReport(const std::string& output_path = "");

class ScopedMonitor {
   public:
    explicit ScopedMonitor(std::string name, std::string tag, bool deep_profiling);
    explicit ScopedMonitor(std::string name, std::string tag);
    explicit ScopedMonitor(std::string name, bool deep_profiling);
    explicit ScopedMonitor(std::string name);
    ~ScopedMonitor();

    ScopedMonitor(const ScopedMonitor&) = delete;
    ScopedMonitor& operator=(const ScopedMonitor&) = delete;

   private:
    std::string name_;
    std::string tag_;
    int pid_{0};
    int64_t start_ns_{0};
    uint64_t scope_id_{};
};

inline void monitor(const std::string& name, const std::function<void()>& fn) {
    ScopedMonitor r(name);
    fn();
}
inline void monitor(const std::string& name, const std::string& tag,
                    const std::function<void()>& fn) {
    ScopedMonitor r(name, tag);
    fn();
}
}  // namespace gpufl

#define GFL_SCOPE(name) if (gpufl::ScopedMonitor _gpufl_scope{name}; true)

#define GFL_SCOPE_TAGGED(name, tag, deep_profiling) \
    if (gpufl::ScopedMonitor _gpufl_scope{name, tag}; true)

#define GFL_SYSTEM_START(name) ::gpufl::systemStart(name)
#define GFL_SYSTEM_STOP(name) ::gpufl::systemStop(name)
