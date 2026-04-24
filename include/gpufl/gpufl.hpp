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
     * "http://localhost:8080". Used by:
     *   - log upload (when {@link remote_upload} is true) → POSTs to
     *     `<backend_url>/api/v1/events/<type>`.
     *   - remote named-config fetch (when {@link config_name} is set) →
     *     GETs `<backend_url>/api/v1/config?config=<name>`.
     *
     * Setting this alone does nothing; you must also opt into at least
     * one of the two capabilities via `remote_upload` or `config_name`.
     */
    std::string backend_url = "";

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
     * `<backend_url>/api/v1/config?config=<name>` and applies the
     * returned field overrides to this InitOptions BEFORE the monitor
     * is initialized. Empty means "no remote fetch" — your local
     * config wins without any network round-trip.
     */
    std::string config_name = "";

    /**
     * When true, gpufl::init() attaches an HttpLogSink to the logger so
     * every NDJSON line is POSTed directly to the backend at
     * {@code <backend_url>/api/v1/events/<type>} using
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
