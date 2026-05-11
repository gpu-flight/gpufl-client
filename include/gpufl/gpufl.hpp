#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gpufl/core/monitor.hpp"

namespace gpufl {
enum class BackendKind { Auto, Nvidia, Amd, None };

struct InitOptions {
    std::string app_name = "gpufl";
    std::string log_path = "";  // if empty, will default to "<app>.log"
    int system_sample_rate_ms =
        0;  // currently less than 50-100 would not be effective.
    // DEPRECATED (1.0.1): no longer has any effect. It used to throttle
    // kernel activity-record processing, but that corrupted kernel timing
    // (see the note in kernel_launch_handler.cpp). All kernel activity
    // records are now always processed. Kept only so existing callers and
    // config files don't break; will be removed in a future major release.
    int kernel_sample_rate_ms = 0;
    BackendKind backend = BackendKind::Auto;
    /**
     * System-metric sampling policy.
     *
     *   true  — Sampler runs continuously from init() to shutdown().
     *           GPU/host telemetry events are emitted on a fixed interval
     *           regardless of scopes. Use for fleet monitoring, dashboards,
     *           any "always-on" use case.
     *
     *   false — Sampler is idle by default. It activates while inside any
     *           GFL_SCOPE region (auto-bracketing), or between explicit
     *           gpufl::systemStart() / systemStop() calls. Outside those
     *           windows zero system-metric events are emitted. Use when
     *           you only care about telemetry during specific phases of
     *           your application (e.g. inside a training-step scope).
     *
     * Renamed from `sampling_auto_start` in this release. The old name
     * referred only to init-time auto-start and didn't capture the
     * scope-bracketing behavior, which was silently broken when the
     * flag was off. C++ callers must rename to `continuous_system_sampling`;
     * the compiler will surface this as a "no member named
     * 'sampling_auto_start'" error pointing at their call site. Python
     * callers using the old kwarg name keep working for one release
     * with a DeprecationWarning (see python/gpufl/__init__.py).
     */
    bool continuous_system_sampling = false;
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
    // Default: Monitor — telemetry only (GPU/host metrics), no CUPTI
    // kernel capture. The safe, lowest-overhead default: it can't hit
    // the CUPTI kernel-path overhead (PyTorch training saw +650% under
    // PC Sampling) or the kernel-path crash/hang modes, and it's all a
    // "just watch my GPU" user needs.
    //
    // Opt in explicitly for tracing/profiling, in order of cost:
    //   Trace         — per-op timeline (kernels, memcpy, sync)
    //   PcSampling    — + stall-reason sampling ("why is it slow")
    //   SassMetrics   — + per-instruction counters (CUPTI replay,
    //                   ~100× costlier on pre-sm_120 GPUs)
    //   PmSampling    — time-series hardware PM samples
    //   RangeProfiler — scope-level hardware throughput counters
    //   Deep          — PcSampling + SassMetrics in one run
    ProfilingEngine profiling_engine = ProfilingEngine::Monitor;

    uint32_t pm_sampling_interval_us = 100;
    uint32_t pm_sampling_max_samples = 4096;
    std::string pm_sampling_preset = "overview";
    std::vector<std::string> pm_sampling_metrics;
    bool pm_sampling_scope_only = true;

    // ── Configuration sources, in precedence order (low → high) ────────────
    //
    //   1. InitOptions defaults (these field initializers)
    //   2. Local config file (if config_file path is set)
    //   3. Programmatic overrides (env vars, auto-tuning in gpufl::init)
    //   4. The caller's explicit field sets in InitOptions
    //

    /** Path to a local JSON config file. See ConfigFileLoader. */
    std::string config_file = "";

    /**
     * GPUFlight backend base URL — e.g. "https://api.gpuflight.com" or
     * "http://localhost:8080". Host-only; do NOT include the API
     * prefix (use {@link api_path} for that).
     *
     * Read by:
     *   - the version-discovery probe at init time
     *   - {@link gpufl::uploadLogs} when you call it post-shutdown
     *     (stored on InitOptions so the deferred upload path can pick
     *     it up without the caller having to re-supply it)
     *
     * Setting this alone does nothing — no HTTP runs during a session.
     * Upload is a separate step (`gpufl::uploadLogs`, or the
     * `gpufl.session()` Python context manager).
     *
     * **DEPRECATION NOTE (v1.2 removal)**: this field — together with
     * {@link api_key} and {@link remote_upload} — is planned for
     * removal in v1.2. Long-term, all backend creds will be passed
     * directly to {@link gpufl::uploadLogs} (and the version probe
     * will read `GPUFL_BACKEND_URL` directly). The fields stay
     * functional in v1.1 to keep the migration painless; if you're
     * starting fresh, prefer passing creds straight to `UploadOptions`.
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
     * API key for log upload to the GPUFlight backend.
     * Sent as `Authorization: Bearer <key>` on event POSTs.
     *
     * **DEPRECATION NOTE (v1.2 removal)**: planned for removal in v1.2
     * together with {@link backend_url} and {@link remote_upload}.
     * Long-term, pass creds directly to {@link gpufl::uploadLogs}.
     */
    std::string api_key = "";

    /**
     * **DEPRECATED — v1.1 backward-compat shim, removed in v1.2.**
     *
     * Previously attached an HttpLogSink that POSTed NDJSON events live
     * during the session. That mechanism is gone in v1.1. To preserve
     * the old "set one flag and forget" UX, this field now triggers an
     * **automatic** call to {@link gpufl::uploadLogs} at the end of
     * {@link gpufl::shutdown}, using {@link backend_url} +
     * {@link api_key} from this InitOptions.
     *
     * Behavior with `remote_upload = true`:
     *   - **At init()**: logs a deprecation message pointing at the
     *     new API. No HTTP is opened.
     *   - **At shutdown()**: after the file sink has flushed and
     *     closed, `gpufl::uploadLogs(uopts)` is invoked synchronously
     *     with creds copied from InitOptions. Failures are logged but
     *     never thrown — the process exits cleanly either way.
     *   - **In Python**: the wrapper emits a `DeprecationWarning` and
     *     also schedules upload via `atexit`, so notebook / script
     *     callers who never explicitly call shutdown() still get the
     *     same delivery.
     *
     * Wall-time impact: the legacy live-streaming model amortized HTTP
     * work across the session; the new shim does it all at shutdown.
     * Expect shutdown to take seconds-to-minutes proportional to log
     * volume. To avoid the wait or to control timing, drop the flag
     * and call `gpufl::uploadLogs(uopts)` directly when convenient.
     *
     * In v1.2 this field — together with {@link backend_url} and
     * {@link api_key} — is removed; creds move entirely onto
     * `UploadOptions`. See `include/gpufl/upload/upload_logs.hpp`.
     */
    bool remote_upload = false;

    /**
     * Global kill switch. When false, {@link gpufl::init} returns false
     * immediately without spawning any backend, opening a logger, or
     * touching CUPTI / NVML / ROCm. Every other public entry point —
     * {@link gpufl::shutdown}, {@link gpufl::systemStart} /
     * {@link gpufl::systemStop}, every {@link ScopedMonitor} ctor /
     * dtor — short-circuits automatically because they all guard on
     * `runtime() != nullptr` and a disabled init never allocates one.
     *
     * Two equivalent ways to flip the switch:
     *   - Set this field to false in code.
     *   - Set the env var `GPUFL_DISABLED` to a truthy value
     *     (`1`, `true`, `yes`, `on`; case-insensitive). The env var
     *     takes precedence over this field, so you can disable gpufl
     *     for a one-off run without editing code:
     *     `GPUFL_DISABLED=1 ./my_app`.
     *
     * Use case: toggle gpufl on/off across runs without #ifdef'ing out
     * the call sites or building a separate "no-gpufl" binary. The
     * disabled path has effectively zero overhead — early-return at
     * init, then null-runtime no-ops at every other call site.
     */
    bool enabled = true;
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

// Comprehensive-profile defaults for "injection" capture mode (the
// libgpufl_inject.so launcher path). Flips most observability flags on
// and selects PcSamplingWithSass for the richest per-instruction view.
// Documented overhead: ~5–15% wall time on heavy CUDA workloads.
//
// The launcher's `--profile` flag picks between this and
// `light_mode_default_options()`; everything else is layered on top
// (env-var overrides + remote named config) inside gpufl::init().
inline InitOptions injection_mode_default_options() {
    InitOptions opts;
    opts.app_name                    = "gpufl-trace";
    opts.sampling_auto_start         = true;
    opts.enable_kernel_details       = true;
    opts.enable_stack_trace          = true;
    opts.enable_source_collection    = true;
    opts.enable_synchronization      = true;
    opts.enable_memory_tracking      = true;
    opts.enable_cuda_graphs_tracking = true;
    opts.enable_external_correlation = true;
    opts.flush_logs_always           = false;
    opts.profiling_engine            = ProfilingEngine::PcSamplingWithSass;
    opts.system_sample_rate_ms       = 100;
    opts.kernel_sample_rate_ms       = 1000;
    opts.remote_upload               = false;
    return opts;
}

// Lower-overhead injection profile: skips source correlation, memory
// tracking, CUDA-graphs tracking; drops to plain PcSampling; raises
// sample intervals ~5x. Roughly ~1–3% overhead on the same workloads.
inline InitOptions light_mode_default_options() {
    InitOptions opts = injection_mode_default_options();
    opts.enable_source_collection    = false;
    opts.enable_memory_tracking      = false;
    opts.enable_cuda_graphs_tracking = false;
    opts.profiling_engine            = ProfilingEngine::PcSampling;
    opts.system_sample_rate_ms       = 500;
    opts.kernel_sample_rate_ms       = 5000;
    return opts;
}

// Generate a text report from the log files written during this session.
// Call after shutdown().
// - No argument: prints the report to console (stdout).
// - With output_path: saves the report to a file.
void generateReport(const std::string& output_path = "");

// Single options object carrying everything you might attach to a
// scope beyond its name. Used as the canonical 2nd argument of the
// 1.0.3+ `ScopedMonitor(name, meta)` ctor and the unified
// `GFL_SCOPE(name, ...)` macro. Designed as an aggregate so callers
// can use designated initializers:
//
//     GFL_SCOPE("hot", .repeat=10, .warmup=3, .tag="ml")
//
// Fields are declared in init-list order — designated initializers
// must list them in the same order (`.tag` before `.repeat` before
// `.warmup`).  Any field can be omitted; defaults all zero / empty.
//
// New fields can be appended without breaking existing callers
// (designated init only sets named members; aggregate default
// initializers cover the rest).
struct ScopeMeta {
    // Optional category tag (e.g. "math", "ml", "io"). When non-empty,
    // replaces the legacy `tag` parameter of ScopedMonitor / Python's
    // `Scope(name, tag)` so the API has a single source of truth.
    std::string tag = {};
    // Number of MEASURED iterations bracketed by this scope. Lets the
    // analyzer divide total scope time by `repeat` to get per-iter avg
    // without the caller pre-computing it. 0 = not provided.
    uint32_t repeat = 0;
    // Iterations the caller ran BEFORE opening this scope, deliberately
    // excluded from measurement (JIT compile / cold cache warmup). 0 = none.
    // Stored for audit / reporting; not used in per-iter math.
    uint32_t warmup = 0;

    // Fluent builders — the portable way to populate ScopeMeta in
    // pure C++17. Designated initializers (`{.repeat=10}`) require
    // C++20 (or GCC/Clang in C++17 mode with extensions); MSVC's
    // /std:c++17 rejects them outright. The builders work on every
    // compiler.
    //
    //     gpufl::ScopeMeta{}.setRepeat(10).setWarmup(3).setTag("ml")
    //
    // Each setter returns `*this` so calls chain. Use them with
    // GFL_BENCH or pass the result directly to ScopedMonitor.
    ScopeMeta& setTag(std::string t)    { tag    = std::move(t); return *this; }
    ScopeMeta& setRepeat(uint32_t r)    { repeat = r;            return *this; }
    ScopeMeta& setWarmup(uint32_t w)    { warmup = w;            return *this; }
};

class ScopedMonitor {
   public:
    explicit ScopedMonitor(std::string name, std::string tag, bool deep_profiling);
    explicit ScopedMonitor(std::string name, std::string tag);
    explicit ScopedMonitor(std::string name, bool deep_profiling);
    explicit ScopedMonitor(std::string name);
    // Canonical 1.0.3+ ctor — single options object. `meta.tag`
    // replaces the legacy separate `tag` parameter; the older
    // (name, tag, ScopeMeta) overload has been retired in favor of
    // this one so there's a single source of truth for the tag.
    explicit ScopedMonitor(std::string name, ScopeMeta meta);
    ~ScopedMonitor();

    ScopedMonitor(const ScopedMonitor&) = delete;
    ScopedMonitor& operator=(const ScopedMonitor&) = delete;

   private:
    // Shared ctor body — all constructors funnel through this so the
    // begin-row push, NVTX push, and profiler-scope hooks live in one
    // place. `meta` defaults to ScopeMeta{} (zeros) for the legacy
    // overloads, so their wire output is byte-for-byte unchanged.
    void init_(const ScopeMeta& meta);

    std::string name_;
    std::string tag_;
    int pid_{0};
    int64_t start_ns_{0};
    uint64_t scope_id_{};
    // True if this scope took an activation on the sampler (i.e.,
    // continuous_system_sampling was off at scope entry and we kicked
    // off system metrics for the scope duration). The destructor uses
    // this to decide whether to deactivate symmetrically. Avoids
    // re-reading g_opts in the destructor, which could see a different
    // value if shutdown() raced ahead.
    bool sampler_activated_{false};
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

namespace detail {

// Helper that powers GFL_BENCH. The trailing lambda body is delivered
// via operator+=, which lets the macro expansion look like a natural
// `{ block }` after `GFL_BENCH(...)`. The body runs `warmup` times
// BEFORE the scope opens (excluded from measurement) and `repeat`
// times INSIDE the scope (bracketed by the BEGIN / END rows).
//
// Body is captured by reference via the `[&]` lambda the macro
// produces, so it sees all enclosing locals — but be aware that
// `return` inside the body returns from the LAMBDA, not the
// enclosing function. Use exceptions for fatal errors instead.
class BenchInvoker {
   public:
    BenchInvoker(std::string name, ScopeMeta meta)
        : name_(std::move(name)), meta_(std::move(meta)) {}

    // Precedence note: `operator+=` was picked over `*` / `<<` because
    // it has lower precedence than the lambda's `[]` capture syntax,
    // so the macro doesn't need extra parens around the lambda.
    template <class Body>
    void operator+=(Body&& body) {
        // Warmup phase: open a "<name>_warmup" sub-scope so the kernel
        // events emitted during warmup are attributed to a separately
        // identifiable bucket in the log instead of leaking into the
        // global / outer scope. The warmup scope itself carries
        // repeat=meta_.warmup so the analyzer's per-iteration math
        // works for cold-start cost too. Tag inherits from the user's
        // meta so warmup and measured group together (e.g. both
        // tagged "ml" if the user passed that).
        if (meta_.warmup > 0) {
            ScopeMeta warmup_meta;
            warmup_meta.tag    = meta_.tag;
            warmup_meta.repeat = meta_.warmup;
            ScopedMonitor warmup_scope(name_ + "_warmup",
                                       std::move(warmup_meta));
            for (uint32_t i = 0; i < meta_.warmup; ++i) body();
        }
        // Measured phase: open the main "<name>" scope.
        ScopedMonitor scope(name_, meta_);
        for (uint32_t i = 0; i < meta_.repeat; ++i) body();
    }

   private:
    std::string name_;
    ScopeMeta meta_;
};

}  // namespace detail
}  // namespace gpufl

// GFL_SCOPE — single-arg scoped instrumentation, unchanged since
// pre-1.0.3. Pass ONLY a name. The body runs exactly once and the
// scope brackets all the work inside it.
//
//     GFL_SCOPE("inference") { model_forward(x); }
//
// For benchmark loops with automatic warmup + repeat, see GFL_BENCH
// below. For attaching metadata (tag / repeat / warmup) to a scope
// without using the auto-loop helper — e.g. you have an irregular
// loop body with `continue` / `break` / early-return semantics that
// don't fit in a lambda — construct ScopedMonitor directly:
//
//     gpufl::ScopeMeta meta;
//     meta.repeat = 10;
//     gpufl::ScopedMonitor scope("hot", meta);
//     for (int i = 0; i < 10; ++i) { ... }
#define GFL_SCOPE(name) if (gpufl::ScopedMonitor _gpufl_scope{name}; true)

#define GFL_SCOPE_TAGGED(name, tag, deep_profiling) \
    if (gpufl::ScopedMonitor _gpufl_scope{name, tag}; true)

// GFL_BENCH — automatic benchmark loop. The body block runs
// `meta.warmup` times BEFORE the scope opens and `meta.repeat` times
// INSIDE the scope (BEGIN row carries both counts). Saves you from
// writing two for-loops + a manual ScopedMonitor:
//
//     GFL_BENCH("hot", gpufl::ScopeMeta{}.setRepeat(10).setWarmup(3)) {
//         my_kernel<<<grid, block>>>(...);
//         cudaDeviceSynchronize();
//     };   // ← trailing ';' is required
//
// The builder form (`ScopeMeta{}.setX(...)`) is recommended because
// it compiles on every major compiler in /std:c++17. Designated
// initializers (`{.repeat=10, .warmup=3}`) work on GCC and Clang in
// C++17 (compiler extension) and on every compiler in C++20, but
// MSVC's /std:c++17 explicitly rejects them — so projects targeting
// MSVC pre-C++20 should stick with builders.
//
// The body is captured by `[&]` so it sees enclosing locals.
//
// IMPORTANT: `return` inside the body returns from the LAMBDA, not
// the enclosing function. Macros that expand to early-return
// (e.g. CHECK_CUDA) won't propagate errors out — throw an exception
// instead, or check status after the macro completes.
#define GFL_BENCH(name, ...) \
    ::gpufl::detail::BenchInvoker{name, ::gpufl::ScopeMeta{__VA_ARGS__}} \
        += [&]()

#define GFL_SYSTEM_START(name) ::gpufl::systemStart(name)
#define GFL_SYSTEM_STOP(name) ::gpufl::systemStop(name)
