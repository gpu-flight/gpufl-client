#include <stdexcept>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gpufl/gpufl.hpp"
#include "gpufl/upload/upload_logs.hpp"
#if GPUFL_HAS_CUDA
#endif

namespace py = pybind11;

class PyScope {
public:
    // `repeat` / `warmup` are benchmark metadata stamped onto the BEGIN
    // row of the underlying scope_event_batch when the scope opens.
    // Both default to 0 — the legacy `with gpufl.Scope("x"):` call path
    // produces byte-identical output to pre-1.0.3.
    PyScope(std::string name, std::string tag, uint32_t repeat, uint32_t warmup)
        : name_(std::move(name)),
          tag_(std::move(tag)),
          repeat_(repeat),
          warmup_(warmup) {}

    void enter() {
        if (repeat_ == 0 && warmup_ == 0) {
            // Legacy fast path — no ScopeMeta allocation needed.
            monitor_ = std::make_unique<gpufl::ScopedMonitor>(name_, tag_);
        } else {
            // 1.0.3+ canonical ctor: tag now lives inside ScopeMeta.
            gpufl::ScopeMeta meta;
            meta.tag    = tag_;
            meta.repeat = repeat_;
            meta.warmup = warmup_;
            monitor_ = std::make_unique<gpufl::ScopedMonitor>(name_, meta);
        }
    }

    void exit(py::object exc_type, py::object exc_value, py::object traceback) {
        monitor_.reset();
    }

private:
    std::string name_;
    std::string tag_;
    uint32_t repeat_{0};
    uint32_t warmup_{0};
    std::unique_ptr<gpufl::ScopedMonitor> monitor_;
};

PYBIND11_MODULE(_gpufl_client, m) {
    m.doc() = "GPUFL Internal C++ Binding";

    // BackendKind has a "None" value, which is a Python keyword —
    // accessible as BackendKind.__members__["None"] but ugly. Add a
    // `None_` alias so users can write `gpufl.BackendKind.None_`. The
    // underscore-suffix is the standard Python convention for
    // keyword-clashing names (mirrors pybind11's `class_`, `type_`).
    // (ProfilingEngine no longer has a "None" member — its floor is
    // `Monitor` — so it needs no such alias.)
    auto backendKindEnum = py::enum_<gpufl::BackendKind>(m, "BackendKind")
        .value("Auto",   gpufl::BackendKind::Auto)
        .value("Nvidia", gpufl::BackendKind::Nvidia)
        .value("Amd",    gpufl::BackendKind::Amd)
        .value("None",   gpufl::BackendKind::None)
        .export_values();
    backendKindEnum.attr("None_") = backendKindEnum.attr("__members__")[py::str("None")];

    // Six canonical names, one per level, no aliases — see the enum in
    // monitor.hpp for the naming rationale (clarity-first, plain intent
    // with the precise CUPTI term where it's the searchable one).
    py::enum_<gpufl::ProfilingEngine>(m, "ProfilingEngine")
        .value("Monitor",       gpufl::ProfilingEngine::Monitor)
        .value("Trace",         gpufl::ProfilingEngine::Trace)
        .value("PcSampling",    gpufl::ProfilingEngine::PcSampling)
        .value("SassMetrics",   gpufl::ProfilingEngine::SassMetrics)
        .value("PmSampling",    gpufl::ProfilingEngine::PmSampling)
        .value("RangeProfiler", gpufl::ProfilingEngine::RangeProfiler)
        .value("Deep",          gpufl::ProfilingEngine::Deep)
        .export_values();

    py::class_<gpufl::InitOptions>(m, "InitOptions")
        .def(py::init<>())
        .def_readwrite("app_name",              &gpufl::InitOptions::app_name)
        .def_readwrite("log_path",              &gpufl::InitOptions::log_path)
        .def_readwrite("continuous_system_sampling", &gpufl::InitOptions::continuous_system_sampling)
        .def_readwrite("system_sample_rate_ms", &gpufl::InitOptions::system_sample_rate_ms)
        .def_readwrite("kernel_sample_rate_ms", &gpufl::InitOptions::kernel_sample_rate_ms)
        .def_readwrite("backend",               &gpufl::InitOptions::backend)
        // `enable_kernel_details` binding removed May 2026 — kernel
        // grid/block details are now always captured. See gpufl.hpp.
        .def_readwrite("enable_debug_output",   &gpufl::InitOptions::enable_debug_output)
        .def_readwrite("enable_stack_trace",    &gpufl::InitOptions::enable_stack_trace)
        .def_readwrite("enable_source_collection", &gpufl::InitOptions::enable_source_collection)
        // feature gates — surfaced on the InitOptions class so
        // power users can tweak them via the dataclass-style API.
        .def_readwrite("enable_external_correlation", &gpufl::InitOptions::enable_external_correlation)
        .def_readwrite("enable_synchronization",      &gpufl::InitOptions::enable_synchronization)
        .def_readwrite("enable_memory_tracking",      &gpufl::InitOptions::enable_memory_tracking)
        .def_readwrite("enable_cuda_graphs_tracking", &gpufl::InitOptions::enable_cuda_graphs_tracking)
        .def_readwrite("pm_sampling_interval_us", &gpufl::InitOptions::pm_sampling_interval_us)
        .def_readwrite("pm_sampling_max_samples", &gpufl::InitOptions::pm_sampling_max_samples)
        .def_readwrite("pm_sampling_preset", &gpufl::InitOptions::pm_sampling_preset)
        .def_readwrite("pm_sampling_metrics", &gpufl::InitOptions::pm_sampling_metrics)
        .def_readwrite("pm_sampling_scope_only", &gpufl::InitOptions::pm_sampling_scope_only)
        .def_readwrite("profiling_engine",      &gpufl::InitOptions::profiling_engine)
        // Backend interactions — backend_url is the BASE URL of the
        // GPUFlight backend. Upload is a separate post-shutdown step
        // via gpufl.upload_logs() / gpufl.session(); nothing on
        // InitOptions controls upload directly anymore.
        // (The historical `config_name` / remote-config-fetch binding
        // was removed along with the backend's ConfigController.)
        .def_readwrite("backend_url",           &gpufl::InitOptions::backend_url)
        .def_readwrite("api_key",               &gpufl::InitOptions::api_key)
        // DEPRECATED in v1.1, planned removal v1.2 alongside
        // backend_url + api_key. See InitOptions docstring in
        // gpufl.hpp. Field is a no-op at the C++ level; the Python
        // wrapper around init() turns remote_upload=True into an
        // atexit-scheduled upload_logs() call for backward compat.
        .def_readwrite("remote_upload",         &gpufl::InitOptions::remote_upload)
        // Global kill switch — when false, gpufl::init returns false
        // immediately and every downstream call is a no-op via the
        // null-runtime guards. The high-level Python wrapper
        // (python/gpufl/__init__.py) also exposes this as the `enabled`
        // kwarg on its init() and short-circuits in Python before this
        // binding is even called. Exposing the field here keeps the raw
        // InitOptions surface consistent for callers who construct
        // InitOptions directly and call init(opts).
        .def_readwrite("enabled",               &gpufl::InitOptions::enabled);

    // Function-style init(). Every C++ InitOptions field that's user-facing
    // is surfaced as a keyword argument here so callers don't have to
    // construct an InitOptions object themselves. Pre-v0.1.1 the binding
    // also accepted legacy `enable_profiling` / `enable_perf_scope` bool
    // flags — those were removed in favor of the single `profiling_engine`
    // enum, which is strictly more expressive.
    m.def("init", [](std::string app_name,
                     std::string log_path,
                     bool continuous_system_sampling,
                     int system_sample_rate_ms,
                     int kernel_sample_rate_ms,
                     gpufl::BackendKind backend,
                     bool enable_debug_output,
                     bool enable_stack_trace,
                     bool enable_source_collection,
                     gpufl::ProfilingEngine profiling_engine,
                     std::string config_file,
                     std::string backend_url,
                     std::string api_key,
                     bool remote_upload,
                     bool enable_external_correlation,
                     bool enable_synchronization,
                     bool enable_memory_tracking,
                     bool enable_cuda_graphs_tracking,
                     uint32_t pm_sampling_interval_us,
                     uint32_t pm_sampling_max_samples,
                     std::string pm_sampling_preset,
                     std::vector<std::string> pm_sampling_metrics,
                     bool pm_sampling_scope_only) -> bool {

        gpufl::InitOptions opts;
        opts.app_name              = app_name;
        opts.log_path              = log_path;
        opts.continuous_system_sampling = continuous_system_sampling;
        opts.system_sample_rate_ms = system_sample_rate_ms;
        opts.kernel_sample_rate_ms = kernel_sample_rate_ms;
        opts.backend               = backend;
        opts.enable_debug_output   = enable_debug_output;
        opts.enable_stack_trace    = enable_stack_trace;
        opts.enable_source_collection = enable_source_collection;
        opts.profiling_engine      = profiling_engine;
        opts.config_file           = config_file;
        opts.backend_url           = std::move(backend_url);
        opts.api_key               = std::move(api_key);
        opts.remote_upload         = remote_upload;
        opts.enable_external_correlation = enable_external_correlation;
        opts.enable_synchronization      = enable_synchronization;
        opts.enable_memory_tracking      = enable_memory_tracking;
        opts.enable_cuda_graphs_tracking = enable_cuda_graphs_tracking;
        opts.pm_sampling_interval_us     = pm_sampling_interval_us;
        opts.pm_sampling_max_samples     = pm_sampling_max_samples;
        opts.pm_sampling_preset          = std::move(pm_sampling_preset);
        opts.pm_sampling_metrics         = std::move(pm_sampling_metrics);
        opts.pm_sampling_scope_only      = pm_sampling_scope_only;

        return gpufl::init(opts);
    }, py::arg("app_name"),
       py::arg("log_path")                    = "",
       py::arg("continuous_system_sampling")  = false,
       py::arg("system_sample_rate_ms")       = 0,
       py::arg("kernel_sample_rate_ms")       = 0,
       py::arg("backend")                     = gpufl::BackendKind::Auto,
       py::arg("enable_debug_output")         = false,
       py::arg("enable_stack_trace")          = false,
       py::arg("enable_source_collection")    = true,
       py::arg("profiling_engine")            = gpufl::ProfilingEngine::Monitor,
       py::arg("config_file")                 = "",
       py::arg("backend_url")                 = "",
       py::arg("api_key")                     = "",
       py::arg("remote_upload")               = false,
       py::arg("enable_external_correlation") = true,
       py::arg("enable_synchronization")      = true,
       py::arg("enable_memory_tracking")      = false,
       py::arg("enable_cuda_graphs_tracking") = false,
       py::arg("pm_sampling_interval_us")     = 100,
       py::arg("pm_sampling_max_samples")     = 4096,
       py::arg("pm_sampling_preset")          = "overview",
       py::arg("pm_sampling_metrics")         = std::vector<std::string>{},
       py::arg("pm_sampling_scope_only")      = true);

    m.def("system_start", [](std::string name) { gpufl::systemStart(std::move(name)); },
        py::arg("name") = "system");

    m.def("system_stop", [](std::string name) { gpufl::systemStop(std::move(name)); },
        py::arg("name") = "system");

    m.def("shutdown", &gpufl::shutdown);

    // ── Deferred bulk upload ────────────────────────────────────────────
    //
    // Replaces the in-process HttpLogSink streaming model. Call this
    // AFTER gpufl::shutdown() to ship the session's NDJSON files to the
    // backend in one bounded post-mortem step. Network failures here
    // cannot affect the host job's exit code or perceived performance.
    py::class_<gpufl::UploadOptions>(m, "UploadOptions")
        .def(py::init<>())
        .def_readwrite("log_path",                 &gpufl::UploadOptions::log_path)
        .def_readwrite("backend_url",              &gpufl::UploadOptions::backend_url)
        .def_readwrite("api_key",                  &gpufl::UploadOptions::api_key)
        .def_readwrite("api_path",                 &gpufl::UploadOptions::api_path)
        .def_readwrite("total_timeout_ms",         &gpufl::UploadOptions::total_timeout_ms)
        .def_readwrite("connect_timeout_ms",       &gpufl::UploadOptions::connect_timeout_ms)
        .def_readwrite("read_timeout_ms",          &gpufl::UploadOptions::read_timeout_ms)
        .def_readwrite("max_retries",              &gpufl::UploadOptions::max_retries)
        .def_readwrite("retry_delay_ms",           &gpufl::UploadOptions::retry_delay_ms)
        .def_readwrite("cursor_filename",          &gpufl::UploadOptions::cursor_filename)
        .def_readwrite("report_progress",          &gpufl::UploadOptions::report_progress)
        .def_readwrite("session_id_filter",        &gpufl::UploadOptions::session_id_filter)
        .def_readwrite("all_sessions",             &gpufl::UploadOptions::all_sessions)
        .def_readwrite("force",                    &gpufl::UploadOptions::force);

    py::class_<gpufl::UploadResult>(m, "UploadResult")
        .def_readonly("success",                 &gpufl::UploadResult::success)
        .def_readonly("files_processed",         &gpufl::UploadResult::files_processed)
        .def_readonly("files_skipped_by_cursor", &gpufl::UploadResult::files_skipped_by_cursor)
        .def_readonly("events_uploaded",         &gpufl::UploadResult::events_uploaded)
        .def_readonly("bytes_uploaded",          &gpufl::UploadResult::bytes_uploaded)
        .def_readonly("elapsed_ms",              &gpufl::UploadResult::elapsed_ms)
        .def_readonly("warnings",                &gpufl::UploadResult::warnings)
        // Phase 3a: async-accept backends return one spool_id per
        // chunk. The CLI doesn't print these by default — they're
        // operator-debugging cookies — but exposing on the Python
        // class lets a notebook user or test harness inspect them.
        .def_readonly("spool_ids",               &gpufl::UploadResult::spool_ids)
        .def("__repr__", [](const gpufl::UploadResult& r) {
            std::ostringstream o;
            o << "<UploadResult success=" << (r.success ? "True" : "False")
              << " events=" << r.events_uploaded
              << " bytes=" << r.bytes_uploaded
              << " files=" << r.files_processed
              << " warnings=" << r.warnings.size()
              << " spool_ids=" << r.spool_ids.size()
              << " elapsed_ms=" << r.elapsed_ms << ">";
            return o.str();
        });

    // Function-style entry — accepts every UploadOptions field as a
    // keyword argument, mirroring the init() binding's shape.
    m.def("upload_logs", [](std::string log_path, std::string backend_url,
                            std::string api_key, std::string api_path,
                            int total_timeout_ms, int connect_timeout_ms,
                            int read_timeout_ms, int max_retries,
                            int retry_delay_ms, std::string cursor_filename,
                            bool report_progress,
                            std::string session_id_filter,
                            bool all_sessions, bool force) -> gpufl::UploadResult {
        gpufl::UploadOptions opts;
        opts.log_path           = std::move(log_path);
        opts.backend_url        = std::move(backend_url);
        opts.api_key            = std::move(api_key);
        opts.api_path           = std::move(api_path);
        opts.total_timeout_ms   = total_timeout_ms;
        opts.connect_timeout_ms = connect_timeout_ms;
        opts.read_timeout_ms    = read_timeout_ms;
        opts.max_retries        = max_retries;
        opts.retry_delay_ms     = retry_delay_ms;
        opts.cursor_filename    = std::move(cursor_filename);
        opts.report_progress    = report_progress;
        opts.session_id_filter  = std::move(session_id_filter);
        opts.all_sessions       = all_sessions;
        opts.force              = force;
        // Release the GIL — upload is I/O bound, can run for minutes,
        // and we don't want to block other Python threads (notebooks
        // often run cleanup tasks concurrently).
        py::gil_scoped_release release;
        return gpufl::uploadLogs(opts);
    },
        py::arg("log_path"),
        py::arg("backend_url"),
        py::arg("api_key"),
        py::arg("api_path")           = "",
        py::arg("total_timeout_ms")   = 5 * 60 * 1000,
        py::arg("connect_timeout_ms") = 10 * 1000,
        py::arg("read_timeout_ms")    = 30 * 1000,
        py::arg("max_retries")        = 1,
        py::arg("retry_delay_ms")     = 1000,
        py::arg("cursor_filename")    = ".gpufl-upload-cursor.json",
        py::arg("report_progress")    = true,
        py::arg("session_id_filter")  = "",
        py::arg("all_sessions")       = false,
        py::arg("force")              = false);

    // F1 (External Correlation) active push/pop. Used by
    // gpufl.torch.attach()'s TorchDispatchMode to tag every aten op's
    // kernels with a stable id derived from the op name — gives the
    // dashboard's "op #N" chip without requiring users to enable
    // torch.profiler.profile() (which is heavy and intended for
    // one-off trace export, not always-on telemetry).
    //
    // We expose with a leading underscore to signal "internal API
    // for the gpufl.torch package, not a stable user-facing surface."
    m.def("_push_external_corr_id",
          [](uint32_t kind, uint64_t id) {
              gpufl::pushExternalCorrelation(kind, id);
          },
          py::arg("kind"), py::arg("id"));
    m.def("_pop_external_corr_id",
          [](uint32_t kind) {
              gpufl::popExternalCorrelation(kind);
          },
          py::arg("kind"));

    // --------------------------

    py::class_<PyScope>(m, "Scope")
        // `repeat` / `warmup` are keyword-only on purpose — they're
        // optional benchmark metadata, not positional name/tag args. The
        // pure-Python `gpufl.Scope` wrapper in __init__.py forwards
        // these when its iterable (repeat=N, warmup=K) form opens the
        // C++ scope.
        .def(py::init<std::string, std::string, uint32_t, uint32_t>(),
             py::arg("name"),
             py::arg("tag")    = "",
             py::kw_only(),
             py::arg("repeat") = 0u,
             py::arg("warmup") = 0u)
        .def("__enter__", [](PyScope &self) {
            self.enter();
            return &self;
        })
        .def("__exit__", &PyScope::exit);
}
