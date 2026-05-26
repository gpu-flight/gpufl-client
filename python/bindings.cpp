#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gpufl/gpufl.hpp"
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

    // BackendKind / ProfilingEngine both have a "None" value, which is a
    // Python keyword — accessible as ProfilingEngine.__members__["None"]
    // but ugly. Add a `None_` Python alias on each enum so users can write
    // `gpufl.ProfilingEngine.None_` directly. The underscore-suffix is the
    // standard Python convention for keyword-clashing names (mirrors
    // pybind11's `class_`, `type_`, etc.).
    auto backendKindEnum = py::enum_<gpufl::BackendKind>(m, "BackendKind")
        .value("Auto",   gpufl::BackendKind::Auto)
        .value("Nvidia", gpufl::BackendKind::Nvidia)
        .value("Amd",    gpufl::BackendKind::Amd)
        .value("None",   gpufl::BackendKind::None)
        .export_values();
    backendKindEnum.attr("None_") = backendKindEnum.attr("__members__")[py::str("None")];

    auto profilingEngineEnum = py::enum_<gpufl::ProfilingEngine>(m, "ProfilingEngine")
        .value("None",               gpufl::ProfilingEngine::None)
        .value("PcSampling",         gpufl::ProfilingEngine::PcSampling)
        .value("SassMetrics",        gpufl::ProfilingEngine::SassMetrics)
        .value("RangeProfiler",      gpufl::ProfilingEngine::RangeProfiler)
        .value("PcSamplingWithSass", gpufl::ProfilingEngine::PcSamplingWithSass)
        .export_values();
    profilingEngineEnum.attr("None_") = profilingEngineEnum.attr("__members__")[py::str("None")];
    // Friendly aliases — Phase 1 of mode renaming. Bound as class
    // attributes pointing at the existing enum entries (not as
    // additional `.value()` calls, which would either fight pybind11's
    // deduplication or end up shadowing the canonical name in repr).
    // gpufl.ProfilingEngine.Deep is the same value as
    // gpufl.ProfilingEngine.PcSamplingWithSass, and repr() will still
    // show the technical name. See
    // gpufl-manual/client/mode-data-layers.html for naming rationale.
    profilingEngineEnum.attr("Continuous") =
        profilingEngineEnum.attr("__members__")[py::str("PcSampling")];
    profilingEngineEnum.attr("Deep") =
        profilingEngineEnum.attr("__members__")[py::str("PcSamplingWithSass")];
    profilingEngineEnum.attr("Range") =
        profilingEngineEnum.attr("__members__")[py::str("RangeProfiler")];

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
        .def_readwrite("profiling_engine",      &gpufl::InitOptions::profiling_engine)
        // Backend interactions — backend_url is the BASE URL of the
        // GPUFlight backend; log upload is opt-in via remote_upload.
        // (The historical `config_name` / remote-config-fetch binding
        // was removed along with the backend's ConfigController.)
        .def_readwrite("backend_url",           &gpufl::InitOptions::backend_url)
        .def_readwrite("api_key",               &gpufl::InitOptions::api_key)
        .def_readwrite("remote_upload",         &gpufl::InitOptions::remote_upload);

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
                     bool enable_cuda_graphs_tracking) -> bool {

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
       py::arg("profiling_engine")            = gpufl::ProfilingEngine::PcSampling,
       py::arg("config_file")                 = "",
       py::arg("backend_url")                 = "",
       py::arg("api_key")                     = "",
       py::arg("remote_upload")               = false,
       py::arg("enable_external_correlation") = true,
       py::arg("enable_synchronization")      = true,
       py::arg("enable_memory_tracking")      = false,
       py::arg("enable_cuda_graphs_tracking") = false);

    m.def("system_start", [](std::string name) { gpufl::systemStart(std::move(name)); },
        py::arg("name") = "system");

    m.def("system_stop", [](std::string name) { gpufl::systemStop(std::move(name)); },
        py::arg("name") = "system");

    m.def("shutdown", &gpufl::shutdown);

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
