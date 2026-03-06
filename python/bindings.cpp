#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gpufl/gpufl.hpp"
#if GPUFL_HAS_CUDA
#endif

namespace py = pybind11;

class PyScope {
public:
    PyScope(std::string name, std::string tag) : name_(name), tag_(tag) {}

    void enter() {
        monitor_ = std::make_unique<gpufl::ScopedMonitor>(name_, tag_);
    }

    void exit(py::object exc_type, py::object exc_value, py::object traceback) {
        monitor_.reset();
    }

private:
    std::string name_;
    std::string tag_;
    std::unique_ptr<gpufl::ScopedMonitor> monitor_;
};

PYBIND11_MODULE(_gpufl_client, m) {
    m.doc() = "GPUFL Internal C++ Binding";

    py::enum_<gpufl::BackendKind>(m, "BackendKind")
        .value("Auto",   gpufl::BackendKind::Auto)
        .value("Nvidia", gpufl::BackendKind::Nvidia)
        .value("Amd",    gpufl::BackendKind::Amd)
        .value("None",   gpufl::BackendKind::None)
        .export_values();

    py::class_<gpufl::InitOptions>(m, "InitOptions")
        .def(py::init<>())
        .def_readwrite("app_name",              &gpufl::InitOptions::app_name)
        .def_readwrite("log_path",              &gpufl::InitOptions::log_path)
        .def_readwrite("sampling_auto_start",   &gpufl::InitOptions::sampling_auto_start)
        .def_readwrite("system_sample_rate_ms", &gpufl::InitOptions::system_sample_rate_ms)
        .def_readwrite("kernel_sample_rate_ms", &gpufl::InitOptions::kernel_sample_rate_ms)
        .def_readwrite("backend",               &gpufl::InitOptions::backend)
        .def_readwrite("enable_kernel_details", &gpufl::InitOptions::enable_kernel_details)
        .def_readwrite("enable_debug_output",   &gpufl::InitOptions::enable_debug_output)
        .def_readwrite("enable_profiling",      &gpufl::InitOptions::enable_profiling)
        .def_readwrite("enable_stack_trace",    &gpufl::InitOptions::enable_stack_trace)
        .def_readwrite("enable_perf_scope",     &gpufl::InitOptions::enable_perf_scope);

    m.def("init", [](std::string app_name,
                     std::string log_path,
                     bool sampling_auto_start,
                     int system_sample_rate_ms,
                     int kernel_sample_rate_ms,
                     gpufl::BackendKind backend,
                     bool enable_kernel_details,
                     bool enable_debug_output,
                     bool enable_profiling,
                     bool enable_stack_trace,
                     bool enable_perf_scope) -> bool {

        gpufl::InitOptions opts;
        opts.app_name              = app_name;
        opts.log_path              = log_path;
        opts.sampling_auto_start   = sampling_auto_start;
        opts.system_sample_rate_ms = system_sample_rate_ms;
        opts.kernel_sample_rate_ms = kernel_sample_rate_ms;
        opts.backend               = backend;
        opts.enable_kernel_details = enable_kernel_details;
        opts.enable_debug_output   = enable_debug_output;
        opts.enable_profiling      = enable_profiling;
        opts.enable_stack_trace    = enable_stack_trace;
        opts.enable_perf_scope     = enable_perf_scope;

        return gpufl::init(opts);
    }, py::arg("app_name"),
       py::arg("log_path")               = "",
       py::arg("sampling_auto_start")    = false,
       py::arg("system_sample_rate_ms")  = 0,
       py::arg("kernel_sample_rate_ms")  = 0,
       py::arg("backend")                = gpufl::BackendKind::Auto,
       py::arg("enable_kernel_details")  = false,
       py::arg("enable_debug_output")    = false,
       py::arg("enable_profiling")       = true,
       py::arg("enable_stack_trace")     = true,
       py::arg("enable_perf_scope")      = false);

    m.def("system_start", [](std::string name) { gpufl::systemStart(std::move(name)); },
        py::arg("name") = "system");

    m.def("system_stop", [](std::string name) { gpufl::systemStop(std::move(name)); },
        py::arg("name") = "system");

    m.def("shutdown", &gpufl::shutdown);

    // --------------------------

    py::class_<PyScope>(m, "Scope")
        .def(py::init<std::string, std::string>(), py::arg("name"), py::arg("tag") = "")
        .def("__enter__", [](PyScope &self) {
            self.enter();
            return &self;
        })
        .def("__exit__", &PyScope::exit);
}
