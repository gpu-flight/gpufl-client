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

    py::class_<gpufl::InitOptions>(m, "InitOptions")
        .def(py::init<>())
        .def_readwrite("appName", &gpufl::InitOptions::appName)
        .def_readwrite("logPath", &gpufl::InitOptions::logPath)
		.def_readwrite("samplingAutoStart", &gpufl::InitOptions::samplingAutoStart)
        .def_readwrite("systemSampleRateMs", &gpufl::InitOptions::systemSampleRateMs)
        .def_readwrite("enableKernelDetails", &gpufl::InitOptions::enableKernelDetails)
        .def_readwrite("enableDebugOutput", &gpufl::InitOptions::enableDebugOutput)
        .def_readwrite("enableStackTrace", &gpufl::InitOptions::enableStackTrace);

    m.def("init", [](std::string app_name,
                 std::string log_path,
				 bool sampling_auto_start,
                 int system_sample_rate_ms,
                 bool enable_kernel_details,
                 bool enable_debug_output,
                 bool enable_stack_trace)->bool {

        gpufl::InitOptions opts;
        opts.appName = app_name;
        opts.logPath = log_path;
		opts.samplingAutoStart = sampling_auto_start;
        opts.systemSampleRateMs = system_sample_rate_ms;
        opts.enableKernelDetails = enable_kernel_details;
        opts.enableDebugOutput = enable_debug_output;
        opts.enableStackTrace = enable_stack_trace;

        return gpufl::init(opts);
    }, py::arg("app_name"),
       py::arg("log_path") = "",
	   py::arg("sampling_auto_start") = false,
       py::arg("system_sample_rate_ms") = 0,
       py::arg("enable_kernel_details") = false,
       py::arg("enable_debug_output") = false,
       py::arg("enable_stack_trace") = false);

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