#include <pybind11/pybind11.h>
#include "gpumon/gpumon.hpp"

namespace py = pybind11;

class PyScope {
public:
    PyScope(std::string name, std::string tag) : name_(name), tag_(tag) {}

    void enter() {
        monitor_ = std::make_unique<gpumon::ScopedMonitor>(name_, tag_);
    }

    void exit(py::object exc_type, py::object exc_value, py::object traceback) {
        monitor_.reset();
    }

private:
    std::string name_;
    std::string tag_;
    std::unique_ptr<gpumon::ScopedMonitor> monitor_;
};

PYBIND11_MODULE(_gpumon_client, m) {
    m.doc() = "GPUMON Internal C++ Binding";

    m.def("init", [](std::string app_name, std::string log_path, int interval_ms) {
        gpumon::InitOptions opts;
        opts.appName = app_name;
        opts.logPath = log_path;
        opts.sampleIntervalMs = interval_ms;
        return gpumon::init(opts);
    }, py::arg("app_name"), py::arg("log_path") = "", py::arg("interval_ms") = 0);

    m.def("shutdown", &gpumon::shutdown);

    py::class_<PyScope>(m, "Scope")
        .def(py::init<std::string, std::string>(), py::arg("name"), py::arg("tag") = "")
        .def("__enter__", [](PyScope &self) {
            self.enter();
            return &self;
        })
        .def("__exit__", &PyScope::exit);
}