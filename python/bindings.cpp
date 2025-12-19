#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gpufl/gpufl.hpp"
#if GPUFL_HAS_CUDA
#include "gpufl/cuda/cuda.hpp"
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

#if GPUFL_HAS_CUDA
class PyKernelScope {
public:
    PyKernelScope(std::string name, std::string tag,
                  std::string grid, std::string block,
                  int dynShared, int numRegs,
                  size_t staticShared, size_t localBytes, size_t constBytes)
        : name_(name), tag_(tag), grid_(grid), block_(block),
          dynShared_(dynShared), numRegs_(numRegs),
          staticShared_(staticShared), localBytes_(localBytes), constBytes_(constBytes) {}

    void enter() {
        monitor_ = std::make_unique<gpufl::cuda::KernelMonitor>(
            name_, tag_, grid_, block_, dynShared_, numRegs_, staticShared_, localBytes_, constBytes_
        );
    }
    void exit(py::object, py::object, py::object) { monitor_.reset(); }
private:
    std::string name_, tag_, grid_, block_;
    int dynShared_, numRegs_;
    size_t staticShared_, localBytes_, constBytes_;
    std::unique_ptr<gpufl::cuda::KernelMonitor> monitor_;
};
#endif

PYBIND11_MODULE(_gpufl_client, m) {
    m.doc() = "GPUFL Internal C++ Binding";

    py::class_<gpufl::InitOptions>(m, "InitOptions")
        .def(py::init<>())
        .def_readwrite("appName", &gpufl::InitOptions::appName)
        .def_readwrite("logPath", &gpufl::InitOptions::logPath)
        .def_readwrite("systemSampleRateMs", &gpufl::InitOptions::systemSampleRateMs);

    m.def("init", [](std::string app_name,
                 std::string log_path,
                 int intervals_ms)->bool {

        std::fprintf(stderr, "[BUILD-VERIFY] Executing init binding. Path: %s\n", log_path.c_str());
        std::fflush(stderr);
        gpufl::InitOptions opts;
        opts.appName = app_name;
        opts.logPath = log_path;
        opts.systemSampleRateMs = intervals_ms;

        return gpufl::init(opts);
    }, py::arg("app_name"),
       py::arg("log_path") = "",
       py::arg("intervals_ms") = 0);

    m.def("system_start", [](std::string name) { gpufl::systemStart(std::move(name)); },
        py::arg("name") = "system");

    m.def("system_stop", [](std::string name) { gpufl::systemStop(std::move(name)); },
        py::arg("name") = "system");

    m.def("shutdown", &gpufl::shutdown);

#if GPUFL_HAS_CUDA
    py::class_<PyKernelScope>(m, "KernelScope")
        .def(py::init<std::string, std::string, std::string, std::string, int, int, size_t, size_t, size_t>(),
             py::arg("name"),
             py::arg("tag") = "",
             py::arg("grid") = "", py::arg("block") = "",
             py::arg("dynShared") = 0, py::arg("numRegs") = 0,
             py::arg("staticShared") = 0, py::arg("localBytes") = 0, py::arg("constBytes") = 0)
        .def("__enter__", [](PyKernelScope &self) { self.enter(); return &self; })
        .def("__exit__", &PyKernelScope::exit);
#endif

    // --------------------------

    py::class_<PyScope>(m, "Scope")
        .def(py::init<std::string, std::string>(), py::arg("name"), py::arg("tag") = "")
        .def("__enter__", [](PyScope &self) {
            self.enter();
            return &self;
        })
        .def("__exit__", &PyScope::exit);
}