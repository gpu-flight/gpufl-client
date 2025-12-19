#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gpufl/gpufl.hpp"

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
        .def_readwrite("scopeSampleRateMs", &gpufl::InitOptions::scopeSampleRateMs)
        .def_readwrite("systemSampleRateMs", &gpufl::InitOptions::systemSampleRateMs);

    m.def("init", [](std::string app_name,
                 std::string log_path,
                 int intervals_ms)->bool {

        std::fprintf(stderr, "[BUILD-VERIFY] Executing init binding. Path: %s\n", log_path.c_str());
        std::fflush(stderr);
        gpufl::InitOptions opts;
        opts.appName = app_name;
        opts.logPath = log_path;
        opts.scopeSampleRateMs = intervals_ms;
        opts.systemSampleRateMs = intervals_ms;

        return gpufl::init(opts);
    }, py::arg("app_name"),
       py::arg("log_path") = "",
       py::arg("intervals_ms") = 0);

    m.def("system_start", [](const int interval_ms, std::string name) { gpufl::systemStart(interval_ms, std::move(name)); },
        py::arg("interval_ms"), py::arg("name") = "system");

    m.def("system_stop", [](std::string name) { gpufl::systemStop(std::move(name)); },
        py::arg("name") = "system");

    m.def("shutdown", &gpufl::shutdown);

#if GPUFL_HAS_CUDA
    m.def("log_kernel", [](std::string name,
                           int gx, int gy, int gz,
                           int bx, int by, int bz,
                           long long start_ns, long long end_ns) {

        // Construct dummy CUDA types from Python integers
        dim3 grid(gx, gy, gz);
        dim3 block(bx, by, bz);

        // Empty attributes (Python doesn't have access to this low-level info)
        cudaFuncAttributes attrs = {};

        gpufl::cuda::logKernelEvent(name, start_ns, end_ns, grid, block, 0, "Success", attrs);

    }, py::arg("name"),
       py::arg("gx"), py::arg("gy"), py::arg("gz"),
       py::arg("bx"), py::arg("by"), py::arg("bz"),
       py::arg("start_ns"), py::arg("end_ns"));
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