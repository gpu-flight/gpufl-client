#include "gpufl/core/runtime.hpp"
namespace gpufl {
static std::unique_ptr<Runtime> g_rt;
Runtime* runtime() { return g_rt.get(); }
void set_runtime(std::unique_ptr<Runtime> rt) { g_rt = std::move(rt); }
}  // namespace gpufl