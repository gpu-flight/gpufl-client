#include "gpufl/core/runtime.hpp"
namespace gpufl {
// Keep the runtime holder alive for process lifetime. Injection-mode atexit
// handlers can run after normal function-local/static teardown has begun.
static auto* g_rt = new std::unique_ptr<Runtime>;
Runtime* runtime() { return g_rt->get(); }
void set_runtime(std::unique_ptr<Runtime> rt) { *g_rt = std::move(rt); }
}  // namespace gpufl
