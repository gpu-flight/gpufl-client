// glibc calls this entry after shared-library C++ initialization and before
// executable constructors. HIP code-object registration can happen in those
// constructors, so wrapping only main() would miss the first code objects.
#include <features.h>
#if !defined(__linux__) || !defined(__GLIBC__)
#error "AMD preload startup requires Linux/glibc"
#endif
#include <dlfcn.h>
#include <unistd.h>

namespace gpufl::inject {
void initializeAmdInjection() noexcept;
}

extern "C" __attribute__((visibility("default")))
int __libc_start_main(int (*main_fn)(int, char**, char**), int argc, char** argv,
                      void (*init)(), void (*fini)(), void (*rtld_fini)(),
                      void* stack_end) {
    using StartMain = decltype(&__libc_start_main);
    auto next = reinterpret_cast<StartMain>(dlsym(RTLD_NEXT, "__libc_start_main"));
    if (!next) _exit(127);
    gpufl::inject::initializeAmdInjection();
    return next(main_fn, argc, argv, init, fini, rtld_fini, stack_end);
}
