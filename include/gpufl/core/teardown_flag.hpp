#pragma once

#include <atomic>

// Process-exit teardown flag.
//
// On Windows, the injection library (libgpufl_inject -> gpufl_inject.dll)
// drives gpufl::shutdown() from an atexit handler. By then the CUDA runtime
// has already begun destroying its primary context (cudart registers its own
// atexit AFTER our injection-time one, so it runs first). Calling
// cudaDeviceSynchronize(), cuptiActivityFlushAll(), or nvmlShutdown() against
// that dying context/driver deadlocks in the kernel - the process becomes
// unkillable.
//
// The inject lib sets this flag (Windows only) immediately before its atexit
// shutdown so the backend teardown skips exactly those driver calls. It is
// NOT set for:
//   - Linux injection (no such teardown race; the flush is needed + safe), or
//   - the embedded SDK's own gpufl::shutdown() mid-process (context alive, so
//     the flush is needed to capture the final activity buffer).
// So normal teardown is unaffected; only the Windows injection at-exit path
// short-circuits the driver calls that would otherwise wedge.

namespace gpufl::detail {

inline std::atomic<bool>& processExitTeardownFlag() {
    static std::atomic flag{false};
    return flag;
}

inline void setProcessExitTeardown(bool value) {
    processExitTeardownFlag().store(value, std::memory_order_release);
}

inline bool isProcessExitTeardown() {
    return processExitTeardownFlag().load(std::memory_order_acquire);
}

}  // namespace gpufl::detail
