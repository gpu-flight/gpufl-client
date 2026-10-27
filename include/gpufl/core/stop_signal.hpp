#pragma once

#include <atomic>

// The signal that requested this process's final shutdown (0: none).
//
// The Linux injection library sets it just before the orderly shutdown it runs
// on SIGTERM (inject/sigterm_flush_linux.hpp). The final shutdown row then
// says the process was stopped by that signal, although it flushed completely.

namespace gpufl::detail {

inline std::atomic<int>& stopSignalSlot() {
    static std::atomic slot{0};
    return slot;
}

inline void setStopSignal(const int signal) {
    stopSignalSlot().store(signal, std::memory_order_release);
}

inline int stopSignal() {
    return stopSignalSlot().load(std::memory_order_acquire);
}

}  // namespace gpufl::detail
