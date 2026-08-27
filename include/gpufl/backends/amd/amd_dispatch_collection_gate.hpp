#pragma once

#include <atomic>

#include "gpufl/core/monitor.hpp"

namespace gpufl::amd {

// Lock-free collection gate read from ROCprofiler's dispatch callback. In
// Always mode the running session supplies a counter profile for every
// dispatch. WindowOnly supplies no profile until a deep window opens, which
// ROCprofiler defines as "collect no counters for this dispatch."
class AmdDispatchCollectionGate {
   public:
    void configure(const DeepArmMode mode) {
        window_only_ = mode == DeepArmMode::WindowOnly;
        running_.store(false, std::memory_order_relaxed);
        window_active_.store(false, std::memory_order_relaxed);
    }

    void start() { running_.store(true, std::memory_order_release); }

    void stop() {
        running_.store(false, std::memory_order_release);
        window_active_.store(false, std::memory_order_release);
    }

    void openWindow() {
        window_active_.store(true, std::memory_order_release);
    }

    void closeWindow() {
        window_active_.store(false, std::memory_order_release);
    }

    bool armed() const {
        if (!running_.load(std::memory_order_acquire)) return false;
        return !window_only_ ||
               window_active_.load(std::memory_order_acquire);
    }

    bool collectDispatch(const bool window_claimed_launch) const {
        if (!running_.load(std::memory_order_acquire)) return false;
        return !window_only_ || window_claimed_launch;
    }

   private:
    bool window_only_ = false;
    std::atomic<bool> running_{false};
    std::atomic<bool> window_active_{false};
};

}  // namespace gpufl::amd
