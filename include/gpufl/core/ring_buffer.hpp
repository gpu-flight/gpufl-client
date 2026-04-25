#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <thread>
#include <type_traits>

namespace gpufl {
// Helper to align data to cache lines
constexpr size_t CACHE_LINE_SIZE = 64;

template <typename T, size_t Size = 4096>
class RingBuffer {
    static_assert((Size != 0) && ((Size & (Size - 1)) == 0),
                  "Buffer Size must be a power of 2");

   public:
    enum class SlotState : uint8_t { FREE = 0, WRITING = 1, READY = 2 };

    struct Slot {
        std::atomic<SlotState> state{SlotState::FREE};
        T data;
    };

   private:
    std::array<Slot, Size> buffer_;
    static constexpr size_t MASK = Size - 1;

    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};

    alignas(CACHE_LINE_SIZE) size_t tail_{0};

   public:
    bool Push(const T& item) {
        const size_t headIdx = head_.fetch_add(1, std::memory_order_acq_rel);
        size_t index = headIdx & MASK;

        Slot* slot = &buffer_[index];

        // Wait for the slot to become FREE. On wraparound the slot still
        // holds READY data the consumer hasn't drained yet — without this
        // backpressure, bursty producers (e.g. SASS metric drain pushing
        // thousands of samples in a tight loop) overrun the ring and
        // silently drop later records (kernel activity records that arrive
        // last at cuptiActivityFlushAll were the original symptom).
        //
        // Bounded wait so a truly stuck consumer cannot deadlock CUPTI
        // callback threads. ~100 spins (~µs) then ~1000 yields (~1 ms
        // total) is comfortably long enough for the collector to drain
        // a few records and short enough that an actually-dead consumer
        // doesn't block CUPTI for noticeable time.
        constexpr int kSpinAttempts = 100;
        constexpr int kYieldAttempts = 1000;
        for (int i = 0; i < kSpinAttempts; ++i) {
            if (slot->state.load(std::memory_order_acquire) == SlotState::FREE)
                break;
        }
        for (int i = 0; i < kYieldAttempts; ++i) {
            if (slot->state.load(std::memory_order_acquire) == SlotState::FREE)
                break;
            std::this_thread::yield();
        }

        SlotState expected = SlotState::FREE;
        if (!slot->state.compare_exchange_strong(expected, SlotState::WRITING,
                                                 std::memory_order_acquire,
                                                 std::memory_order_relaxed)) {
            return false;
        }

        slot->data = item;
        slot->state.store(SlotState::READY, std::memory_order_release);
        return true;
    }

    /**
     * Only ONE thread should call this
     */
    bool Consume(T& outItem) {
        size_t index = tail_ & MASK;
        Slot* slot = &buffer_[index];

        if (slot->state.load(std::memory_order_acquire) != SlotState::READY) {
            return false;
        }

        outItem = std::move(slot->data);

        slot->state.store(SlotState::FREE, std::memory_order_release);

        tail_++;
        return true;
    }
};
}  // namespace gpufl