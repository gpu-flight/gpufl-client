#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
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

    // All producers serialize only the tiny reserve/copy/publish critical
    // section.  Do not reserve a sequence with fetch_add before a slot is
    // known free: a timed-out producer would leave an unfillable hole and the
    // single consumer could never advance beyond it.
    alignas(CACHE_LINE_SIZE) size_t head_{0};
    std::mutex producer_mu_;

    alignas(CACHE_LINE_SIZE) size_t tail_{0};

    alignas(CACHE_LINE_SIZE) std::atomic<size_t> dropped_{0};

   public:
    bool Push(const T& item) {
        // CUPTI can invoke producers concurrently.  Bound lock acquisition
        // just like the old slot wait: a delayed producer drops its own event
        // without advancing head_, so it cannot poison the FIFO sequence.
        constexpr int kSpinAttempts = 100;
        constexpr int kYieldAttempts = 1000;
        std::unique_lock<std::mutex> producerLock(producer_mu_,
                                                  std::defer_lock);
        for (int i = 0; i < kSpinAttempts && !producerLock.owns_lock(); ++i) {
            if (producerLock.try_lock()) break;
        }
        for (int i = 0; i < kYieldAttempts && !producerLock.owns_lock(); ++i) {
            if (producerLock.try_lock()) break;
            if (!producerLock.owns_lock()) std::this_thread::yield();
        }
        if (!producerLock.owns_lock()) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        Slot* slot = &buffer_[head_ & MASK];

        // Wait for the slot to become FREE. On wraparound the slot still
        // holds READY data the consumer hasn't drained yet - without this
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
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        slot->data = item;
        slot->state.store(SlotState::READY, std::memory_order_release);
        ++head_;
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

    size_t droppedCount() const {
        return dropped_.load(std::memory_order_relaxed);
    }

    size_t resetDroppedCount() {
        return dropped_.exchange(0, std::memory_order_relaxed);
    }
};
}  // namespace gpufl
