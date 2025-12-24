#pragma once

#include <atomic>
#include <array>
#include <cstddef>
#include <type_traits>

namespace gpufl {
    // Helper to align data to cache lines
    constexpr size_t CACHE_LINE_SIZE = 64;

    template <typename T, size_t Size = 4096>
    class RingBuffer {
        static_assert((Size != 0) && ((Size & (Size - 1)) == 0), "Buffer Size must be a power of 2");

    public:
        enum class SlotState : uint8_t {
            FREE = 0,
            WRITING = 1,
            READY = 2
        };

        struct Slot {
            std::atomic<SlotState> state { SlotState::FREE };
            T data;
        };

    private:
        std::array<Slot, Size> buffer_;
        static constexpr size_t MASK = Size - 1;

        alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};

        alignas(CACHE_LINE_SIZE) size_t tail_{0};

    public:
        bool Push(const T& item) {
            size_t headIdx = head_.fetch_add(1, std::memory_order_acq_rel);
            size_t index = headIdx & MASK;

            Slot* slot = &buffer_[index];
            SlotState expected = SlotState::FREE;

            // Try to transition FREE -> WRITING
            if (!slot->state.compare_exchange_strong(expected, SlotState::WRITING,
                                                     std::memory_order_acquire,
                                                     std::memory_order_relaxed)) {
                // Failure: The buffer is full (the tail hasn't caught up),
                // or another thread is still messing with this slot (rare race).
                // For a profiler, we DROP the packet
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
}