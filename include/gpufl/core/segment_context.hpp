#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <limits>

namespace gpufl {

class Logger;
class SegmentDictionaryEmitter;
class SegmentRuntime;
struct Runtime;

/**
 * Identity shared by every SegmentContext of one run part.
 *
 * A run is a chain of parts joined by roll_chain_id. Each part owns a distinct
 * run_id and its own segment numbering; a rollover mints a new RunPartContext
 * with a fresh run_id, an incremented part_index, and previous_run_id pointing
 * at the part it succeeded. An ordinary segment cut keeps the same one.
 *
 * job_start and run_end read run identity from here rather than from a mutable
 * process global, so a part still carries the correct identity after a roll.
 *
 * Identity is const; the byte counter is not. It lives here because the budget
 * is scoped to this object, so a roll resets it by construction.
 */
struct RunPartContext {
    RunPartContext(std::string roll_chain_id_value, std::string run_id_value,
                   std::string previous_run_id_value,
                   uint32_t part_index_value,
                   int64_t run_started_mono_ns_value,
                   uint32_t first_segment_index_value = 0)
        : roll_chain_id(std::move(roll_chain_id_value)),
          run_id(std::move(run_id_value)),
          previous_run_id(std::move(previous_run_id_value)),
          part_index(part_index_value),
          run_started_mono_ns(run_started_mono_ns_value),
          first_segment_index(first_segment_index_value) {}

    const std::string roll_chain_id;
    const std::string run_id;
    const std::string previous_run_id;
    const uint32_t part_index;
    const int64_t run_started_mono_ns;
    const uint32_t first_segment_index;

    /** Lock-free (logger write path). Saturates: a wrap would read as a tiny
     *  budget and roll continuously. */
    void addSerializedBytes(const uint64_t bytes) const noexcept {
        auto current = serialized_bytes_.load(std::memory_order_relaxed);
        for (;;) {
            const uint64_t next =
                bytes > (std::numeric_limits<uint64_t>::max)() - current
                    ? (std::numeric_limits<uint64_t>::max)()
                    : current + bytes;
            if (serialized_bytes_.compare_exchange_weak(
                    current, next, std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                return;
            }
        }
    }

    uint64_t serializedBytes() const noexcept {
        return serialized_bytes_.load(std::memory_order_relaxed);
    }

   private:
    mutable std::atomic<uint64_t> serialized_bytes_{0};
};

/**
 * Immutable identity and output ownership for one segment.
 *
 * Writers acquire one SegmentWriteLease and retain it for the complete
 * serialization/write operation. Publishing a new context is the storage
 * linearization point: a writer already leased against the old context may
 * finish there, while new writers move to the new context.
 */
struct SegmentContext {
    SegmentContext(
        std::string run_id_value, std::string session_id_value,
        uint32_t segment_index_value, int64_t actual_start_ns_value,
        std::shared_ptr<Logger> logger_value,
        std::shared_ptr<SegmentDictionaryEmitter> dictionary_value = {},
        std::shared_ptr<const RunPartContext> run_part_value = {})
        : run_id(std::move(run_id_value)),
          session_id(std::move(session_id_value)),
          segment_index(segment_index_value),
          actual_start_ns(actual_start_ns_value),
          logger(std::move(logger_value)),
          dictionary(std::move(dictionary_value)),
          run_part(std::move(run_part_value)) {}

    const std::string run_id;
    const std::string session_id;
    const uint32_t segment_index;
    const int64_t actual_start_ns;
    const std::shared_ptr<Logger> logger;
    const std::shared_ptr<SegmentDictionaryEmitter> dictionary;
    const std::shared_ptr<const RunPartContext> run_part;

   private:
    friend class SegmentWriteLease;
    friend class SegmentRuntime;
    friend class Sampler;
    friend struct Runtime;

    bool tryAcquireWriter(const char* owner) const noexcept;
    void releaseWriter(const char* owner) const noexcept;
    void sealForRetirement() const noexcept;
    bool waitForWriters(std::chrono::milliseconds timeout,
                        uint64_t* remaining) const noexcept;
    std::string activeWriterSummary() const;

    mutable std::atomic<bool> accepting_writers_{true};
    mutable std::atomic<uint64_t> active_writers_{0};
    mutable std::mutex writer_drain_mu_;
    mutable std::condition_variable writer_drain_cv_;
    // Segmentation is opt-in, so retaining owner counts here adds no cost to
    // ordinary sessions. When a drain times out this turns "one writer leaked"
    // into an actionable producer name instead of an untraceable hang.
    mutable std::mutex writer_owner_mu_;
    mutable std::unordered_map<std::string, uint64_t> writer_owners_;
};

/**
 * One complete serialization/write operation against an immutable segment.
 *
 * Move-only by design: a writer cannot accidentally extend retirement by
 * caching a copied context handle in an unrelated container.
 */
class SegmentWriteLease {
   public:
    SegmentWriteLease() noexcept = default;
    SegmentWriteLease(std::nullptr_t) noexcept {}
    ~SegmentWriteLease();

    SegmentWriteLease(const SegmentWriteLease&) = delete;
    SegmentWriteLease& operator=(const SegmentWriteLease&) = delete;
    SegmentWriteLease(SegmentWriteLease&& other) noexcept;
    SegmentWriteLease& operator=(SegmentWriteLease&& other) noexcept;

    const SegmentContext* operator->() const noexcept {
        return context_.get();
    }
    const SegmentContext& operator*() const noexcept { return *context_; }
    const SegmentContext* get() const noexcept { return context_.get(); }
    explicit operator bool() const noexcept {
        return static_cast<bool>(context_);
    }
    bool operator==(std::nullptr_t) const noexcept { return !context_; }
    bool operator!=(std::nullptr_t) const noexcept {
        return static_cast<bool>(context_);
    }
    void reset() noexcept;

   private:
    friend struct Runtime;
    explicit SegmentWriteLease(
        std::shared_ptr<const SegmentContext> context,
        const char* owner) noexcept
        : context_(std::move(context)), owner_(owner) {}

    std::shared_ptr<const SegmentContext> context_;
    const char* owner_ = "general";
};

inline uint32_t wireSegmentIndex(const SegmentContext& context) {
    if (!context.run_part) return context.segment_index;
    return context.segment_index - context.run_part->first_segment_index;
}

}  // namespace gpufl
