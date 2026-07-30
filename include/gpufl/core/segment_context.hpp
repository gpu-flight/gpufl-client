#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>

namespace gpufl {

class Logger;
class SegmentDictionaryEmitter;
class SegmentRuntime;
struct Runtime;

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
        std::shared_ptr<SegmentDictionaryEmitter> dictionary_value = {})
        : run_id(std::move(run_id_value)),
          session_id(std::move(session_id_value)),
          segment_index(segment_index_value),
          actual_start_ns(actual_start_ns_value),
          logger(std::move(logger_value)),
          dictionary(std::move(dictionary_value)) {}

    const std::string run_id;
    const std::string session_id;
    const uint32_t segment_index;
    const int64_t actual_start_ns;
    const std::shared_ptr<Logger> logger;
    const std::shared_ptr<SegmentDictionaryEmitter> dictionary;

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

}  // namespace gpufl
