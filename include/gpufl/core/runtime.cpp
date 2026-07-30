#include "gpufl/core/runtime.hpp"

#include <thread>
namespace gpufl {
// Keep the runtime holder alive for process lifetime. Injection-mode atexit
// handlers can run after normal function-local/static teardown has begun.
static auto* g_rt = new std::unique_ptr<Runtime>;

bool SegmentContext::tryAcquireWriter(const char* const owner) const noexcept {
    if (!accepting_writers_.load(std::memory_order_seq_cst)) return false;
    active_writers_.fetch_add(1, std::memory_order_seq_cst);
    if (!run_id.empty()) {
        // Diagnostics must never make a previously valid acquisition fail.
        // Allocation can throw on the first sighting of an owner label.
        try {
            std::lock_guard<std::mutex> lock(writer_owner_mu_);
            ++writer_owners_[owner ? owner : "general"];
        } catch (...) {
        }
    }
    if (accepting_writers_.load(std::memory_order_seq_cst)) return true;
    releaseWriter(owner);
    return false;
}

void SegmentContext::releaseWriter(const char* const owner) const noexcept {
    if (!run_id.empty()) {
        std::lock_guard<std::mutex> lock(writer_owner_mu_);
        const auto it = writer_owners_.find(owner ? owner : "general");
        if (it != writer_owners_.end()) {
            if (it->second <= 1) {
                writer_owners_.erase(it);
            } else {
                --it->second;
            }
        }
    }
    const uint64_t before =
        active_writers_.fetch_sub(1, std::memory_order_seq_cst);
    // The normal hot path is one atomic decrement. Once sealed, the last
    // writer takes the mutex before notifying so wait_for cannot miss the
    // transition between its predicate check and sleep.
    if (before == 1 &&
        !accepting_writers_.load(std::memory_order_seq_cst)) {
        std::lock_guard<std::mutex> lock(writer_drain_mu_);
        writer_drain_cv_.notify_all();
    }
}

std::string SegmentContext::activeWriterSummary() const {
    std::lock_guard<std::mutex> lock(writer_owner_mu_);
    std::string summary;
    for (const auto& [owner, count] : writer_owners_) {
        if (!summary.empty()) summary += ", ";
        summary += owner;
        summary += "=";
        summary += std::to_string(count);
    }
    return summary.empty() ? "none" : summary;
}

void SegmentContext::sealForRetirement() const noexcept {
    accepting_writers_.store(false, std::memory_order_seq_cst);
}

bool SegmentContext::waitForWriters(const std::chrono::milliseconds timeout,
                                    uint64_t* const remaining) const noexcept {
    std::unique_lock<std::mutex> lock(writer_drain_mu_);
    const bool drained = writer_drain_cv_.wait_for(lock, timeout, [this] {
        return active_writers_.load(std::memory_order_seq_cst) == 0;
    });
    if (remaining) {
        *remaining = active_writers_.load(std::memory_order_seq_cst);
    }
    return drained;
}

SegmentWriteLease::~SegmentWriteLease() { reset(); }

SegmentWriteLease::SegmentWriteLease(SegmentWriteLease&& other) noexcept
    : context_(std::move(other.context_)), owner_(other.owner_) {}

SegmentWriteLease& SegmentWriteLease::operator=(
    SegmentWriteLease&& other) noexcept {
    if (this != &other) {
        reset();
        context_ = std::move(other.context_);
        owner_ = other.owner_;
    }
    return *this;
}

void SegmentWriteLease::reset() noexcept {
    if (!context_) return;
    const auto context = std::move(context_);
    context->releaseWriter(owner_);
}

SegmentWriteLease
Runtime::acquireSegmentContext(const char* const owner) const noexcept {
    for (;;) {
        auto context = std::atomic_load_explicit(
            &active_segment_context, std::memory_order_acquire);
        if (!context) return {};
        if (context->tryAcquireWriter(owner)) {
            return SegmentWriteLease(std::move(context), owner);
        }
        // Publication sealed this context after our atomic load. Retry against
        // the newly-published context instead of writing into retirement.
        std::this_thread::yield();
    }
}

bool Runtime::hasSegmentContext() const noexcept {
    return static_cast<bool>(std::atomic_load_explicit(
        &active_segment_context, std::memory_order_acquire));
}

std::shared_ptr<const SegmentContext>
Runtime::peekSegmentContext() const noexcept {
    return std::atomic_load_explicit(&active_segment_context,
                                     std::memory_order_acquire);
}

bool Runtime::publishSegmentContext(
    std::shared_ptr<const SegmentContext> context) noexcept {
    if (!context || !context->logger || context->session_id.empty()) {
        return false;
    }
    const auto old = std::atomic_load_explicit(
        &active_segment_context, std::memory_order_acquire);
    if (old) old->sealForRetirement();
    std::atomic_store_explicit(&active_segment_context, std::move(context),
                               std::memory_order_release);
    return true;
}

std::shared_ptr<const SegmentContext>
Runtime::sealActiveSegmentContext() noexcept {
    auto context = std::atomic_load_explicit(
        &active_segment_context, std::memory_order_acquire);
    if (context) {
        context->sealForRetirement();
        std::atomic_store_explicit(
            &active_segment_context,
            std::shared_ptr<const SegmentContext>{},
            std::memory_order_release);
    }
    return context;
}

Runtime* runtime() { return g_rt->get(); }
void set_runtime(std::unique_ptr<Runtime> rt) { *g_rt = std::move(rt); }
}  // namespace gpufl
