#pragma once
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/batch_buffer.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/segment_context.hpp"

namespace gpufl {
class Logger;

template <typename T>
class ISystemCollector {
   public:
    virtual ~ISystemCollector() = default;

    virtual std::vector<T> sampleAll() = 0;
};

/**
 * Periodic system-metric sampler. Polls the device collector (NVML / ROCm
 * SMI) and host collector (/proc) at a fixed interval, batches rows, and
 * pushes batches into the logger.
 *
 * Activation model: ref-counted. `configure()` is called once with the
 * collectors and interval. `activate()` increments the counter; the
 * worker thread starts on the 0→1 transition. `deactivate()` decrements;
 * the thread stops on the 1→0 transition. Multiple overlapping
 * activators (auto-start baseline + GFL_SCOPE entry + explicit
 * systemStart) all coexist correctly - the sampler runs while any
 * activation is in flight and idles when the counter returns to zero.
 *
 * `shutdown()` is the final teardown - called from gpufl::shutdown() to
 * zero the counter and join the worker regardless of remaining
 * activations (defends against leaked scopes / missed deactivations).
 */
class Sampler {
   public:
    using SegmentProvider = std::function<SegmentWriteLease()>;
    using SegmentPeekProvider =
        std::function<std::shared_ptr<const SegmentContext>()>;
    using RowObserver =
        std::function<void(uint32_t, uint64_t, int64_t, int64_t)>;

    Sampler();
    ~Sampler();

    /**
     * Store collector / logger / interval. Idempotent. Does NOT start
     * the worker thread. Call once after the Runtime is built, before
     * any activate(). Subsequent calls update the stored params; only
     * apply on the NEXT activation transition (we don't restart a
     * running worker mid-flight to swap params).
     */
    void configure(std::string appName, std::string sessionId,
                   std::shared_ptr<Logger> logger,
                   std::shared_ptr<ISystemCollector<DeviceSample>> collector,
                   int sampleIntervalMs,
                   HostCollector* hostCollector = nullptr);

    /**
     * Segmentation-aware configuration. The sampler retains one writer lease
     * for the batch and uses a read-only snapshot to detect publication. This
     * is important during Windows injection teardown: ExitProcess can end the
     * worker without running stack destructors, so a per-iteration writer
     * lease could otherwise leak and block final-segment retirement.
     */
    void configure(std::string appName, SegmentProvider segmentProvider,
                   SegmentPeekProvider segmentPeekProvider,
                   std::shared_ptr<ISystemCollector<DeviceSample>> collector,
                   int sampleIntervalMs,
                   HostCollector* hostCollector = nullptr,
                   RowObserver rowObserver = {});

    /**
     * Increment the activation counter. On 0→1, spawn the worker
     * thread. Safe to call before configure() - silently no-ops until
     * configured (intervalMs > 0 and collector set).
     */
    void activate();

    /**
     * Decrement the activation counter. On 1→0, stop and join the
     * worker. Calls that would push the counter below zero clamp at
     * zero and log a one-shot warning (indicates a programming error
     * - unbalanced activate/deactivate).
     */
    void deactivate();

    /**
     * Final teardown - zero the counter, stop the worker. Called from
     * gpufl::shutdown(). Idempotent.
     */
    void shutdown();

    /** True if the worker thread is currently sampling. */
    bool running() const { return running_.load(); }

    /** Current activation count. For tests / debug. */
    int activations() const { return activations_.load(); }

   private:
    static constexpr int kMetricBatchSize = 4;  // flush every N samples

    void runLoop_();
    void flushBatches_(
        const SegmentWriteLease& context);

    // Spawns the worker. Caller must hold mu_ and ensure no worker is
    // currently running.
    void startWorkerLocked_();

    // Signals the worker to stop and joins it. Caller must hold mu_.
    void stopWorkerLocked_();

    std::atomic<int>  activations_{0};
    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::thread th_;

    std::string appName_;
    SegmentProvider segment_provider_;
    SegmentPeekProvider segment_peek_provider_;
    RowObserver row_observer_;
    SegmentWriteLease batch_context_;
    std::shared_ptr<ISystemCollector<DeviceSample>> collector_;
    HostCollector* host_collector_{nullptr};  // non-owning
    int intervalMs_{0};

    BatchBuffer<DeviceMetricBatchRow> batch_;
    uint64_t batch_id_ = 0;

    BatchBuffer<HostMetricBatchRow> host_batch_;
    uint64_t host_batch_id_ = 0;
};
}  // namespace gpufl
