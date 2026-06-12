#pragma once

#include <cupti_pcsampling.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"

namespace gpufl {

// ---- PC Sampling buffer management ----------------------------------------

struct PCSamplingBuffers {
    CUpti_PCSamplingData*   data      = nullptr;
    CUpti_PCSamplingPCData* pcRecords = nullptr;
};

struct PCSamplingDeleter {
    void operator()(const PCSamplingBuffers* b) const;
};

// ---- Engine ----------------------------------------------------------------

class PcSamplingEngine final : public IProfilingEngine {
   public:
    PcSamplingEngine() = default;
    ~PcSamplingEngine() override = default;
    const char* name() const override { return "PcSamplingEngine"; }

    bool initialize(const MonitorOptions& opts,
                    const EngineContext& ctx) override;
    void start()    override;
    void stop()     override;
    void shutdown() override;

    void onScopeStart(const char* name) override;
    void onScopeStop(const char* name)  override;
    void drainData() override;
    void flushBeforeCudaTeardown(const char* reason) override;
    void onLaunchTick() override;

    /// True when using the PC Sampling API (newer GPUs) rather than
    /// the legacy Activity API.  Used by the composite engine to decide
    /// whether SASS metrics can safely coexist.
    bool isSamplingAPI() const {
        return pc_sampling_method_ == Method::SamplingAPI;
    }

    /**
     * True when cuptiPCSamplingEnable or cuptiActivityEnable(PC_SAMPLING)
     * returned CUPTI_ERROR_INSUFFICIENT_PRIVILEGES during start(). Used
     * by `gpufl::init()` to surface a clear error to the user.
     */
    bool hasInsufficientPrivileges() const override {
        return sampling_api_blocked_.load(std::memory_order_relaxed);
    }

    bool stallReasonsUnavailable() const override {
        return stall_reasons_unavailable_.load(std::memory_order_relaxed);
    }

    /** Operational means we have an active method (not None) AND we're not blocked. */
    bool isOperational() const override {
        return pc_sampling_method_ != Method::None
               && !sampling_api_blocked_.load(std::memory_order_relaxed);
    }

    /** True once at least one PC sample was emitted this session. */
    bool producedData() const override {
        return produced_data_.load(std::memory_order_relaxed);
    }

   private:
    enum class Method {
        None,        // PC Sampling not available / not initialized
        ActivityAPI, // CUPTI_ACTIVITY_KIND_PC_SAMPLING (older GPUs)
        SamplingAPI, // cuptiPCSampling* API (newer GPUs)
    };

    bool EnableSamplingFeatures_();
    void StartPcSampling_();
    /// @param sync_device cudaDeviceSynchronize before stopping. Callers on
    ///        plain threads pass true; the CUPTI-callback path passes false -
    ///        cudart inside a CUPTI callback can re-enter the driver, and in
    ///        KERNEL_SERIALIZED mode every sampled kernel has already
    ///        completed by the time the next API callback runs anyway.
    void StopAndCollectPcSampling_(bool sync_device = true);
    /// The cuptiPCSamplingGetData drain loop: parses PC records into
    /// PC_SAMPLE activity records. Callable while sampling is still armed
    /// (the NVIDIA pc_sampling sample's serialized-mode pattern - only
    /// completed kernels' samples are returned) or after a Stop.
    void CollectPcSamplingData_();
    /// Shared mid-session collect: throttled armed-GetData, safe to call
    /// from CUPTI callbacks (try_lock, no cudart, no PCSamplingStop -
    /// Stop returns 999 inside CUPTI callbacks). `force` bypasses the
    /// interval throttle (used at process-scope end - the last healthy
    /// moment before Windows process-exit teardown).
    void MaybePeriodicCollect_(const char* reason, bool force);

    MonitorOptions opts_;
    EngineContext  ctx_;

    Method pc_sampling_method_ = Method::None;
    std::atomic<int> pc_sampling_ref_count_{0};
    std::atomic<bool> sampling_api_ready_{false};
    std::atomic<bool> sampling_api_started_{false};
    std::atomic<bool> sampling_api_blocked_{false};
    std::atomic<bool> stall_reasons_unavailable_{false};
    std::atomic<bool> produced_data_{false};
    bool privilege_probed_ = false;

    // Serializes the start/stop/collect lifecycle across its callers: the
    // engine's own cycle thread, scope-begin re-arms on app threads, and
    // session stop/shutdown. Without it a cycle's Stop could interleave
    // with a scope-begin Start.
    std::mutex sampling_lifecycle_mu_;
    // Minimum gap between periodic collects. Short GPU phases (a script
    // whose kernels all finish within seconds of context creation) must
    // still get at least one mid-run collect before exit teardown breaks
    // cuptiPCSamplingStop, so this errs small; each collect is one
    // stop→GetData→start cycle (~sub-ms) on the app thread.
    static constexpr int64_t kCollectIntervalNs = 1'000'000'000;  // 1 s
    // Last periodic collect (cycle), wall ns. 0 = none yet.
    std::atomic<int64_t> last_cycle_ns_{0};

    // The engine owns its collection cadence: the monitor collector thread
    // (which calls drainData) spends whole sessions inside synchronous
    // nvdisasm disassembly (flushBatches → flushDisassembly), so a cycle
    // hung off it starves and the session's samples die with the
    // process-exit cuptiPCSamplingStop. Started when sampling first arms;
    // joined by stop()/shutdown() BEFORE they take the lifecycle mutex.
    std::thread cycle_thread_;
    std::atomic<bool> cycle_thread_running_{false};
    void StartCycleThread_();
    void StopCycleThread_();

    std::unique_ptr<PCSamplingBuffers, PCSamplingDeleter> pc_sampling_buffers_;
    size_t num_stall_reasons_ = 0;  // original slot count; must reset before each getData

    mutable std::mutex                       stall_reason_mu_;
    mutable std::unordered_map<uint32_t, std::string> stall_reason_map_;
};

}  // namespace gpufl
