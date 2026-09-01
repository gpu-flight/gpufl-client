#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>

#include "gpufl/backends/amd/engine/amd_profiling_engine.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl::amd {

/// Pulls device-wide hardware counters through ROCprofiler's synchronous
/// device-counting service and emits them through the PM time-series schema.
class DeviceCounterEngine final : public AmdProfilingEngine {
   public:
    DeviceCounterEngine() = default;
    ~DeviceCounterEngine() override { shutdown(); }

    bool initialize(rocprofiler_context_id_t context,
                    rocprofiler_agent_id_t gpu_agent,
                    uint32_t gpu_device_id,
                    const MonitorOptions& opts) override;
    void start() override;
    void stop() override;
    void drain() override;
    void service() override;
    void shutdown() override;

    bool hasData() const override {
        return sample_row_count_.load(std::memory_order_relaxed) > 0;
    }
    bool isPrepared() const override {
        return config_valid_.load(std::memory_order_acquire) &&
               service_configured_.load(std::memory_order_acquire);
    }
    bool isArmed() const override {
        return armed_.load(std::memory_order_acquire);
    }

    void onScopeStart(const char* name) override;
    void onScopeStop(const char* name) override;

   private:
    struct CounterInfo {
        rocprofiler_counter_id_t id{};
        std::string name;
        size_t record_count = 1;
    };

    bool discoverCounters(rocprofiler_agent_id_t agent);
    bool createCounterConfig(rocprofiler_agent_id_t agent,
                             const std::vector<std::string>& requested);
    bool scopeGated() const {
        return pm_sampling_scope_only_ ||
               deep_arm_mode_ == DeepArmMode::WindowOnly;
    }
    void startSamplingLocked();
    void stopSamplingLocked();
    void sampleLocked(bool force,
                      std::optional<int64_t> attributed_ts_ns = std::nullopt);

    static void configureCallback(
        rocprofiler_context_id_t context,
        rocprofiler_agent_id_t agent,
        rocprofiler_device_counting_agent_cb_t set_config,
        void* user_data);

    rocprofiler_context_id_t context_{};
    rocprofiler_agent_id_t gpu_agent_{};
    uint32_t gpu_device_id_ = 0;
    rocprofiler_counter_config_id_t config_id_{};

    std::unordered_map<uint64_t, CounterInfo> counter_info_;
    std::vector<std::string> metrics_;
    std::vector<rocprofiler_counter_record_t> records_;

    uint64_t interval_ns_ = 100'000;
    uint32_t max_samples_ = 4096;
    std::string preset_ = "overview";
    bool pm_sampling_scope_only_ = true;
    DeepArmMode deep_arm_mode_ = DeepArmMode::Always;

    mutable std::mutex sample_mutex_;
    bool session_active_ = false;
    bool config_emitted_ = false;
    int64_t next_sample_ns_ = 0;
    int64_t attribution_start_ns_ = 0;
    int64_t last_sample_ns_ = 0;
    uint32_t sample_index_ = 0;
    uint64_t sample_failures_ = 0;

    std::atomic<bool> config_valid_{false};
    std::atomic<bool> service_configured_{false};
    std::atomic<bool> profile_accepted_{false};
    std::atomic<bool> armed_{false};
    std::atomic<uint64_t> sample_row_count_{0};
};

}  // namespace gpufl::amd
