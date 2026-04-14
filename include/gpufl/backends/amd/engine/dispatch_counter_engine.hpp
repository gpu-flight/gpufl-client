#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>

#include "gpufl/backends/amd/engine/amd_profiling_engine.hpp"

namespace gpufl::amd {

/// Collects per-kernel hardware counters via ROCprofiler dispatch counting service.
/// Each kernel dispatch triggers a callback with counter values (SQ_WAVES, cache stats, etc.).
/// Results are emitted as profile_sample_batch records with sample_kind="sass_metric".
class DispatchCounterEngine final : public AmdProfilingEngine {
   public:
    DispatchCounterEngine() = default;
    ~DispatchCounterEngine() override = default;

    bool initialize(rocprofiler_context_id_t context,
                    rocprofiler_agent_id_t gpu_agent,
                    const MonitorOptions& opts) override;
    void start() override;
    void stop() override;
    void drain() override;
    void shutdown() override;

   private:
    /// Counter metadata for resolving record IDs to human-readable names.
    struct CounterInfo {
        rocprofiler_counter_id_t id{};
        std::string name;
        std::string block;
    };

    bool discoverCounters(rocprofiler_agent_id_t agent);
    bool createCounterConfig(rocprofiler_agent_id_t agent);

    static void dispatchCallback(
        rocprofiler_dispatch_counting_service_data_t dispatch_data,
        rocprofiler_counter_config_id_t* config,
        rocprofiler_user_data_t* user_data,
        void* callback_data);

    static void recordCallback(
        rocprofiler_dispatch_counting_service_data_t dispatch_data,
        rocprofiler_counter_record_t* record_data,
        size_t record_count,
        rocprofiler_user_data_t user_data,
        void* callback_data);

    rocprofiler_context_id_t context_{};
    rocprofiler_counter_config_id_t config_id_{};
    bool config_valid_ = false;

    mutable std::mutex counter_mu_;
    std::unordered_map<uint64_t, CounterInfo> counter_info_;  // counter_id.handle → info
    std::vector<std::string> requested_counters_;             // names of counters we want
};

}  // namespace gpufl::amd
