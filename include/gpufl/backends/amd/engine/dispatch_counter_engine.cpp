#if !(GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK)
#error "dispatch_counter_engine.cpp requires GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK"
#endif

#include "gpufl/backends/amd/engine/dispatch_counter_engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl::amd {

namespace {

// Counters we attempt to collect on each kernel dispatch.
// Not all may be available on every GPU architecture — the engine
// discovers what's supported and tries to create a compatible config.
// Counter groups that conflict are resolved by incremental fallback.
//
// NOTE: On RDNA 4 (gfx1201) with ROCm 7.2.0, many SQ instruction
// counters return 0. This is a driver limitation that will be fixed
// in future ROCm releases. The pipeline is correct — values will
// populate when driver support is added.
constexpr const char* kDesiredCounters[] = {
    // Wave/occupancy (universally supported)
    "SQ_WAVES",
    "SQ_BUSY_CYCLES",
    "SQ_WAVE_CYCLES",
    "GPUBusy",
    // Instruction mix
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SMEM",
    "SQ_INSTS_LDS",
    "SQ_INSTS_FLAT",
    // Cache
    "GL2C_HIT",
    "GL2C_MISS",
    // Memory throughput
    "FETCH_SIZE",
};

bool CheckStatus(rocprofiler_status_t status, const char* call) {
    if (status == ROCPROFILER_STATUS_SUCCESS) return true;
    GFL_LOG_ERROR("[DispatchCounterEngine] ", call, " failed: status=",
                  static_cast<int>(status));
    return false;
}

}  // namespace

bool DispatchCounterEngine::initialize(const rocprofiler_context_id_t context,
                                       const rocprofiler_agent_id_t gpu_agent,
                                       const MonitorOptions& /*opts*/) {
    context_ = context;

    if (!discoverCounters(gpu_agent)) {
        GFL_LOG_ERROR("[DispatchCounterEngine] No counters discovered");
        return false;
    }

    if (!createCounterConfig(gpu_agent)) {
        GFL_LOG_ERROR("[DispatchCounterEngine] Failed to create counter config");
        return false;
    }

    // Use callback mode: dispatch callback sets which counters to collect,
    // record callback receives the results after kernel completion.
    if (!CheckStatus(
            rocprofiler_configure_callback_dispatch_counting_service(
                context_, &DispatchCounterEngine::dispatchCallback, this,
                &DispatchCounterEngine::recordCallback, this),
            "rocprofiler_configure_callback_dispatch_counting_service")) {
        return false;
    }

    GFL_LOG_DEBUG("[DispatchCounterEngine] initialized with ",
                  counter_info_.size(), " counters");
    return true;
}

void DispatchCounterEngine::start() {
    // Context start is handled by the backend
}

void DispatchCounterEngine::stop() {
    // Context stop is handled by the backend
}

void DispatchCounterEngine::drain() {
    // Callback mode delivers data synchronously — nothing to drain
}

void DispatchCounterEngine::shutdown() {
    if (config_valid_) {
        rocprofiler_destroy_counter_config(config_id_);
        config_valid_ = false;
    }
}

bool DispatchCounterEngine::discoverCounters(const rocprofiler_agent_id_t agent) {
    struct DiscoveryCtx {
        DispatchCounterEngine* engine;
        std::vector<rocprofiler_counter_id_t> all_ids;
    };
    DiscoveryCtx ctx{this, {}};

    auto cb = [](rocprofiler_agent_id_t /*agent*/,
                 rocprofiler_counter_id_t* counters,
                 size_t num_counters,
                 void* user_data) -> rocprofiler_status_t {
        auto* c = static_cast<DiscoveryCtx*>(user_data);
        for (size_t i = 0; i < num_counters; ++i) {
            c->all_ids.push_back(counters[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    if (!CheckStatus(
            rocprofiler_iterate_agent_supported_counters(agent, cb, &ctx),
            "rocprofiler_iterate_agent_supported_counters")) {
        return false;
    }

    GFL_LOG_DEBUG("[DispatchCounterEngine] agent supports ", ctx.all_ids.size(), " counters");

    // Resolve names and build lookup table
    for (const auto& cid : ctx.all_ids) {
        rocprofiler_counter_info_v0_t info{};
        if (rocprofiler_query_counter_info(
                cid, ROCPROFILER_COUNTER_INFO_VERSION_0, &info) == ROCPROFILER_STATUS_SUCCESS) {
            counter_info_[cid.handle] = CounterInfo{
                cid,
                info.name ? info.name : "",
                info.block ? info.block : "",
            };
        }
    }

    return !counter_info_.empty();
}

bool DispatchCounterEngine::createCounterConfig(const rocprofiler_agent_id_t agent) {
    // Find counter IDs matching our desired list
    std::vector<rocprofiler_counter_id_t> selected;
    requested_counters_.clear();

    for (const char* desired : kDesiredCounters) {
        for (const auto& [handle, info] : counter_info_) {
            if (info.name == desired) {
                selected.push_back(info.id);
                requested_counters_.push_back(info.name);
                break;
            }
        }
    }

    if (selected.empty()) {
        GFL_LOG_ERROR("[DispatchCounterEngine] none of the desired counters are available");
        return false;
    }

    // Try creating config with all selected counters. If that fails (counter
    // group conflicts), try adding counters one by one and keep those that work.
    auto status = rocprofiler_create_counter_config(
        agent, selected.data(), selected.size(), &config_id_);

    if (status != ROCPROFILER_STATUS_SUCCESS) {
        GFL_LOG_DEBUG("[DispatchCounterEngine] bulk config failed, trying incremental");
        selected.clear();
        requested_counters_.clear();

        for (const char* desired : kDesiredCounters) {
            for (const auto& [handle, info] : counter_info_) {
                if (info.name != desired) continue;

                // Try adding this counter to existing set
                auto trial = selected;
                trial.push_back(info.id);
                rocprofiler_counter_config_id_t trial_config{};
                if (rocprofiler_create_counter_config(
                        agent, trial.data(), trial.size(), &trial_config) ==
                    ROCPROFILER_STATUS_SUCCESS) {
                    // Destroy previous config if we had one
                    if (!selected.empty()) {
                        rocprofiler_destroy_counter_config(config_id_);
                    }
                    config_id_ = trial_config;
                    selected.push_back(info.id);
                    requested_counters_.push_back(info.name);
                }
                break;
            }
        }

        if (selected.empty()) {
            GFL_LOG_ERROR("[DispatchCounterEngine] no compatible counters found");
            return false;
        }
    }

    GFL_LOG_DEBUG("[DispatchCounterEngine] configured ", requested_counters_.size(),
                  " counters for profiling");
    for (const auto& name : requested_counters_) {
        GFL_LOG_DEBUG("[DispatchCounterEngine]   - ", name);
    }

    config_valid_ = true;
    return true;
}

void DispatchCounterEngine::dispatchCallback(
    rocprofiler_dispatch_counting_service_data_t /*dispatch_data*/,
    rocprofiler_counter_config_id_t* config,
    rocprofiler_user_data_t* /*user_data*/,
    void* callback_data) {
    auto* engine = static_cast<DispatchCounterEngine*>(callback_data);
    if (engine && engine->config_valid_ && config) {
        *config = engine->config_id_;
    }
}

void DispatchCounterEngine::recordCallback(
    rocprofiler_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_counter_record_t* record_data,
    const size_t record_count,
    rocprofiler_user_data_t /*user_data*/,
    void* callback_data) {
    auto* engine = static_cast<DispatchCounterEngine*>(callback_data);
    if (!engine || !record_data || record_count == 0) return;

    const auto& info = dispatch_data.dispatch_info;
    const auto corr_id = dispatch_data.correlation_id.internal;

    for (size_t i = 0; i < record_count; ++i) {
        const auto& rec = record_data[i];

        // Resolve counter name from instance ID
        rocprofiler_counter_id_t counter_id{};
        if (rocprofiler_query_record_counter_id(rec.id, &counter_id) !=
            ROCPROFILER_STATUS_SUCCESS) {
            continue;
        }

        std::string counter_name;
        {
            std::lock_guard<std::mutex> lock(engine->counter_mu_);
            auto it = engine->counter_info_.find(counter_id.handle);
            if (it != engine->counter_info_.end()) {
                counter_name = it->second.name;
            }
        }
        if (counter_name.empty()) continue;

        // Emit as ActivityRecord — same format as NVIDIA SASS metrics
        ActivityRecord out{};
        out.type = TraceType::PC_SAMPLE;
        out.device_id = 0;  // TODO: resolve from agent_id
        out.cpu_start_ns = static_cast<int64_t>(dispatch_data.start_timestamp);
        out.duration_ns = static_cast<int64_t>(
            dispatch_data.end_timestamp >= dispatch_data.start_timestamp
                ? dispatch_data.end_timestamp - dispatch_data.start_timestamp
                : 0);
        out.corr_id = static_cast<uint32_t>(corr_id & 0xFFFFFFFF);
        out.metric_value = static_cast<uint64_t>(rec.counter_value);
        std::snprintf(out.metric_name, sizeof(out.metric_name), "%s",
                      counter_name.c_str());
        std::snprintf(out.sample_kind, sizeof(out.sample_kind), "sass_metric");

        // Kernel name is resolved later by the monitor's CollectorLoop
        // via correlation ID → kernel dispatch mapping

        g_monitorBuffer.Push(out);
    }
}

}  // namespace gpufl::amd
