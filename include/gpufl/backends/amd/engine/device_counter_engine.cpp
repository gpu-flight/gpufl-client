#if !(GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK)
#error "device_counter_engine.cpp requires GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK"
#endif

#include "gpufl/backends/amd/engine/device_counter_engine.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "gpufl/backends/amd/amd_profiling_policy.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl::amd {
namespace {

bool CheckStatus(const rocprofiler_status_t status, const char* call) {
    if (status == ROCPROFILER_STATUS_SUCCESS) return true;
    GFL_LOG_ERROR("[DeviceCounterEngine] ", call, " failed: status=",
                  static_cast<int>(status));
    return false;
}

}  // namespace

bool DeviceCounterEngine::initialize(
    const rocprofiler_context_id_t context,
    const rocprofiler_agent_id_t gpu_agent,
    const uint32_t gpu_device_id,
    const MonitorOptions& opts) {
    context_ = context;
    gpu_agent_ = gpu_agent;
    gpu_device_id_ = gpu_device_id;
    interval_ns_ =
        static_cast<uint64_t>(std::max(opts.pm_sampling_interval_us, 1u)) *
        1000u;
    max_samples_ = opts.pm_sampling_max_samples;
    preset_ = opts.pm_sampling_preset;
    pm_sampling_scope_only_ = opts.pm_sampling_scope_only;
    deep_arm_mode_ = opts.deep_arm_mode;
    sample_row_count_.store(0, std::memory_order_relaxed);
    profile_accepted_.store(false, std::memory_order_relaxed);
    service_configured_.store(false, std::memory_order_relaxed);

    if (!discoverCounters(gpu_agent_)) {
        GFL_LOG_ERROR("[DeviceCounterEngine] no counters discovered");
        return false;
    }

    const auto requested =
        ResolveAmdDeviceCountingMetrics(opts.pm_sampling_metrics);
    if (!createCounterConfig(gpu_agent_, requested)) {
        GFL_LOG_ERROR("[DeviceCounterEngine] failed to create counter config");
        return false;
    }

    const rocprofiler_buffer_id_t no_buffer{};
    if (!CheckStatus(
            rocprofiler_configure_device_counting_service(
                context_, no_buffer, gpu_agent_,
                &DeviceCounterEngine::configureCallback, this),
            "rocprofiler_configure_device_counting_service")) {
        shutdown();
        return false;
    }
    service_configured_.store(true, std::memory_order_release);

    GFL_LOG_DEBUG("[DeviceCounterEngine] initialized device=", gpu_device_id_,
                  " metrics=", metrics_.size(),
                  " interval_us=", interval_ns_ / 1000u,
                  " output_records=", records_.size(),
                  " scope_gated=", scopeGated() ? "true" : "false");
    return true;
}

void DeviceCounterEngine::start() {
    std::lock_guard lock(sample_mutex_);
    session_active_ = true;

    if (!config_emitted_) {
        Monitor::EmitPmSamplingConfig(
            gpu_device_id_, static_cast<uint32_t>(interval_ns_ / 1000u),
            max_samples_, preset_, metrics_);
        config_emitted_ = true;
    }

    if (!profile_accepted_.load(std::memory_order_acquire)) {
        // ROCprofiler may invoke the service callback just after
        // rocprofiler_start_context returns. session_active_ remains true so
        // the first scope can arm once the callback accepts the profile.
        GFL_LOG_DEBUG(
            "[DeviceCounterEngine] awaiting ROCprofiler counter "
            "configuration callback");
        return;
    }
    if (!scopeGated()) startSamplingLocked();
}

void DeviceCounterEngine::stop() {
    std::lock_guard lock(sample_mutex_);
    stopSamplingLocked();
    session_active_ = false;
}

void DeviceCounterEngine::drain() {
    std::lock_guard lock(sample_mutex_);
    sampleLocked(true);
}

void DeviceCounterEngine::service() {
    std::lock_guard lock(sample_mutex_);
    sampleLocked(false);
}

void DeviceCounterEngine::shutdown() {
    {
        std::lock_guard lock(sample_mutex_);
        stopSamplingLocked();
        session_active_ = false;
    }
    profile_accepted_.store(false, std::memory_order_release);
    service_configured_.store(false, std::memory_order_release);
    if (config_valid_.exchange(false, std::memory_order_acq_rel)) {
        (void) CheckStatus(rocprofiler_destroy_counter_config(config_id_),
                           "rocprofiler_destroy_counter_config");
    }
}

void DeviceCounterEngine::onScopeStart(const char*) {
    if (!scopeGated()) return;
    std::lock_guard lock(sample_mutex_);
    if (session_active_) startSamplingLocked();
}

void DeviceCounterEngine::onScopeStop(const char*) {
    if (!scopeGated()) return;
    std::lock_guard lock(sample_mutex_);
    stopSamplingLocked();
}

bool DeviceCounterEngine::discoverCounters(
    const rocprofiler_agent_id_t agent) {
    std::vector<rocprofiler_counter_id_t> counter_ids;
    const auto callback =
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters,
           const size_t count, void* user_data) -> rocprofiler_status_t {
        auto* ids =
            static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
        ids->insert(ids->end(), counters, counters + count);
        return ROCPROFILER_STATUS_SUCCESS;
    };
    if (!CheckStatus(rocprofiler_iterate_agent_supported_counters(
                         agent, callback, &counter_ids),
                     "rocprofiler_iterate_agent_supported_counters")) {
        return false;
    }

    for (const auto id : counter_ids) {
        rocprofiler_counter_info_v0_t info_v0{};
        if (rocprofiler_query_counter_info(
                id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info_v0) !=
            ROCPROFILER_STATUS_SUCCESS) {
            continue;
        }

        size_t record_count = 1;
        rocprofiler_counter_info_v1_t info_v1{};
        if (rocprofiler_query_counter_info(
                id, ROCPROFILER_COUNTER_INFO_VERSION_1, &info_v1) ==
                ROCPROFILER_STATUS_SUCCESS &&
            info_v1.dimensions_instances_count > 0) {
            record_count =
                static_cast<size_t>(info_v1.dimensions_instances_count);
        }

        const std::string name = info_v0.name ? info_v0.name : "";
        if (!name.empty()) {
            counter_info_[id.handle] = CounterInfo{id, name, record_count};
        }
    }
    return !counter_info_.empty();
}

bool DeviceCounterEngine::createCounterConfig(
    const rocprofiler_agent_id_t agent,
    const std::vector<std::string>& requested) {
    std::vector<const CounterInfo*> candidates;
    for (const auto& requested_name : requested) {
        const auto it = std::find_if(
            counter_info_.begin(), counter_info_.end(),
            [&requested_name](const auto& entry) {
                return entry.second.name == requested_name;
            });
        if (it == counter_info_.end()) {
            GFL_LOG_WARN("[DeviceCounterEngine] counter unavailable: ",
                         requested_name);
            continue;
        }
        candidates.push_back(&it->second);
    }
    if (candidates.empty()) return false;

    std::vector<rocprofiler_counter_id_t> selected;
    selected.reserve(candidates.size());
    for (const auto* info : candidates) selected.push_back(info->id);

    auto status = rocprofiler_create_counter_config(
        agent, selected.data(), selected.size(), &config_id_);
    std::vector<const CounterInfo*> configured;
    if (status == ROCPROFILER_STATUS_SUCCESS) {
        configured = candidates;
    } else {
        selected.clear();
        for (const auto* candidate : candidates) {
            auto trial = selected;
            trial.push_back(candidate->id);
            rocprofiler_counter_config_id_t trial_config{};
            if (rocprofiler_create_counter_config(
                    agent, trial.data(), trial.size(), &trial_config) !=
                ROCPROFILER_STATUS_SUCCESS) {
                GFL_LOG_WARN(
                    "[DeviceCounterEngine] counter conflicts with active set: ",
                    candidate->name);
                continue;
            }
            if (!selected.empty()) {
                (void) rocprofiler_destroy_counter_config(config_id_);
            }
            config_id_ = trial_config;
            selected = std::move(trial);
            configured.push_back(candidate);
        }
    }
    if (configured.empty()) return false;

    size_t record_count = 0;
    metrics_.clear();
    for (const auto* info : configured) {
        metrics_.push_back(info->name);
        record_count += info->record_count;
        GFL_LOG_DEBUG("[DeviceCounterEngine] configured counter: ",
                      info->name, " records=", info->record_count);
    }
    records_.resize(std::max<size_t>(record_count, 1));
    config_valid_.store(true, std::memory_order_release);
    return true;
}

void DeviceCounterEngine::startSamplingLocked() {
    if (armed_.load(std::memory_order_relaxed) ||
        !profile_accepted_.load(std::memory_order_acquire)) {
        return;
    }
    next_sample_ns_ = detail::GetTimestampNs();
    attribution_start_ns_ = next_sample_ns_;
    last_sample_ns_ = 0;
    armed_.store(true, std::memory_order_release);
    GFL_LOG_DEBUG("[DeviceCounterEngine] sampling armed");
    Monitor::BeginPmScopeAttribution(next_sample_ns_);
    if (deep_arm_mode_ == DeepArmMode::WindowOnly) {
        GFL_LOG_INFO("deep window armed: amd.device_counting");
    }
}

void DeviceCounterEngine::stopSamplingLocked() {
    if (!armed_.load(std::memory_order_acquire)) return;
    std::optional<int64_t> attributed_ts_ns;
    if (scopeGated()) {
        // The scope close timestamp is captured immediately before the perf
        // stop hook reaches us. Attribute this final pull to the midpoint of
        // its collection interval rather than to callback overhead after the
        // close; otherwise every short scope's only sample appears unscoped.
        const int64_t now_ns = detail::GetTimestampNs();
        const int64_t interval_start_ns =
            std::max(attribution_start_ns_, last_sample_ns_);
        attributed_ts_ns =
            interval_start_ns + (now_ns - interval_start_ns) / 2;
    }
    sampleLocked(true, attributed_ts_ns);
    armed_.store(false, std::memory_order_release);
    Monitor::EndPmScopeAttribution();
    attribution_start_ns_ = 0;
    last_sample_ns_ = 0;
}

void DeviceCounterEngine::sampleLocked(
    const bool force, const std::optional<int64_t> attributed_ts_ns) {
    if (!armed_.load(std::memory_order_acquire) || records_.empty()) return;

    const int64_t before_ns = detail::GetTimestampNs();
    if (!force && before_ns < next_sample_ns_) return;
    next_sample_ns_ =
        before_ns + static_cast<int64_t>(interval_ns_);

    rocprofiler_user_data_t user_data{};
    user_data.value = sample_index_;
    size_t record_count = records_.size();
    auto status = rocprofiler_sample_device_counting_service(
        context_, user_data, ROCPROFILER_COUNTER_FLAG_NONE, records_.data(),
        &record_count);
    if (status == ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES &&
        record_count > records_.size()) {
        records_.resize(record_count);
        status = rocprofiler_sample_device_counting_service(
            context_, user_data, ROCPROFILER_COUNTER_FLAG_NONE,
            records_.data(), &record_count);
    }
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        ++sample_failures_;
        if (sample_failures_ == 1) {
            GFL_LOG_ERROR(
                "[DeviceCounterEngine] device counter sample failed: status=",
                static_cast<int>(status));
        }
        return;
    }
    records_.resize(record_count);
    sample_failures_ = 0;

    std::unordered_map<std::string, double> values;
    for (size_t i = 0; i < record_count; ++i) {
        const auto& record = records_[i];
        rocprofiler_counter_id_t counter_id{};
        const auto query_status =
            rocprofiler_query_record_counter_id(record.id, &counter_id);
        if (record.agent_id.handle != 0 &&
            record.agent_id.handle != gpu_agent_.handle) {
            continue;
        }
        if (query_status != ROCPROFILER_STATUS_SUCCESS) {
            continue;
        }

        std::string counter_name;
        if (const auto info = counter_info_.find(counter_id.handle);
            info != counter_info_.end()) {
            counter_name = info->second.name;
        } else {
            // ROCprofiler 1.0 on gfx1201 can return the configured derived
            // counter under a different high-word generation than the ID
            // discovered before HSA startup. Querying the record's own ID is
            // authoritative and preserves the native metric name.
            rocprofiler_counter_info_v0_t record_info{};
            if (rocprofiler_query_counter_info(
                    counter_id, ROCPROFILER_COUNTER_INFO_VERSION_0,
                    &record_info) == ROCPROFILER_STATUS_SUCCESS &&
                record_info.name) {
                counter_name = record_info.name;
                counter_info_.emplace(
                    counter_id.handle,
                    CounterInfo{counter_id, counter_name, 1});
            }
        }
        if (counter_name.empty() || !std::isfinite(record.counter_value)) {
            continue;
        }
        values[counter_name] += record.counter_value;
    }

    const int64_t after_ns = detail::GetTimestampNs();
    const int64_t sample_ns = attributed_ts_ns.value_or(
        before_ns + (after_ns - before_ns) / 2);
    std::vector<PmSampleInput> rows;
    rows.reserve(metrics_.size());
    for (const auto& metric : metrics_) {
        const auto value = values.find(metric);
        if (value == values.end()) continue;
        PmSampleInput row;
        row.sample_index = sample_index_;
        row.ts_ns = sample_ns;
        row.device_id = gpu_device_id_;
        row.metric_name = metric;
        row.value = value->second;
        rows.push_back(std::move(row));
    }
    ++sample_index_;
    if (rows.empty()) return;

    last_sample_ns_ = sample_ns;
    Monitor::PushPmSamples(rows);
    Monitor::PublishScopeRetentionWatermark(sample_ns);
    sample_row_count_.fetch_add(rows.size(), std::memory_order_relaxed);
}

void DeviceCounterEngine::configureCallback(
    const rocprofiler_context_id_t context,
    const rocprofiler_agent_id_t agent,
    const rocprofiler_device_counting_agent_cb_t set_config,
    void* user_data) {
    auto* engine = static_cast<DeviceCounterEngine*>(user_data);
    if (!engine || !set_config || agent.handle != engine->gpu_agent_.handle) {
        return;
    }
    const auto status = set_config(context, engine->config_id_);
    const bool accepted = status == ROCPROFILER_STATUS_SUCCESS;
    engine->profile_accepted_.store(accepted, std::memory_order_release);
    if (accepted) {
        std::lock_guard lock(engine->sample_mutex_);
        if (engine->session_active_ && !engine->scopeGated()) {
            engine->startSamplingLocked();
        }
    }
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        GFL_LOG_ERROR(
            "[DeviceCounterEngine] set_config failed at context start: status=",
            static_cast<int>(status));
    }
}

}  // namespace gpufl::amd
