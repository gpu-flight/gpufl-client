#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"

#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_sass_metrics.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/runtime.hpp"

namespace gpufl {

namespace {
std::vector<const char*> kSassMetricNames = {
    "smsp__sass_inst_executed",
    "smsp__sass_thread_inst_executed",
    "smsp__sass_sectors_mem_global",
    "smsp__sass_sectors_mem_global_ideal",        // sm_120+: aggregate ideal
    "smsp__sass_sectors_mem_global_op_ld_ideal",  // fallback: load ideal
                                                  // (sm_86)
    "smsp__sass_sectors_mem_global_op_st_ideal",  // fallback: store ideal
                                                  // (sm_86)
};

/** Metrics that CUPTI accepts on pre-sm_120 GPUs but incur massive per-launch
 *  overhead due to separate kernel replay passes (one per metric class).
 *  Measured ~120x slowdown on RTX 3090 (sm_86) for tiny-kernel workloads.
 *  We proactively skip these on older hardware so users keep functional
 *  profiling (instruction + thread + actual-sectors counts) at reasonable cost.
 *  The frontend surfaces the skip via sass_config.skipped_metrics. */
bool IsExpensiveOnPreSm120(const char* name) {
    if (!name) return false;
    return std::strcmp(name, "smsp__sass_sectors_mem_global_op_ld_ideal") ==
               0 ||
           std::strcmp(name, "smsp__sass_sectors_mem_global_op_st_ideal") == 0;
}

bool IsInsufficientPrivilege(CUptiResult res) {
    if (res == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) return true;
#ifdef CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES
    if (res == CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES)
        return true;
#endif
    return false;
}

bool IsExpectedTeardownError(CUptiResult res) {
    return res == CUPTI_SUCCESS || res == CUPTI_ERROR_NOT_INITIALIZED ||
           IsInsufficientPrivilege(res);
}

void FreeCuptiCorrelationString(char* s) {
    if (!s) return;
#if defined(_WIN32) && defined(_DEBUG)
    // CUPTI may allocate these with a different CRT heap than the debug app.
    // Avoid invalid heap-pointer assertions in Debug on Windows.
    (void)s;
#else
    std::free(s);
#endif
}
}  // namespace

bool SassMetricsEngine::initialize(const MonitorOptions& opts,
                                   const EngineContext& ctx) {
    opts_ = opts;
    ctx_ = ctx;
    enabled_ = false;
    config_set_ = false;
    GFL_LOG_DEBUG("[SassMetricsEngine] initialized");
    return true;
}

void SassMetricsEngine::start() {
    if (!ctx_.cuda_ctx) {
        GFL_LOG_ERROR("[SassMetricsEngine] No CUDA context available");
        return;
    }

    // Initialize profiler device — required before SASS Metrics APIs
    CUpti_Profiler_Initialize_Params p = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUptiResult initRes = cuptiProfilerInitialize(&p);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerInitialize",
                              initRes)) {
        if (IsInsufficientPrivilege(initRes)) {
            insufficient_privileges_ = true;
            GFL_LOG_ERROR(
                "[SassMetricsEngine] Insufficient privileges for CUPTI "
                "SASS metrics. Skipping SASS instrumentation.");
        }
        return;
    }

    EnableSassMetrics_();
}

void SassMetricsEngine::stop() {}

void SassMetricsEngine::shutdown() {
    if (enabled_) {
        CUpti_SassMetricsDisable_Params disableParams = {
            CUpti_SassMetricsDisable_Params_STRUCT_SIZE};
        disableParams.ctx = ctx_.cuda_ctx;
        CUptiResult disableRes = cuptiSassMetricsDisable(&disableParams);
        if (!IsExpectedTeardownError(disableRes)) {
            LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsDisable",
                                  disableRes);
        }
    }

    if (config_set_) {
        CUpti_SassMetricsUnsetConfig_Params unsetParams = {
            CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE};
        unsetParams.deviceIndex = ctx_.device_id;
        CUptiResult unsetRes = cuptiSassMetricsUnsetConfig(&unsetParams);
        if (!IsExpectedTeardownError(unsetRes)) {
            LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsUnsetConfig",
                                  unsetRes);
        }
    }

    if (sass_metrics_buffers_) {
        if (sass_metrics_buffers_->config)
            std::free(sass_metrics_buffers_->config);
        if (sass_metrics_buffers_->data) std::free(sass_metrics_buffers_->data);
        delete sass_metrics_buffers_;
        sass_metrics_buffers_ = nullptr;
    }
    enabled_ = false;
    config_set_ = false;
}

void SassMetricsEngine::onScopeStop(const char* /*name*/) {
    if (!enabled_) return;
    StopAndCollectSassMetrics_();
}

// ---- Private helpers -------------------------------------------------------

void SassMetricsEngine::EnableSassMetrics_() {
    if (ctx_.chip_name.empty()) {
        if (LogCuptiErrorIfFailed(
                this->name(), "cuptiGetDeviceId",
                cuptiGetDeviceId(ctx_.cuda_ctx, &ctx_.device_id))) {
            return;
        }
        ctx_.chip_name = getChipName(ctx_.device_id);
    }

    metric_id_to_name_.clear();
    if (!sass_metrics_buffers_) {
        sass_metrics_buffers_ = new SassMetricsBuffers();
        sass_metrics_buffers_->numMetrics = kSassMetricNames.size();
        sass_metrics_buffers_->config = static_cast<CUpti_SassMetrics_Config*>(
            std::calloc(sass_metrics_buffers_->numMetrics,
                        sizeof(CUpti_SassMetrics_Config)));
    }

    // Proactively skip fallback sector-ideal metrics on pre-sm_120 GPUs.
    // CUPTI accepts them but each requires a separate kernel replay pass,
    // causing ~120x per-launch slowdown (measured on RTX 3090 sm_86 with a
    // 2000-launch tiny-kernel workload: 120s vs ~1s on RTX 5060 sm_120).
    // Users keep instruction/thread/actual-sector metrics; coalescing
    // efficiency is unavailable and surfaced via sass_config.skipped_metrics
    // (the frontend shows a hardware-limitation banner in the Memory tab).
    ComputeCapability cc =
        GetComputeCapability(static_cast<int>(ctx_.device_id));
    const bool skipExpensive = cc.valid() && !cc.atLeast(12, 0);
    if (skipExpensive) {
        GFL_LOG_DEBUG(
            "[SassMetricsEngine] GPU compute capability sm_", cc.major,
            cc.minor,
            " (< sm_120) — skipping sector-ideal fallback metrics to avoid "
            "kernel-replay overhead. Coalescing efficiency will be "
            "unavailable.");
    }

    size_t validConfigs = 0;
    for (size_t i = 0; i < kSassMetricNames.size(); ++i) {
        // Proactive skip for known-expensive metrics on older GPUs.
        if (skipExpensive && IsExpensiveOnPreSm120(kSassMetricNames[i])) {
            skipped_metrics_.push_back(kSassMetricNames[i]);
            continue;
        }
        CUpti_SassMetrics_GetProperties_Params propParams = {
            CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE};
        propParams.pChipName = ctx_.chip_name.c_str();
        propParams.pMetricName = kSassMetricNames[i];
        if (LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsGetProperties",
                                  cuptiSassMetricsGetProperties(&propParams))) {
            GFL_LOG_DEBUG(
                "[SassMetricsEngine] Metric not available on this GPU, "
                "skipping: ",
                kSassMetricNames[i]);
            skipped_metrics_.push_back(kSassMetricNames[i]);
            continue;
        }
        sass_metrics_buffers_->config[validConfigs].metricId =
            propParams.metric.metricId;
        sass_metrics_buffers_->config[validConfigs].outputGranularity =
            CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU;
        metric_id_to_name_[propParams.metric.metricId] = kSassMetricNames[i];
        ++validConfigs;
    }

    if (validConfigs == 0) {
        GFL_LOG_ERROR("[SassMetricsEngine] No valid SASS metrics for this GPU");
        return;
    }

    CUpti_SassMetricsSetConfig_Params setConfigParams = {
        CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE};
    setConfigParams.deviceIndex = ctx_.device_id;
    setConfigParams.numOfMetricConfig = validConfigs;
    setConfigParams.pConfigs = sass_metrics_buffers_->config;
    CUptiResult res = cuptiSassMetricsSetConfig(&setConfigParams);
    if (res == CUPTI_SUCCESS || res == CUPTI_ERROR_INVALID_OPERATION) {
        if (res == CUPTI_SUCCESS) config_set_ = true;

        CUpti_SassMetricsEnable_Params enableParams = {
            CUpti_SassMetricsEnable_Params_STRUCT_SIZE};
        enableParams.ctx = ctx_.cuda_ctx;
        // Always use lazy patching (=1): cubins are patched on their first
        // cuLaunchKernel call rather than at module-load time, avoiding upfront
        // cost on unexecuted kernels.
        //
        // The original concern was that lazy patching intercepts cuLaunchKernel
        // at the same level as KERNEL_SERIALIZED PC Sampling (SamplingAPI),
        // which could deadlock in PcSamplingWithSass mode.  In practice this
        // conflict is impossible: SamplingAPI requires cuptiPCSamplingEnable(),
        // but SASS requires cuptiProfilerInitialize() first, and that call
        // blocks the PC Sampling API.  Whenever SASS enables successfully,
        // SamplingAPI PC sampling is already disabled, so the deadlock
        // condition can never be reached.
        enableParams.enableLazyPatching = 1;
        CUptiResult enableRes = cuptiSassMetricsEnable(&enableParams);
        if (LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsEnable",
                                  enableRes)) {
            if (config_set_) {
                CUpti_SassMetricsUnsetConfig_Params unsetParams = {
                    CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE};
                unsetParams.deviceIndex = ctx_.device_id;
                CUptiResult unsetRes =
                    cuptiSassMetricsUnsetConfig(&unsetParams);
                if (!IsExpectedTeardownError(unsetRes)) {
                    LogCuptiErrorIfFailed(
                        this->name(), "cuptiSassMetricsUnsetConfig", unsetRes);
                }
                config_set_ = false;
            }
            if (IsInsufficientPrivilege(enableRes)) {
                insufficient_privileges_ = true;
                GFL_LOG_ERROR(
                    "[SassMetricsEngine] Insufficient privileges for CUPTI "
                    "SASS metrics. Skipping SASS instrumentation.");
            }
            return;
        }
        enabled_ = true;
    } else {
        LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsSetConfig", res);
        GFL_LOG_ERROR("[SassMetricsEngine] Failed to enable SASS Metrics");
        return;
    }
    GFL_LOG_DEBUG("[SassMetricsEngine] SASS Metrics Enabled");

    // Emit sass_config event so the backend can distinguish "metric not
    // supported on this GPU" from "metric produced no data for this kernel".
    if (Runtime* rt = runtime(); rt && rt->logger) {
        SassConfigEvent evt;
        evt.session_id = rt->session_id;
        evt.ts_ns = static_cast<int64_t>(detail::GetTimestampNs());
        evt.device_id = ctx_.device_id;
        for (const auto& [id, name] : metric_id_to_name_)
            evt.configured_metrics.push_back(name);
        evt.skipped_metrics = skipped_metrics_;
        rt->logger->write(model::SassConfigModel(evt));
    }
}

void SassMetricsEngine::StopAndCollectSassMetrics_() {
    if (!enabled_ || !ctx_.cuda_ctx) return;

    CUpti_SassMetricsGetDataProperties_Params props = {
        CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE};
    props.ctx = ctx_.cuda_ctx;
    if (cuptiSassMetricsGetDataProperties(&props) != CUPTI_SUCCESS ||
        props.numOfPatchedInstructionRecords == 0) {
        return;
    }

    size_t nRecords = props.numOfPatchedInstructionRecords;
    size_t nInstances = props.numOfInstances;
    auto* data = static_cast<CUpti_SassMetrics_Data*>(
        std::calloc(nRecords, sizeof(CUpti_SassMetrics_Data)));
    auto* instances = static_cast<CUpti_SassMetrics_InstanceValue*>(std::calloc(
        nRecords * nInstances, sizeof(CUpti_SassMetrics_InstanceValue)));
    if (!data || !instances) {
        std::free(instances);
        std::free(data);
        GFL_LOG_ERROR("[SassMetricsEngine] Failed to allocate metric buffers");
        return;
    }

    for (size_t i = 0; i < nRecords; ++i) {
        data[i].structSize = sizeof(CUpti_SassMetrics_Data);
        data[i].pInstanceValues = &instances[i * nInstances];
    }

    CUpti_SassMetricsFlushData_Params flushParams = {
        CUpti_SassMetricsFlushData_Params_STRUCT_SIZE};
    flushParams.ctx = ctx_.cuda_ctx;
    flushParams.numOfPatchedInstructionRecords = nRecords;
    flushParams.numOfInstances = nInstances;
    flushParams.pMetricsData = data;

    CUptiResult flushRes = cuptiSassMetricsFlushData(&flushParams);
    if (flushRes == CUPTI_SUCCESS) {
        // Build the entire burst locally, then push in a single bulk call.
        // Bypassing g_monitorBuffer (which the per-sample loop used to use)
        // avoids ring-buffer overrun: a multi-thousand-sample SASS drain
        // saturated the 8K ring faster than the collector could drain it,
        // silently dropping the kernel activity records that arrive last
        // at cuptiActivityFlushAll().  Routing this drain straight into
        // g_profileBatch under g_scopeBatchMu eliminates the contention.
        std::vector<ProfileSampleInput> samples;
        samples.reserve(nRecords * nInstances);
        const int64_t now_ns = detail::GetTimestampNs();
        for (size_t i = 0; i < nRecords; ++i) {
            char srcFile[256]{};
            uint32_t srcLine = 0;
            bool hasSource = false;
            // Grab data pointer under lock, then call CUPTI outside to avoid
            // deadlock if CUPTI triggers a module-load callback.
            const uint8_t* cubinData = nullptr;
            size_t cubinSize = 0;
            if (ctx_.cubin_mu && ctx_.cubin_by_crc) {
                std::lock_guard lk(*ctx_.cubin_mu);
                auto it = ctx_.cubin_by_crc->find(data[i].cubinCrc);
                if (it != ctx_.cubin_by_crc->end()) {
                    cubinData = it->second.data.data();
                    cubinSize = it->second.data.size();
                }
            }
            if (cubinData) {
                CUpti_GetSassToSourceCorrelationParams corrParams = {
                    sizeof(CUpti_GetSassToSourceCorrelationParams)};
                corrParams.cubin = cubinData;
                corrParams.cubinSize = cubinSize;
                corrParams.functionName = data[i].functionName;
                corrParams.pcOffset = data[i].pcOffset;
                CUptiResult res = cuptiGetSassToSourceCorrelation(&corrParams);
                if (res == CUPTI_SUCCESS) {
                    if (corrParams.fileName) {
                        if (corrParams.dirName &&
                            corrParams.dirName[0] != '\0') {
                            std::snprintf(srcFile, sizeof(srcFile), "%s/%s",
                                          corrParams.dirName,
                                          corrParams.fileName);
                        } else {
                            std::snprintf(srcFile, sizeof(srcFile), "%s",
                                          corrParams.fileName);
                        }
                        hasSource = true;
                    }
                    srcLine = corrParams.lineNumber;
                    FreeCuptiCorrelationString(corrParams.fileName);
                    FreeCuptiCorrelationString(corrParams.dirName);
                } else {
                    LogCuptiErrorIfFailed(
                        this->name(), "cuptiGetSassToSourceCorrelation", res);
                }
            }

            const std::string functionName =
                data[i].functionName ? data[i].functionName : "unknown";
            const std::string sourceFile = hasSource ? srcFile : std::string();
            // function_key matches the format CollectorLoop's PC_SAMPLE
            // branch used (function_name + "@" + source_file) — keeps
            // function_id values stable across the refactor.
            const std::string functionKey = functionName + "@" + sourceFile;
            for (size_t j = 0; j < nInstances; ++j) {
                const auto& [metricId, value] = data[i].pInstanceValues[j];
                ProfileSampleInput s;
                s.ts_ns = now_ns;
                s.device_id = ctx_.device_id;
                s.pc_offset = data[i].pcOffset;
                s.sample_kind = 1;  // sass_metric
                s.function_key = functionKey;
                auto itName = metric_id_to_name_.find(metricId);
                s.metric_name = itName != metric_id_to_name_.end()
                                    ? itName->second
                                    : "metric_" + std::to_string(metricId);
                s.metric_value = value;
                s.source_file = sourceFile;
                s.source_line = hasSource ? srcLine : 0;
                samples.push_back(std::move(s));
            }
        }
        Monitor::PushProfileSamples(samples);
    } else {
        LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsFlushData",
                              flushRes);
    }
    std::free(instances);
    std::free(data);
}

}  // namespace gpufl
