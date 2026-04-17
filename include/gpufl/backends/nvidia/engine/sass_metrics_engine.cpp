#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"

#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_sass_metrics.h>

#include <cstdio>
#include <vector>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/runtime.hpp"

namespace gpufl {

namespace {
std::vector<const char*> kSassMetricNames = {
    "smsp__sass_inst_executed",
    "smsp__sass_thread_inst_executed",
    "smsp__sass_sectors_mem_global",
    "smsp__sass_sectors_mem_global_ideal",        // sm_120+: aggregate ideal
    "smsp__sass_sectors_mem_global_op_ld_ideal",  // fallback: load ideal (sm_86)
    "smsp__sass_sectors_mem_global_op_st_ideal",  // fallback: store ideal (sm_86)
};

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

    size_t validConfigs = 0;
    for (size_t i = 0; i < kSassMetricNames.size(); ++i) {
        CUpti_SassMetrics_GetProperties_Params propParams = {
            CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE};
        propParams.pChipName = ctx_.chip_name.c_str();
        propParams.pMetricName = kSassMetricNames[i];
        if (LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsGetProperties",
                                  cuptiSassMetricsGetProperties(&propParams))) {
            GFL_LOG_DEBUG(
                "[SassMetricsEngine] Metric not available on this GPU, "
                "skipping: ", kSassMetricNames[i]);
            skipped_metrics_.push_back(kSassMetricNames[i]);
            continue;
        }
        sass_metrics_buffers_->config[validConfigs].metricId = propParams.metric.metricId;
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

            for (size_t j = 0; j < nInstances; ++j) {
                const auto& inst = data[i].pInstanceValues[j];
                ActivityRecord out{};
                out.type = TraceType::PC_SAMPLE;
                out.cpu_start_ns = detail::GetTimestampNs();
                out.device_id = ctx_.device_id;
                out.pc_offset = data[i].pcOffset;
                std::snprintf(out.sample_kind, sizeof(out.sample_kind), "%s",
                              "sass_metric");
                std::snprintf(
                    out.function_name, sizeof(out.function_name), "%s",
                    data[i].functionName ? data[i].functionName : "unknown");
                if (hasSource) {
                    std::snprintf(out.source_file, sizeof(out.source_file),
                                  "%s", srcFile);
                    out.source_line = srcLine;
                }
                auto itName = metric_id_to_name_.find(inst.metricId);
                if (itName != metric_id_to_name_.end()) {
                    std::snprintf(out.metric_name, sizeof(out.metric_name),
                                  "%s", itName->second.c_str());
                } else {
                    std::snprintf(
                        out.metric_name, sizeof(out.metric_name), "metric_%llu",
                        static_cast<unsigned long long>(inst.metricId));
                }
                out.metric_value = inst.value;
                g_monitorBuffer.Push(out);
            }
        }
    } else {
        LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsFlushData",
                              flushRes);
    }
    std::free(instances);
    std::free(data);
}

}  // namespace gpufl
