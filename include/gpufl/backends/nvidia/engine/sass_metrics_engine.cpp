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
#include "gpufl/core/ring_buffer.hpp"

namespace gpufl {

extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;
namespace {
std::vector<const char*> kSassMetricNames = {
    "smsp__sass_inst_executed",
    "smsp__sass_thread_inst_executed",
};

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
    if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerInitialize",
                              cuptiProfilerInitialize(&p))) {
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
        LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsDisable",
                              cuptiSassMetricsDisable(&disableParams));
    }

    if (config_set_) {
        CUpti_SassMetricsUnsetConfig_Params unsetParams = {
            CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE};
        unsetParams.deviceIndex = ctx_.device_id;
        LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsUnsetConfig",
                              cuptiSassMetricsUnsetConfig(&unsetParams));
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
            continue;
        }
        sass_metrics_buffers_->config[i].metricId = propParams.metric.metricId;
        sass_metrics_buffers_->config[i].outputGranularity =
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
        enableParams.enableLazyPatching = 1;
        if (LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsEnable",
                                  cuptiSassMetricsEnable(&enableParams))) {
            return;
        }
        enabled_ = true;
    } else {
        LogCuptiErrorIfFailed(this->name(), "cuptiSassMetricsSetConfig", res);
        GFL_LOG_ERROR("[SassMetricsEngine] Failed to enable SASS Metrics");
        return;
    }
    GFL_LOG_DEBUG("[SassMetricsEngine] SASS Metrics Enabled");
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
            if (ctx_.cubin_mu && ctx_.cubin_by_crc) {
                std::lock_guard<std::mutex> lk(*ctx_.cubin_mu);
                auto it = ctx_.cubin_by_crc->find(data[i].cubinCrc);
                if (it != ctx_.cubin_by_crc->end()) {
                    CUpti_GetSassToSourceCorrelationParams corrParams = {
                        sizeof(CUpti_GetSassToSourceCorrelationParams)};
                    corrParams.cubin = it->second.data.data();
                    corrParams.cubinSize = it->second.data.size();
                    corrParams.functionName = data[i].functionName;
                    corrParams.pcOffset = data[i].pcOffset;
                    CUptiResult res =
                        cuptiGetSassToSourceCorrelation(&corrParams);
                    if (res == CUPTI_SUCCESS) {
                        if (corrParams.fileName) {
                            std::snprintf(srcFile, sizeof(srcFile), "%s",
                                          corrParams.fileName);
                            hasSource = true;
                        }
                        srcLine = corrParams.lineNumber;
                        FreeCuptiCorrelationString(corrParams.fileName);
                        FreeCuptiCorrelationString(corrParams.dirName);
                    } else {
                        LogCuptiErrorIfFailed(this->name(),
                                              "cuptiGetSassToSourceCorrelation",
                                              res);
                    }
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
