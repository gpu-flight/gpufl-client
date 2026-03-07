#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"

#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_sass_metrics.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/ring_buffer.hpp"

namespace gpufl {

extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;
namespace {
std::vector<const char*> kSassMetricNames = {
    "smsp__sass_inst_executed",
};
}

bool SassMetricsEngine::initialize(const MonitorOptions& opts,
                                    const EngineContext& ctx) {
    opts_ = opts;
    ctx_  = ctx;
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
    LogCuptiErrorIfFailed("SassMetrics", "cuptiProfilerInitialize",
                          cuptiProfilerInitialize(&p));

    EnableSassMetrics_();
}

void SassMetricsEngine::stop() {}

void SassMetricsEngine::shutdown() {
    if (sass_metrics_buffers_) {
        if (sass_metrics_buffers_->config)
            std::free(sass_metrics_buffers_->config);
        if (sass_metrics_buffers_->data)
            std::free(sass_metrics_buffers_->data);
        delete sass_metrics_buffers_;
        sass_metrics_buffers_ = nullptr;
    }
}

void SassMetricsEngine::onScopeStop(const char* /*name*/) {
    StopAndCollectSassMetrics_();
}

// ---- Private helpers -------------------------------------------------------

void SassMetricsEngine::EnableSassMetrics_() {
    if (ctx_.chip_name.empty()) {
        if (LogCuptiErrorIfFailed("SassMetrics", "cuptiGetDeviceId",
                                  cuptiGetDeviceId(ctx_.cuda_ctx,
                                                   &ctx_.device_id))) {
            return;
        }
        ctx_.chip_name = getChipName(ctx_.device_id);
    }

    metric_id_to_name_.clear();
    if (!sass_metrics_buffers_) {
        sass_metrics_buffers_ = new SassMetricsBuffers();
        sass_metrics_buffers_->numMetrics = kSassMetricNames.size();
        sass_metrics_buffers_->config =
            static_cast<CUpti_SassMetrics_Config*>(
                std::calloc(sass_metrics_buffers_->numMetrics,
                            sizeof(CUpti_SassMetrics_Config)));
    }

    for (size_t i = 0; i < kSassMetricNames.size(); ++i) {
        CUpti_SassMetrics_GetProperties_Params propParams = {
            CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE};
        propParams.pChipName   = ctx_.chip_name.c_str();
        propParams.pMetricName = kSassMetricNames[i];
        if (LogCuptiErrorIfFailed("SassMetrics", "cuptiSassMetricsGetProperties",
                                  cuptiSassMetricsGetProperties(&propParams))) {
            continue;
        }
        sass_metrics_buffers_->config[i].metricId = propParams.metric.metricId;
        sass_metrics_buffers_->config[i].outputGranularity =
            CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU;
        metric_id_to_name_[propParams.metric.metricId] = kSassMetricNames[i];
    }

    CUpti_SassMetricsSetConfig_Params setConfigParams = {
        CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE};
    setConfigParams.deviceIndex        = 0;
    setConfigParams.numOfMetricConfig  = sass_metrics_buffers_->numMetrics;
    setConfigParams.pConfigs           = sass_metrics_buffers_->config;
    CUptiResult res = cuptiSassMetricsSetConfig(&setConfigParams);
    if (res == CUPTI_SUCCESS || res == CUPTI_ERROR_INVALID_OPERATION) {
        CUpti_SassMetricsEnable_Params enableParams = {
            CUpti_SassMetricsEnable_Params_STRUCT_SIZE};
        enableParams.ctx                = ctx_.cuda_ctx;
        enableParams.enableLazyPatching = 1;
        cuptiSassMetricsEnable(&enableParams);
    } else {
        GFL_LOG_ERROR("[SassMetricsEngine] Failed to enable SASS Metrics");
        return;
    }
    GFL_LOG_DEBUG("[SassMetricsEngine] SASS Metrics Enabled");
}

void SassMetricsEngine::StopAndCollectSassMetrics_() {
    if (!ctx_.cuda_ctx) return;

    CUpti_SassMetricsGetDataProperties_Params props = {
        CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE};
    props.ctx = ctx_.cuda_ctx;
    if (cuptiSassMetricsGetDataProperties(&props) != CUPTI_SUCCESS ||
        props.numOfPatchedInstructionRecords == 0) {
        return;
    }

    size_t nRecords   = props.numOfPatchedInstructionRecords;
    size_t nInstances = props.numOfInstances;
    auto* data = static_cast<CUpti_SassMetrics_Data*>(
        std::calloc(nRecords, sizeof(CUpti_SassMetrics_Data)));
    auto* instances = static_cast<CUpti_SassMetrics_InstanceValue*>(
        std::calloc(nRecords * nInstances,
                    sizeof(CUpti_SassMetrics_InstanceValue)));

    for (size_t i = 0; i < nRecords; ++i) {
        data[i].structSize    = sizeof(CUpti_SassMetrics_Data);
        data[i].pInstanceValues = &instances[i * nInstances];
    }

    CUpti_SassMetricsFlushData_Params flushParams = {
        CUpti_SassMetricsFlushData_Params_STRUCT_SIZE};
    flushParams.ctx                          = ctx_.cuda_ctx;
    flushParams.numOfPatchedInstructionRecords = nRecords;
    flushParams.numOfInstances               = nInstances;
    flushParams.pMetricsData                 = data;

    if (cuptiSassMetricsFlushData(&flushParams) == CUPTI_SUCCESS) {
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
                    corrParams.cubin        = it->second.data.data();
                    corrParams.cubinSize    = it->second.data.size();
                    corrParams.functionName = data[i].functionName;
                    corrParams.pcOffset     = data[i].pcOffset;
                    CUptiResult res = cuptiGetSassToSourceCorrelation(&corrParams);
                    if (res == CUPTI_SUCCESS) {
                        if (corrParams.fileName) {
                            std::snprintf(srcFile, sizeof(srcFile), "%s",
                                          corrParams.fileName);
                            hasSource = true;
                        }
                        srcLine = corrParams.lineNumber;
                        if (corrParams.fileName) std::free(corrParams.fileName);
                        if (corrParams.dirName) std::free(corrParams.dirName);
                    } else {
                        LogCuptiErrorIfFailed("SASS Metrics",
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
                std::snprintf(out.function_name, sizeof(out.function_name), "%s",
                              data[i].functionName ? data[i].functionName : "unknown");
                if (hasSource) {
                    std::snprintf(out.source_file, sizeof(out.source_file), "%s",
                                  srcFile);
                    out.source_line = srcLine;
                }
                auto itName = metric_id_to_name_.find(inst.metricId);
                if (itName != metric_id_to_name_.end()) {
                    std::snprintf(out.metric_name, sizeof(out.metric_name), "%s",
                                  itName->second.c_str());
                } else {
                    std::snprintf(out.metric_name, sizeof(out.metric_name),
                                  "metric_%llu",
                                  static_cast<unsigned long long>(inst.metricId));
                }
                out.metric_value = inst.value;
                g_monitorBuffer.Push(out);
            }
        }
    }
    std::free(instances);
    std::free(data);
}

}  // namespace gpufl
