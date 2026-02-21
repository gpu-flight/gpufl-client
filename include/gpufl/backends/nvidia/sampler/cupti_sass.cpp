#include "cupti_sass.hpp"

#include <cupti_pcsampling.h>
#include <cupti_sass_metrics.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cstdlib>
#include <iostream>

#include "gpufl/core/debug_logger.hpp"

namespace gpufl::nvidia {
    SourceCorrelation CuptiSass::sampleSourceCorrelation(const void* cubin, size_t cubinSize, const char* functionName, uint64_t pcOffset) {
        CUpti_GetSassToSourceCorrelationParams params = {CUpti_GetSassToSourceCorrelationParamsSize};
        params.cubin = cubin;
        params.cubinSize = cubinSize;
        params.functionName = functionName;
        params.pcOffset = pcOffset;

        CUptiResult res = cuptiGetSassToSourceCorrelation(&params);
        if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[SASS Metrics] cuptiGetSassToSourceCorrelation FAILED: ", err);
            return {};
        }

        SourceCorrelation result = {};
        if (params.fileName) {
            result.fileName = params.fileName;
            free(params.fileName);
        }
        if (params.dirName) {
            result.dirName = params.dirName;
            free(params.dirName);
        }
        result.lineNumber = params.lineNumber;

        return result;
    }

    bool CuptiSass::setMetricsConfig(uint32_t deviceIndex, const std::vector<std::string>& metricNames) {
        CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
        cuptiProfilerInitialize(&profilerInitializeParams);

        CUpti_Device_GetChipName_Params chipNameParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
        chipNameParams.deviceIndex = deviceIndex;
        GFL_LOG_DEBUG("start....");
        CUptiResult res = cuptiDeviceGetChipName(&chipNameParams);
        if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[SASS Metrics] cuptiDeviceGetChipName FAILED: ", err);
            return false;
        }
        const char* chipName = chipNameParams.pChipName;
        
        std::vector<CUpti_SassMetrics_Config> configs;
        for (const auto& metricName : metricNames) {
            CUpti_SassMetrics_GetProperties_Params propParams = {CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE};
            propParams.pChipName = chipName;
            propParams.pMetricName = metricName.c_str();
            
            res = cuptiSassMetricsGetProperties(&propParams);
            if (res != CUPTI_SUCCESS) {
                const char* err; cuptiGetResultString(res, &err);
                GFL_LOG_ERROR("[SASS Metrics] Failed to get properties for metric ", metricName, ": ", err);
                continue;
            }

            CUpti_SassMetrics_Config config = {};
            config.metricId = propParams.metric.metricId;
            config.outputGranularity = CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU;
            configs.push_back(config);
        }

        if (configs.empty()) {
            GFL_LOG_ERROR("[SASS Metrics] No valid metrics to configure.");
            return false;
        }

        CUpti_SassMetricsSetConfig_Params setParams = {CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE};
        setParams.deviceIndex = deviceIndex;
        setParams.numOfMetricConfig = configs.size();
        setParams.pConfigs = configs.data();

        res = cuptiSassMetricsSetConfig(&setParams);
        if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[SASS Metrics] cuptiSassMetricsSetConfig FAILED: ", err);
            return false;
        }

        GFL_LOG_DEBUG("[SASS Metrics] Configured ", configs.size(), " metrics for device ", deviceIndex);
        return true;
    }

    bool CuptiSass::unsetMetricsConfig(uint32_t deviceIndex) {
        CUpti_SassMetricsUnsetConfig_Params params = {CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE};
        params.deviceIndex = deviceIndex;

        CUptiResult res = cuptiSassMetricsUnsetConfig(&params);
        if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[SASS Metrics] cuptiSassMetricsUnsetConfig FAILED: ", err);
            return false;
        }
        return true;
    }

    bool CuptiSass::enableMetrics(void* ctx) {
        CUpti_SassMetricsEnable_Params params = {CUpti_SassMetricsEnable_Params_STRUCT_SIZE};
        params.ctx = static_cast<CUcontext>(ctx);
        params.enableLazyPatching = false;

        CUptiResult res = cuptiSassMetricsEnable(&params);
        if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[SASS Metrics] cuptiSassMetricsEnable FAILED: ", err);
            return false;
        }
        return true;
    }

    bool CuptiSass::disableMetrics(void* ctx) {
        CUpti_SassMetricsDisable_Params params = {CUpti_SassMetricsDisable_Params_STRUCT_SIZE};
        params.ctx = (CUcontext)ctx;

        CUptiResult res = cuptiSassMetricsDisable(&params);
        if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[SASS Metrics] cuptiSassMetricsDisable FAILED: ", err);
            return false;
        }
        return true;
    }

    std::vector<SassMetricData> CuptiSass::flushMetricsData(void* ctx) {
        CUpti_SassMetricsGetDataProperties_Params propParams = {CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE};
        propParams.ctx = (CUcontext)ctx;

        CUptiResult res = cuptiSassMetricsGetDataProperties(&propParams);
        if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[SASS Metrics] cuptiSassMetricsGetDataProperties FAILED: ", err);
            return {};
        }

        if (propParams.numOfPatchedInstructionRecords == 0) {
            return {};
        }

        std::vector<CUpti_SassMetrics_Data> rawData(propParams.numOfPatchedInstructionRecords);
        std::vector<CUpti_SassMetrics_InstanceValue> instanceValues(propParams.numOfPatchedInstructionRecords * propParams.numOfInstances);

        for (size_t i = 0; i < propParams.numOfPatchedInstructionRecords; ++i) {
            rawData[i].structSize = sizeof(CUpti_SassMetrics_Data);
            rawData[i].pInstanceValues = &instanceValues[i * propParams.numOfInstances];
        }

        CUpti_SassMetricsFlushData_Params flushParams = {CUpti_SassMetricsFlushData_Params_STRUCT_SIZE};
        flushParams.ctx = (CUcontext)ctx;
        flushParams.numOfPatchedInstructionRecords = propParams.numOfPatchedInstructionRecords;
        flushParams.numOfInstances = propParams.numOfInstances;
        flushParams.pMetricsData = rawData.data();

        res = cuptiSassMetricsFlushData(&flushParams);
        if (res != CUPTI_SUCCESS) {
            const char* err; cuptiGetResultString(res, &err);
            GFL_LOG_ERROR("[SASS Metrics] cuptiSassMetricsFlushData FAILED: ", err);
            return {};
        }

        std::vector<SassMetricData> result;
        for (const auto& raw : rawData) {
            SassMetricData data;
            data.cubinCrc = raw.cubinCrc;
            data.functionIndex = raw.functionIndex;
            data.functionName = raw.functionName ? raw.functionName : "";
            data.pcOffset = raw.pcOffset;
            for (size_t j = 0; j < propParams.numOfInstances; ++j) {
                data.values.push_back({raw.pInstanceValues[j].metricId, raw.pInstanceValues[j].value});
            }
            result.push_back(std::move(data));
        }

        return result;
    }
}
