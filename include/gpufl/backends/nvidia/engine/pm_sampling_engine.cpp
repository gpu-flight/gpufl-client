#include "gpufl/backends/nvidia/engine/pm_sampling_engine.hpp"

#if GPUFL_HAS_PERFWORKS
#include <cupti.h>
#include <cupti_profiler_target.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstdint>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/debug_logger.hpp"

namespace gpufl {

namespace {
std::vector<std::string> OverviewMetrics() {
    return {
        "sm__warps_launched.sum",
    };
}

#if GPUFL_HAS_PERFWORKS
bool IsPrivilegeError(CUptiResult res) {
    return res == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES;
}
#endif
}  // namespace

bool PmSamplingEngine::initialize(const MonitorOptions& opts,
                                  const EngineContext& ctx) {
    opts_ = opts;
    ctx_ = ctx;
    metrics_ = ResolveMetrics_();
    GFL_LOG_DEBUG("[PmSamplingEngine] initialized preset=", opts_.pm_sampling_preset,
                  " metrics=", metrics_.size(),
                  " interval_us=", opts_.pm_sampling_interval_us,
                  " max_samples=", opts_.pm_sampling_max_samples,
                  " scope_only=", opts_.pm_sampling_scope_only ? "true" : "false");
    return true;
}

void PmSamplingEngine::start() {
#if GPUFL_HAS_PERFWORKS
    if (!opts_.pm_sampling_scope_only) {
        StartPmSampling_();
    } else {
        attempted_.store(true, std::memory_order_relaxed);
        if (!config_emitted_) {
            EmitConfig_();
            config_emitted_ = true;
        }
        GFL_LOG_DEBUG("[PmSamplingEngine] scope-only mode: PM sampling arms on scope start");
    }
#else
    GFL_LOG_ERROR("[PmSamplingEngine] Not built with GPUFL_HAS_PERFWORKS");
#endif
}

void PmSamplingEngine::stop() {
#if GPUFL_HAS_PERFWORKS
    StopPmSampling_();
#endif
}

void PmSamplingEngine::shutdown() {
#if GPUFL_HAS_PERFWORKS
    StopPmSampling_();
    DisablePmSampling_();
#endif
    operational_.store(false, std::memory_order_relaxed);
}

void PmSamplingEngine::onScopeStart(const char*) {
#if GPUFL_HAS_PERFWORKS
    if (opts_.pm_sampling_scope_only) StartPmSampling_();
#endif
}

void PmSamplingEngine::onScopeStop(const char*) {
#if GPUFL_HAS_PERFWORKS
    if (opts_.pm_sampling_scope_only) StopPmSampling_();
#endif
}

std::vector<std::string> PmSamplingEngine::ResolveMetrics_() const {
    if (!opts_.pm_sampling_metrics.empty()) return opts_.pm_sampling_metrics;
    return OverviewMetrics();
}

void PmSamplingEngine::EmitConfig_() const {
    Monitor::EmitPmSamplingConfig(ctx_.device_id,
                                  opts_.pm_sampling_interval_us,
                                  opts_.pm_sampling_max_samples,
                                  opts_.pm_sampling_preset,
                                  metrics_);
}

#if GPUFL_HAS_PERFWORKS
bool PmSamplingEngine::InitializePmSampling_() {
    if (pm_initialized_) return true;
    if (!ctx_.cuda_ctx) {
        GFL_LOG_ERROR("[PmSamplingEngine] No CUDA context available");
        return false;
    }

    CUpti_Profiler_Initialize_Params profilerInit = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUptiResult res = cuptiProfilerInitialize(&profilerInit);
    if (res != CUPTI_SUCCESS && res != CUPTI_ERROR_INVALID_OPERATION) {
        if (IsPrivilegeError(res)) insufficient_privileges_.store(true, std::memory_order_relaxed);
        LogCuptiErrorIfFailed(this->name(), "cuptiProfilerInitialize", res);
        return false;
    }
    profiler_initialized_ = true;
    profiler_init_owned_ = (res == CUPTI_SUCCESS);

    if (ctx_.chip_name.empty()) {
        uint32_t dev = ctx_.device_id;
        CUptiResult devRes = cuptiGetDeviceId(ctx_.cuda_ctx, &dev);
        if (LogCuptiErrorIfFailed(this->name(), "cuptiGetDeviceId", devRes)) return false;
        ctx_.device_id = dev;
        ctx_.chip_name = getChipName(ctx_.device_id);
    }
    if (ctx_.chip_name.empty()) {
        GFL_LOG_ERROR("[PmSamplingEngine] missing chip name; PM Sampling unavailable");
        return false;
    }

    GFL_LOG_DEBUG("[PmSamplingEngine] chip_name_ = ", ctx_.chip_name);

    CUpti_Profiler_GetCounterAvailability_Params ca = {
        CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    ca.ctx = ctx_.cuda_ctx;
    ca.pCounterAvailabilityImage = nullptr;
    res = cuptiProfilerGetCounterAvailability(&ca);
    if (res != CUPTI_SUCCESS) {
        if (IsPrivilegeError(res)) insufficient_privileges_.store(true, std::memory_order_relaxed);
        LogCuptiErrorIfFailed(this->name(), "cuptiProfilerGetCounterAvailability(size)", res);
        return false;
    }
    counter_availability_image_.assign(ca.counterAvailabilityImageSize, 0);
    ca.pCounterAvailabilityImage = counter_availability_image_.data();
    res = cuptiProfilerGetCounterAvailability(&ca);
    if (res != CUPTI_SUCCESS) {
        if (IsPrivilegeError(res)) insufficient_privileges_.store(true, std::memory_order_relaxed);
        LogCuptiErrorIfFailed(this->name(), "cuptiProfilerGetCounterAvailability(data)", res);
        return false;
    }

    CUpti_Profiler_Host_Initialize_Params hostInit = {};
    hostInit.structSize = CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_Initialize_Params, pHostObject);
    hostInit.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    hostInit.pChipName = ctx_.chip_name.c_str();
    hostInit.pCounterAvailabilityImage = counter_availability_image_.data();
    res = cuptiProfilerHostInitialize(&hostInit);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerHostInitialize", res)) return false;
    host_object_ = hostInit.pHostObject;

    metric_name_ptrs_.clear();
    metric_name_ptrs_.reserve(metrics_.size());
    for (const auto& metric : metrics_) metric_name_ptrs_.push_back(metric.c_str());

    if (!BuildConfigImage_()) return false;

    CUpti_PmSampling_Enable_Params enable = {CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
    enable.deviceIndex = ctx_.device_id;
    res = cuptiPmSamplingEnable(&enable);
    if (res != CUPTI_SUCCESS) {
        if (IsPrivilegeError(res)) insufficient_privileges_.store(true, std::memory_order_relaxed);
        LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingEnable", res);
        return false;
    }
    pm_object_ = enable.pPmSamplingObject;

    CUpti_PmSampling_SetConfig_Params setConfig = {
        CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
    setConfig.pPmSamplingObject = pm_object_;
    setConfig.configSize = config_image_.size();
    setConfig.pConfig = config_image_.data();
    setConfig.hardwareBufferSize = 8 * 1024 * 1024;
    setConfig.samplingInterval = static_cast<uint64_t>(opts_.pm_sampling_interval_us) * 1000ull;
    setConfig.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
    setConfig.hwBufferAppendMode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST;
    res = cuptiPmSamplingSetConfig(&setConfig);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingSetConfig", res)) return false;

    pm_initialized_ = true;
    operational_.store(true, std::memory_order_relaxed);
    GFL_LOG_DEBUG("[PmSamplingEngine] PM sampling configured chip=", ctx_.chip_name,
                  " metrics=", metrics_.size(),
                  " interval_ns=", setConfig.samplingInterval,
                  " max_samples=", opts_.pm_sampling_max_samples);
    return true;
}

bool PmSamplingEngine::BuildConfigImage_() {
    if (!host_object_ || metric_name_ptrs_.empty()) return false;

    CUpti_Profiler_Host_ConfigAddMetrics_Params add = {
        CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    add.pHostObject = host_object_;
    add.ppMetricNames = metric_name_ptrs_.data();
    add.numMetrics = metric_name_ptrs_.size();
    CUptiResult res = cuptiProfilerHostConfigAddMetrics(&add);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerHostConfigAddMetrics", res)) return false;

    CUpti_Profiler_Host_GetConfigImageSize_Params size = {
        CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    size.pHostObject = host_object_;
    res = cuptiProfilerHostGetConfigImageSize(&size);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerHostGetConfigImageSize", res)) return false;
    if (size.configImageSize == 0) {
        GFL_LOG_ERROR("[PmSamplingEngine] PM config image size is zero");
        return false;
    }

    config_image_.assign(size.configImageSize, 0);
    CUpti_Profiler_Host_GetConfigImage_Params image = {
        CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    image.pHostObject = host_object_;
    image.configImageSize = config_image_.size();
    image.pConfigImage = config_image_.data();
    res = cuptiProfilerHostGetConfigImage(&image);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerHostGetConfigImage", res)) return false;

    CUpti_Profiler_Host_GetNumOfPasses_Params passes = {
        CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
    passes.configImageSize = config_image_.size();
    passes.pConfigImage = config_image_.data();
    res = cuptiProfilerHostGetNumOfPasses(&passes);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerHostGetNumOfPasses", res)) return false;
    if (passes.numOfPasses > 1) {
        GFL_LOG_ERROR("[PmSamplingEngine] PM Sampling requires single-pass metrics; requested config needs ",
                      passes.numOfPasses, " passes");
        return false;
    }
    return true;
}

bool PmSamplingEngine::CreateCounterDataImage_() {
    if (!pm_object_ || metric_name_ptrs_.empty()) return false;
    CUpti_PmSampling_GetCounterDataSize_Params size = {
        CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
    size.pPmSamplingObject = pm_object_;
    size.pMetricNames = metric_name_ptrs_.data();
    size.numMetrics = metric_name_ptrs_.size();
    size.maxSamples = opts_.pm_sampling_max_samples;
    CUptiResult res = cuptiPmSamplingGetCounterDataSize(&size);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingGetCounterDataSize", res)) return false;
    if (size.counterDataSize == 0) {
        GFL_LOG_ERROR("[PmSamplingEngine] PM counter data image size is zero");
        return false;
    }

    counter_data_image_.assign(size.counterDataSize, 0);
    CUpti_PmSampling_CounterDataImage_Initialize_Params init = {
        CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    init.pPmSamplingObject = pm_object_;
    init.counterDataSize = counter_data_image_.size();
    init.pCounterData = counter_data_image_.data();
    res = cuptiPmSamplingCounterDataImageInitialize(&init);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingCounterDataImageInitialize", res)) return false;
    return true;
}

void PmSamplingEngine::DecodeAndEmit_() {
    if (!pm_object_ || !host_object_ || counter_data_image_.empty()) return;

    CUpti_PmSampling_DecodeData_Params decode = {
        CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
    decode.pPmSamplingObject = pm_object_;
    decode.pCounterDataImage = counter_data_image_.data();
    decode.counterDataImageSize = counter_data_image_.size();
    CUptiResult res = cuptiPmSamplingDecodeData(&decode);
    if (res != CUPTI_SUCCESS && res != CUPTI_ERROR_OUT_OF_MEMORY) {
        LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingDecodeData", res);
        return;
    }
    if (decode.overflow || res == CUPTI_ERROR_OUT_OF_MEMORY) {
        GFL_LOG_ERROR("[PmSamplingEngine] PM sampling hardware buffer overflow; increase pm_sampling_max_samples or interval_us");
    }

    CUpti_PmSampling_GetCounterDataInfo_Params info = {
        CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
    info.pCounterDataImage = counter_data_image_.data();
    info.counterDataImageSize = counter_data_image_.size();
    res = cuptiPmSamplingGetCounterDataInfo(&info);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingGetCounterDataInfo", res)) return;

    std::vector<PmSampleInput> rows;
    const size_t completed = std::min<size_t>(info.numCompletedSamples,
                                              opts_.pm_sampling_max_samples);
    rows.reserve(completed * metrics_.size());
    std::vector<double> values(metrics_.size(), 0.0);

    for (size_t sample = 0; sample < completed; ++sample) {
        CUpti_PmSampling_CounterData_GetSampleInfo_Params sampleInfo = {
            CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
        sampleInfo.pPmSamplingObject = pm_object_;
        sampleInfo.pCounterDataImage = counter_data_image_.data();
        sampleInfo.counterDataImageSize = counter_data_image_.size();
        sampleInfo.sampleIndex = sample;
        res = cuptiPmSamplingCounterDataGetSampleInfo(&sampleInfo);
        if (LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingCounterDataGetSampleInfo", res)) continue;

        CUpti_Profiler_Host_EvaluateToGpuValues_Params eval = {
            CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
        eval.pHostObject = host_object_;
        eval.pCounterDataImage = counter_data_image_.data();
        eval.counterDataImageSize = counter_data_image_.size();
        eval.rangeIndex = sample;
        eval.ppMetricNames = metric_name_ptrs_.data();
        eval.numMetrics = metric_name_ptrs_.size();
        eval.pMetricValues = values.data();
        res = cuptiProfilerHostEvaluateToGpuValues(&eval);
        if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerHostEvaluateToGpuValues", res)) continue;

        const uint64_t mid = sampleInfo.startTimestamp +
            ((sampleInfo.endTimestamp - sampleInfo.startTimestamp) / 2ull);
        for (size_t metric = 0; metric < metrics_.size(); ++metric) {
            PmSampleInput row;
            row.sample_index = static_cast<uint32_t>(sample);
            row.ts_ns = static_cast<int64_t>(mid);
            row.device_id = ctx_.device_id;
            row.metric_name = metrics_[metric];
            row.value = values[metric];
            rows.push_back(std::move(row));
        }
    }

    if (!rows.empty()) {
        Monitor::PushPmSamples(rows);
        produced_data_.store(true, std::memory_order_relaxed);
    }
    GFL_LOG_DEBUG("[PmSamplingEngine] decoded PM samples completed=", completed,
                  " populated=", info.numPopulatedSamples,
                  " rows=", rows.size());
}

void PmSamplingEngine::DisablePmSampling_() {
    std::lock_guard<std::mutex> lk(pm_mu_);
    if (running_ && pm_object_) {
        CUpti_PmSampling_Stop_Params stop = {CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
        stop.pPmSamplingObject = pm_object_;
        LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingStop", cuptiPmSamplingStop(&stop));
        running_ = false;
    }
    if (pm_object_) {
        CUpti_PmSampling_Disable_Params p = {CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
        p.pPmSamplingObject = pm_object_;
        LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingDisable", cuptiPmSamplingDisable(&p));
        pm_object_ = nullptr;
        pm_initialized_ = false;
    }
    if (host_object_) {
        CUpti_Profiler_Host_Deinitialize_Params deinit = {
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        deinit.pHostObject = host_object_;
        LogCuptiErrorIfFailed(this->name(), "cuptiProfilerHostDeinitialize", cuptiProfilerHostDeinitialize(&deinit));
        host_object_ = nullptr;
    }
    if (profiler_initialized_ && profiler_init_owned_) {
        CUpti_Profiler_DeInitialize_Params deinit = {
            CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
        LogCuptiErrorIfFailed(this->name(), "cuptiProfilerDeInitialize", cuptiProfilerDeInitialize(&deinit));
    }
    profiler_initialized_ = false;
    profiler_init_owned_ = false;
}

void PmSamplingEngine::StartPmSampling_() {
    std::lock_guard<std::mutex> lk(pm_mu_);
    if (running_) return;
    attempted_.store(true, std::memory_order_relaxed);
    if (!config_emitted_) {
        EmitConfig_();
        config_emitted_ = true;
    }
    if (!InitializePmSampling_()) {
        operational_.store(false, std::memory_order_relaxed);
        return;
    }
    if (!CreateCounterDataImage_()) {
        operational_.store(false, std::memory_order_relaxed);
        return;
    }

    CUpti_PmSampling_Start_Params start = {CUpti_PmSampling_Start_Params_STRUCT_SIZE};
    start.pPmSamplingObject = pm_object_;
    CUptiResult res = cuptiPmSamplingStart(&start);
    if (LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingStart", res)) {
        operational_.store(false, std::memory_order_relaxed);
        return;
    }
    running_ = true;
    operational_.store(true, std::memory_order_relaxed);
    GFL_LOG_DEBUG("[PmSamplingEngine] >>> STARTED (Scope Begin) <<<");
}

void PmSamplingEngine::StopPmSampling_() {
    std::lock_guard<std::mutex> lk(pm_mu_);
    if (!running_ || !pm_object_) return;
    cudaDeviceSynchronize();
    CUpti_PmSampling_Stop_Params stop = {CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
    stop.pPmSamplingObject = pm_object_;
    CUptiResult res = cuptiPmSamplingStop(&stop);
    if (!LogCuptiErrorIfFailed(this->name(), "cuptiPmSamplingStop", res)) {
        GFL_LOG_DEBUG("[PmSamplingEngine] <<< COLLECTING (Scope End) >>>");
        DecodeAndEmit_();
    }
    running_ = false;
}
#endif

}  // namespace gpufl
