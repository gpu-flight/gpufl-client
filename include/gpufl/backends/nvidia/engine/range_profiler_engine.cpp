#include "gpufl/backends/nvidia/engine/range_profiler_engine.hpp"

#if GPUFL_HAS_PERFWORKS
#include <cupti.h>
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/debug_logger.hpp"

namespace gpufl {

#if GPUFL_HAS_PERFWORKS
namespace {
std::vector<const char*> kPerfMetricNames = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
};
}  // namespace
#endif

bool RangeProfilerEngine::initialize(const MonitorOptions& opts,
                                      const EngineContext& ctx) {
    opts_ = opts;
    ctx_  = ctx;
    GFL_LOG_DEBUG("[RangeProfilerEngine] initialized");
    return true;
}

void RangeProfilerEngine::start() {
#if GPUFL_HAS_PERFWORKS
    attempted_.store(true, std::memory_order_relaxed);
    if (!perf_session_active_) {
        InitPerfworksSession_(mode_ == Mode::Scope);
    }
    if (mode_ == Mode::KernelReplay && perf_session_active_) {
        if (kernel_replay_running_) {
            GFL_LOG_DEBUG("[RangeProfilerKernelReplay] start skipped: already running");
            return;
        }
        CUpti_RangeProfiler_Start_Params p = {
            CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        if (LogCuptiErrorIfFailed(this->name(), "cuptiRangeProfilerStart",
                                  cuptiRangeProfilerStart(&p))) {
            return;
        }
        kernel_replay_running_ = true;
        kernel_replay_decoded_ = false;
        GFL_LOG_DEBUG("[RangeProfilerKernelReplay] started");
    }
#else
    GFL_LOG_ERROR("[RangeProfilerEngine] Not built with GPUFL_HAS_PERFWORKS");
#endif
}

void RangeProfilerEngine::stop() {
#if GPUFL_HAS_PERFWORKS
    if (mode_ != Mode::KernelReplay || !perf_session_active_) return;
    if (!kernel_replay_running_) {
        GFL_LOG_DEBUG("[RangeProfilerKernelReplay] stop skipped: not running");
        return;
    }
    CUpti_RangeProfiler_Stop_Params p = {
        CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
    p.pRangeProfilerObject = range_profiler_object_;
    CUptiResult result = cuptiRangeProfilerStop(&p);
    kernel_replay_running_ = false;
    if (LogCuptiErrorIfFailed(this->name(), "cuptiRangeProfilerStop",
                              result)) {
        return;
    }
    if (!p.isAllPassSubmitted) {
        GFL_LOG_DEBUG("[RangeProfilerKernelReplay] stop returned "
                      "isAllPassSubmitted=false");
    }
    if (!kernel_replay_decoded_) {
        DecodeKernelReplayEvents_();
        kernel_replay_decoded_ = true;
    }
#endif
}

void RangeProfilerEngine::shutdown() {
#if GPUFL_HAS_PERFWORKS
    if (mode_ == Mode::KernelReplay && kernel_replay_running_) {
        stop();
    }
    if (range_profiler_object_) {
        CUpti_RangeProfiler_Disable_Params p = {
            CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        LogCuptiErrorIfFailed(this->name(), "cuptiRangeProfilerDisable",
                              cuptiRangeProfilerDisable(&p));
        range_profiler_object_ = nullptr;
    }
    if (perf_host_object_) {
        CUpti_Profiler_Host_Deinitialize_Params hdp = {
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        hdp.pHostObject = perf_host_object_;
        cuptiProfilerHostDeinitialize(&hdp);
        perf_host_object_ = nullptr;
    }
    if (perf_session_active_) {
        CUpti_Profiler_DeInitialize_Params dp = {
            CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
        LogCuptiErrorIfFailed(this->name(), "cuptiProfilerDeInitialize",
                              cuptiProfilerDeInitialize(&dp));
        perf_session_active_ = false;
    }
    kernel_replay_running_ = false;
#endif
    operational_.store(false, std::memory_order_relaxed);
}

void RangeProfilerEngine::onPerfScopeStart(const char* name) {
#if GPUFL_HAS_PERFWORKS
    if (mode_ != Mode::Scope) return;
    attempted_.store(true, std::memory_order_relaxed);
    GFL_LOG_DEBUG("[RangeProfilerEngine] onPerfScopeStart name=",
                  name ? name : "(null)", " active=", perf_session_active_);
    if (!perf_session_active_) {
        if (!InitPerfworksSession_(true)) return;
    }
    {
        CUpti_RangeProfiler_CounterDataImage_Initialize_Params p = {
            CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        p.counterDataSize      = perf_counter_data_image_.size();
        p.pCounterData         = perf_counter_data_image_.data();
        if (LogCuptiErrorIfFailed(
                "RangeProfiler",
                "cuptiRangeProfilerCounterDataImageInitialize",
                cuptiRangeProfilerCounterDataImageInitialize(&p))) {
            return;
        }
    }
    {
        CUpti_RangeProfiler_Start_Params p = {
            CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        if (LogCuptiErrorIfFailed(this->name(), "cuptiRangeProfilerStart",
                                  cuptiRangeProfilerStart(&p))) {
            return;
        }
    }
    {
        CUpti_RangeProfiler_PushRange_Params p = {
            CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        p.pRangeName           = name;
        LogCuptiErrorIfFailed(this->name(), "cuptiRangeProfilerPushRange",
                              cuptiRangeProfilerPushRange(&p));
    }
#endif
}

void RangeProfilerEngine::onPerfScopeStop(const char* name) {
#if GPUFL_HAS_PERFWORKS
    if (mode_ != Mode::Scope) return;
    GFL_LOG_DEBUG("[RangeProfilerEngine] onPerfScopeStop name=",
                  (name ? name : "(null)"), " active=", perf_session_active_);
    if (!perf_session_active_) {
        GFL_LOG_DEBUG("[RangeProfilerEngine] onPerfScopeStop skipped: session inactive");
        return;
    }
    {
        CUpti_RangeProfiler_PopRange_Params p = {
            CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        LogCuptiErrorIfFailed(this->name(), "cuptiRangeProfilerPopRange",
                              cuptiRangeProfilerPopRange(&p));
    }
    {
        CUpti_RangeProfiler_Stop_Params p = {
            CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        if (LogCuptiErrorIfFailed(this->name(), "cuptiRangeProfilerStop",
                                  cuptiRangeProfilerStop(&p))) {
            return;
        }
        if (!p.isAllPassSubmitted) {
            GFL_LOG_DEBUG("[RangeProfiler] Additional replay passes required; "
                          "metrics may be partial in this run");
            return;
        }
    }
    EndPerfPassAndDecode_();
#endif
}

std::optional<PerfMetricEvent> RangeProfilerEngine::takeLastPerfEvent() {
#if GPUFL_HAS_PERFWORKS
    std::lock_guard<std::mutex> lk(perf_mu_);
    if (!perf_has_event_) {
        GFL_LOG_DEBUG("[RangeProfilerEngine] takeLastPerfEvent: no event available");
        return std::nullopt;
    }
    perf_has_event_ = false;
    GFL_LOG_DEBUG("[RangeProfilerEngine] takeLastPerfEvent: returning event");
    return perf_last_event_;
#else
    return std::nullopt;
#endif
}

std::vector<KernelPerfMetricEvent> RangeProfilerEngine::takeKernelPerfEvents() {
#if GPUFL_HAS_PERFWORKS
    std::lock_guard<std::mutex> lk(perf_mu_);
    std::vector<KernelPerfMetricEvent> out;
    out.swap(kernel_perf_events_);
    return out;
#else
    return {};
#endif
}

// ---- Private (Perfworks) ---------------------------------------------------

#if GPUFL_HAS_PERFWORKS

bool RangeProfilerEngine::InitPerfworksSession_() {
    return InitPerfworksSession_(mode_ == Mode::Scope);
}

bool RangeProfilerEngine::InitPerfworksSession_(bool require_single_pass) {
    if (!ctx_.cuda_ctx) {
        GFL_LOG_ERROR("[RangeProfilerEngine] No CUDA context available");
        return false;
    }

    // Initialize the profiler device
    {
        CUpti_Profiler_Initialize_Params p = {
            CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerInitialize",
                                  cuptiProfilerInitialize(&p))) {
            return false;
        }
    }

    if (ctx_.chip_name.empty()) {
        if (LogCuptiErrorIfFailed(this->name(), "cuptiGetDeviceId",
                                  cuptiGetDeviceId(ctx_.cuda_ctx,
                                                   &ctx_.device_id))) {
            return false;
        }
        ctx_.chip_name = getChipName(ctx_.device_id);
    }
    GFL_LOG_DEBUG("[RangeProfilerEngine] chip_name_ = ", ctx_.chip_name);

    std::vector<uint8_t> counterAvailImage;
    {
        CUpti_Profiler_GetCounterAvailability_Params p = {
            CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
        p.pPriv                    = nullptr;
        p.ctx                      = ctx_.cuda_ctx;
        p.pCounterAvailabilityImage = nullptr;
        if (LogCuptiErrorIfFailed(
                "Perfworks", "cuptiProfilerGetCounterAvailability(first)",
                cuptiProfilerGetCounterAvailability(&p))) {
            return false;
        }
        counterAvailImage.resize(p.counterAvailabilityImageSize);
        p.pCounterAvailabilityImage = counterAvailImage.data();
        if (LogCuptiErrorIfFailed(
                "Perfworks", "cuptiProfilerGetCounterAvailability",
                cuptiProfilerGetCounterAvailability(&p))) {
            return false;
        }
    }

    auto deinitHostObject = [&]() {
        if (!perf_host_object_) return true;
        CUpti_Profiler_Host_Deinitialize_Params p = {
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        p.pHostObject = perf_host_object_;
        perf_host_object_ = nullptr;
        return !LogCuptiErrorIfFailed(this->name(), "cuptiProfilerHostDeinitialize",
                                      cuptiProfilerHostDeinitialize(&p));
    };

    auto initHostObject = [&]() {
        CUpti_Profiler_Host_Initialize_Params hi = {
            CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
        hi.profilerType              = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
        hi.pChipName                 = ctx_.chip_name.c_str();
        hi.pCounterAvailabilityImage = counterAvailImage.data();
        if (LogCuptiErrorIfFailed(this->name(), "cuptiProfilerHostInitialize",
                                  cuptiProfilerHostInitialize(&hi))) {
            return false;
        }
        perf_host_object_ = hi.pHostObject;
        return true;
    };

    auto addMetric = [&](const char* metricName) {
        CUpti_Profiler_Host_ConfigAddMetrics_Params am = {
            CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
        am.pHostObject    = perf_host_object_;
        am.ppMetricNames  = &metricName;
        am.numMetrics     = 1;
        CUptiResult res = cuptiProfilerHostConfigAddMetrics(&am);
        if (res == CUPTI_SUCCESS) {
            active_metric_names_.push_back(metricName);
            GFL_LOG_DEBUG("[RangeProfilerEngine] accepted metric ", metricName);
            return true;
        }
        GFL_LOG_DEBUG("[RangeProfilerEngine] skipping unsupported metric ",
                      metricName, " result=", static_cast<int>(res));
        return false;
    };

    auto buildConfigImage = [&](size_t& numPasses) {
        {
            CUpti_Profiler_Host_GetConfigImageSize_Params gs = {
                CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
            gs.pHostObject = perf_host_object_;
            if (LogCuptiErrorIfFailed(this->name(),
                                      "cuptiProfilerHostGetConfigImageSize",
                                      cuptiProfilerHostGetConfigImageSize(&gs))) {
                return false;
            }
            perf_config_image_.assign(gs.configImageSize, 0);
        }
        {
            CUpti_Profiler_Host_GetConfigImage_Params gc = {
                CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
            gc.pHostObject      = perf_host_object_;
            gc.configImageSize  = perf_config_image_.size();
            gc.pConfigImage     = perf_config_image_.data();
            if (LogCuptiErrorIfFailed(this->name(),
                                      "cuptiProfilerHostGetConfigImage",
                                      cuptiProfilerHostGetConfigImage(&gc))) {
                return false;
            }
        }
        {
            CUpti_Profiler_Host_GetNumOfPasses_Params p = {
                CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
            p.configImageSize = perf_config_image_.size();
            p.pConfigImage    = perf_config_image_.data();
            if (LogCuptiErrorIfFailed(this->name(),
                                      "cuptiProfilerHostGetNumOfPasses",
                                      cuptiProfilerHostGetNumOfPasses(&p))) {
                return false;
            }
            numPasses = p.numOfPasses;
            GFL_LOG_DEBUG("[RangeProfilerEngine] config requires ",
                          numPasses, " pass(es)");
        }
        return true;
    };

    if (!initHostObject()) return false;
    active_metric_names_.clear();
    for (const char* metricName : kPerfMetricNames) {
        addMetric(metricName);
    }
    if (active_metric_names_.empty()) {
        GFL_LOG_ERROR("[RangeProfilerEngine] no Range Profiler metrics were accepted");
        return false;
    }

    size_t numPasses = 0;
    if (!buildConfigImage(numPasses)) return false;
    if (require_single_pass && numPasses > 1 && active_metric_names_.size() > 1) {
        std::vector<const char*> candidateMetricNames = active_metric_names_;
        GFL_LOG_DEBUG("[RangeProfilerEngine] metric group requires ",
                      numPasses, " passes; searching for a single-pass metric");
        bool foundSinglePassMetric = false;
        for (const char* singlePassMetric : candidateMetricNames) {
            if (!deinitHostObject()) return false;
            if (!initHostObject()) return false;
            active_metric_names_.clear();
            if (!addMetric(singlePassMetric)) continue;
            if (!buildConfigImage(numPasses)) return false;
            if (numPasses <= 1) {
                foundSinglePassMetric = true;
                GFL_LOG_DEBUG("[RangeProfilerEngine] using single-pass metric ",
                              singlePassMetric);
                break;
            }
            GFL_LOG_DEBUG("[RangeProfilerEngine] metric ", singlePassMetric,
                          " still requires ", numPasses, " passes");
        }
        if (!foundSinglePassMetric) {
            GFL_LOG_ERROR("[RangeProfilerEngine] no accepted Range Profiler "
                          "metric can be collected in one pass");
            return false;
        }
    }
    if (require_single_pass && numPasses > 1) {
        GFL_LOG_ERROR("[RangeProfilerEngine] Range Profiler metric config requires ",
                      numPasses, " passes; scope-hook profiling only supports "
                      "single-pass metrics");
        return false;
    }

    const size_t   maxNumRanges = mode_ == Mode::KernelReplay ? 1024 : 8;
    const uint32_t maxNumRangeTreeNodes =
        mode_ == Mode::KernelReplay ? 1024 : 8;

    {
        CUpti_RangeProfiler_Enable_Params p = {
            CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
        p.ctx = ctx_.cuda_ctx;
        if (LogCuptiErrorIfFailed(this->name(), "cuptiRangeProfilerEnable",
                                  cuptiRangeProfilerEnable(&p))) {
            return false;
        }
        range_profiler_object_ = p.pRangeProfilerObject;
    }
    {
        CUpti_RangeProfiler_GetCounterDataSize_Params p = {
            CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        p.pMetricNames         = active_metric_names_.data();
        p.numMetrics           = active_metric_names_.size();
        p.maxNumOfRanges       = maxNumRanges;
        p.maxNumRangeTreeNodes = maxNumRangeTreeNodes;
        if (LogCuptiErrorIfFailed(
                "RangeProfiler", "cuptiRangeProfilerGetCounterDataSize",
                cuptiRangeProfilerGetCounterDataSize(&p))) {
            return false;
        }
        perf_counter_data_image_.resize(p.counterDataSize);
    }
    {
        CUpti_RangeProfiler_CounterDataImage_Initialize_Params p = {
            CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        p.counterDataSize      = perf_counter_data_image_.size();
        p.pCounterData         = perf_counter_data_image_.data();
        if (LogCuptiErrorIfFailed(
                "RangeProfiler",
                "cuptiRangeProfilerCounterDataImageInitialize",
                cuptiRangeProfilerCounterDataImageInitialize(&p))) {
            return false;
        }
    }
    {
        CUpti_RangeProfiler_SetConfig_Params p = {
            CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
        p.pRangeProfilerObject   = range_profiler_object_;
        p.configSize             = perf_config_image_.size();
        p.pConfig                = perf_config_image_.data();
        p.counterDataImageSize   = perf_counter_data_image_.size();
        p.pCounterDataImage      = perf_counter_data_image_.data();
        p.range                  = mode_ == Mode::KernelReplay
                                   ? CUPTI_AutoRange : CUPTI_UserRange;
        p.replayMode             = mode_ == Mode::KernelReplay
                                   ? CUPTI_KernelReplay : CUPTI_UserReplay;
        p.maxRangesPerPass       = mode_ == Mode::KernelReplay
                                   ? maxNumRanges : 1;
        p.numNestingLevels       = 1;
        p.minNestingLevel        = 1;
        p.passIndex              = 0;
        p.targetNestingLevel     = 1;
        if (LogCuptiErrorIfFailed(this->name(), "cuptiRangeProfilerSetConfig",
                                  cuptiRangeProfilerSetConfig(&p))) {
            return false;
        }
    }

    perf_session_active_ = true;
    operational_.store(true, std::memory_order_relaxed);
    GFL_LOG_DEBUG("[RangeProfilerEngine] Session initialized for chip: ",
                  ctx_.chip_name);
    return true;
}

void RangeProfilerEngine::EndPerfPassAndDecode_() {
    if (!perf_host_object_ || !range_profiler_object_) {
        GFL_LOG_DEBUG("[RangeProfilerEngine] EndPerfPassAndDecode skipped: "
                      "profiler object is null");
        return;
    }
    {
        CUpti_RangeProfiler_DecodeData_Params p = {
            CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        if (LogCuptiErrorIfFailed(this->name(),
                                  "cuptiRangeProfilerDecodeData",
                                  cuptiRangeProfilerDecodeData(&p))) {
            return;
        }
        if (p.numOfRangeDropped > 0) {
            GFL_LOG_DEBUG("[RangeProfiler] Dropped ranges: ",
                          p.numOfRangeDropped);
        }
    }
    size_t numRanges = 0;
    {
        CUpti_RangeProfiler_GetCounterDataInfo_Params p = {
            CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
        p.pCounterDataImage  = perf_counter_data_image_.data();
        p.counterDataImageSize = perf_counter_data_image_.size();
        if (LogCuptiErrorIfFailed(this->name(),
                                  "cuptiRangeProfilerGetCounterDataInfo",
                                  cuptiRangeProfilerGetCounterDataInfo(&p))) {
            return;
        }
        numRanges = p.numTotalRanges;
    }
    if (numRanges == 0) {
        GFL_LOG_DEBUG("[RangeProfilerEngine] No ranges decoded for this scope");
        return;
    }

    std::vector<const char*> metricNames = active_metric_names_;
    std::vector<double>      values(active_metric_names_.size(), -1.0);

    CUpti_Profiler_Host_EvaluateToGpuValues_Params p = {
        CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    p.pHostObject         = perf_host_object_;
    p.pCounterDataImage   = perf_counter_data_image_.data();
    p.counterDataImageSize = perf_counter_data_image_.size();
    p.rangeIndex          = numRanges - 1;
    p.ppMetricNames       = metricNames.data();
    p.numMetrics          = metricNames.size();
    p.pMetricValues       = values.data();
    GFL_LOG_DEBUG("[RangeProfilerEngine] EvaluateToGpuValues rangeIndex=",
                  p.rangeIndex, " numMetrics=", p.numMetrics);
    if (LogCuptiErrorIfFailed(this->name(),
                              "cuptiProfilerHostEvaluateToGpuValues",
                              cuptiProfilerHostEvaluateToGpuValues(&p))) {
        return;
    }
    for (size_t i = 0; i < metricNames.size() && i < values.size(); ++i) {
        GFL_LOG_DEBUG("[RangeProfilerEngine] metric ", metricNames[i], " = ",
                      values[i], " finite=", std::isfinite(values[i]));
    }

    std::lock_guard<std::mutex> lk(perf_mu_);
    perf_last_event_ = PerfMetricEvent{};
    for (size_t i = 0; i < metricNames.size() && i < values.size(); ++i) {
        const char* metricName = metricNames[i];
        const double value = values[i];
        if (std::strcmp(metricName, "sm__throughput.avg.pct_of_peak_sustained_elapsed") == 0) {
            if (std::isfinite(value)) perf_last_event_.sm_throughput_pct = value;
        } else if (std::strcmp(metricName, "l1tex__t_sector_hit_rate.pct") == 0) {
            perf_last_event_.l1_hit_rate_pct = std::isfinite(value) ? value : -1.0;
        } else if (std::strcmp(metricName, "lts__t_sector_hit_rate.pct") == 0) {
            perf_last_event_.l2_hit_rate_pct = std::isfinite(value) ? value : -1.0;
        } else if (std::strcmp(metricName, "dram__bytes_read.sum") == 0) {
            perf_last_event_.dram_read_bytes =
                (value >= 0.0) ? static_cast<int64_t>(value) : -1;
        } else if (std::strcmp(metricName, "dram__bytes_write.sum") == 0) {
            perf_last_event_.dram_write_bytes =
                (value >= 0.0) ? static_cast<int64_t>(value) : -1;
        } else if (std::strcmp(metricName, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active") == 0) {
            if (std::isfinite(value)) perf_last_event_.tensor_active_pct = value;
        }
    }
    perf_has_event_ = true;
    produced_data_.store(true, std::memory_order_relaxed);
    GFL_LOG_DEBUG("[RangeProfilerEngine] Decoded metrics, perf_has_event_=true");
}

void RangeProfilerEngine::DecodeKernelReplayEvents_() {
    if (!perf_host_object_ || !range_profiler_object_) {
        GFL_LOG_DEBUG("[RangeProfilerKernelReplay] decode skipped: "
                      "profiler object is null");
        return;
    }
    {
        CUpti_RangeProfiler_DecodeData_Params p = {
            CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        if (LogCuptiErrorIfFailed(this->name(),
                                  "cuptiRangeProfilerDecodeData",
                                  cuptiRangeProfilerDecodeData(&p))) {
            return;
        }
        if (p.numOfRangeDropped > 0) {
            GFL_LOG_DEBUG("[RangeProfilerKernelReplay] Dropped ranges: ",
                          p.numOfRangeDropped);
        }
    }

    size_t numRanges = 0;
    {
        CUpti_RangeProfiler_GetCounterDataInfo_Params p = {
            CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
        p.pCounterDataImage    = perf_counter_data_image_.data();
        p.counterDataImageSize = perf_counter_data_image_.size();
        if (LogCuptiErrorIfFailed(this->name(),
                                  "cuptiRangeProfilerGetCounterDataInfo",
                                  cuptiRangeProfilerGetCounterDataInfo(&p))) {
            return;
        }
        numRanges = p.numTotalRanges;
    }
    if (numRanges == 0) {
        GFL_LOG_DEBUG("[RangeProfilerKernelReplay] no ranges decoded");
        return;
    }

    std::vector<KernelPerfMetricEvent> decoded;
    decoded.reserve(numRanges);
    std::vector<const char*> metricNames = active_metric_names_;
    std::vector<double> values(active_metric_names_.size(), -1.0);

    for (size_t rangeIndex = 0; rangeIndex < numRanges; ++rangeIndex) {
        std::fill(values.begin(), values.end(), -1.0);
        const char* rangeName = "";
        {
            CUpti_RangeProfiler_CounterData_GetRangeInfo_Params p = {
                CUpti_RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE};
            p.pCounterDataImage    = perf_counter_data_image_.data();
            p.counterDataImageSize = perf_counter_data_image_.size();
            p.rangeIndex           = rangeIndex;
            p.rangeDelimiter       = "/";
            if (!LogCuptiErrorIfFailed(
                    this->name(), "cuptiRangeProfilerCounterDataGetRangeInfo",
                    cuptiRangeProfilerCounterDataGetRangeInfo(&p))) {
                rangeName = p.rangeName ? p.rangeName : "";
            }
        }
        {
            CUpti_Profiler_Host_EvaluateToGpuValues_Params p = {
                CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
            p.pHostObject          = perf_host_object_;
            p.pCounterDataImage    = perf_counter_data_image_.data();
            p.counterDataImageSize = perf_counter_data_image_.size();
            p.rangeIndex           = rangeIndex;
            p.ppMetricNames        = metricNames.data();
            p.numMetrics           = metricNames.size();
            p.pMetricValues        = values.data();
            if (LogCuptiErrorIfFailed(
                    this->name(), "cuptiProfilerHostEvaluateToGpuValues",
                    cuptiProfilerHostEvaluateToGpuValues(&p))) {
                continue;
            }
        }

        KernelPerfMetricEvent ev;
        ev.device_id = static_cast<int>(ctx_.device_id);
        ev.range_index = rangeIndex;
        ev.range_name = rangeName;
        ev.kernel_name = rangeName;
        ev.launch_ordinal = static_cast<uint32_t>(rangeIndex + 1);
        for (size_t i = 0; i < metricNames.size() && i < values.size(); ++i) {
            const char* metricName = metricNames[i];
            const double value = values[i];
            if (std::strcmp(metricName, "sm__throughput.avg.pct_of_peak_sustained_elapsed") == 0) {
                if (std::isfinite(value)) ev.sm_throughput_pct = value;
            } else if (std::strcmp(metricName, "l1tex__t_sector_hit_rate.pct") == 0) {
                ev.l1_hit_rate_pct = std::isfinite(value) ? value : -1.0;
            } else if (std::strcmp(metricName, "lts__t_sector_hit_rate.pct") == 0) {
                ev.l2_hit_rate_pct = std::isfinite(value) ? value : -1.0;
            } else if (std::strcmp(metricName, "dram__bytes_read.sum") == 0) {
                ev.dram_read_bytes =
                    (value >= 0.0) ? static_cast<int64_t>(value) : -1;
            } else if (std::strcmp(metricName, "dram__bytes_write.sum") == 0) {
                ev.dram_write_bytes =
                    (value >= 0.0) ? static_cast<int64_t>(value) : -1;
            } else if (std::strcmp(metricName, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active") == 0) {
                if (std::isfinite(value)) ev.tensor_active_pct = value;
            }
        }
        decoded.push_back(std::move(ev));
    }

    {
        std::lock_guard<std::mutex> lk(perf_mu_);
        kernel_perf_events_.insert(kernel_perf_events_.end(),
                                   std::make_move_iterator(decoded.begin()),
                                   std::make_move_iterator(decoded.end()));
    }
    if (!decoded.empty()) {
        produced_data_.store(true, std::memory_order_relaxed);
        GFL_LOG_DEBUG("[RangeProfilerKernelReplay] decoded ",
                      decoded.size(), " kernel range event(s)");
    }
}

#endif  // GPUFL_HAS_PERFWORKS

}  // namespace gpufl
