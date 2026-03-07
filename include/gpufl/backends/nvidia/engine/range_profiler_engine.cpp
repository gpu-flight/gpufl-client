#include "gpufl/backends/nvidia/engine/range_profiler_engine.hpp"

#if GPUFL_HAS_PERFWORKS
#include <cupti.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <nvperf_host.h>   // NVPW_CounterDataBuilder_* — deprecated but still
                           // required in CUDA ≤13.x; no CUPTI Host equivalent yet
#endif

#include <cmath>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/debug_logger.hpp"

namespace gpufl {

#if GPUFL_HAS_PERFWORKS
namespace {
std::vector<const char*> kPerfMetricNames = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
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
    if (!perf_session_active_) {
        InitPerfworksSession_();
    }
#else
    GFL_LOG_ERROR("[RangeProfilerEngine] Not built with GPUFL_HAS_PERFWORKS");
#endif
}

void RangeProfilerEngine::stop() {}

void RangeProfilerEngine::shutdown() {
#if GPUFL_HAS_PERFWORKS
    if (range_profiler_object_) {
        CUpti_RangeProfiler_Disable_Params p = {
            CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        LogCuptiErrorIfFailed("RangeProfiler", "cuptiRangeProfilerDisable",
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
        LogCuptiErrorIfFailed("Perfworks", "cuptiProfilerDeInitialize",
                              cuptiProfilerDeInitialize(&dp));
        perf_session_active_ = false;
    }
#endif
}

void RangeProfilerEngine::onPerfScopeStart(const char* name) {
#if GPUFL_HAS_PERFWORKS
    GFL_LOG_DEBUG("[RangeProfilerEngine] onPerfScopeStart name=",
                  (name ? name : "(null)"), " active=", perf_session_active_);
    if (!perf_session_active_) {
        if (!InitPerfworksSession_()) return;
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
        if (LogCuptiErrorIfFailed("RangeProfiler", "cuptiRangeProfilerStart",
                                  cuptiRangeProfilerStart(&p))) {
            return;
        }
    }
    {
        CUpti_RangeProfiler_PushRange_Params p = {
            CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        p.pRangeName           = name;
        LogCuptiErrorIfFailed("RangeProfiler", "cuptiRangeProfilerPushRange",
                              cuptiRangeProfilerPushRange(&p));
    }
#endif
}

void RangeProfilerEngine::onPerfScopeStop(const char* name) {
#if GPUFL_HAS_PERFWORKS
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
        LogCuptiErrorIfFailed("RangeProfiler", "cuptiRangeProfilerPopRange",
                              cuptiRangeProfilerPopRange(&p));
    }
    {
        CUpti_RangeProfiler_Stop_Params p = {
            CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        if (LogCuptiErrorIfFailed("RangeProfiler", "cuptiRangeProfilerStop",
                                  cuptiRangeProfilerStop(&p))) {
            return;
        }
        if (!p.isAllPassSubmitted) {
            GFL_LOG_DEBUG("[RangeProfiler] Additional replay passes required; "
                          "metrics may be partial in this run");
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

// ---- Private (Perfworks) ---------------------------------------------------

#if GPUFL_HAS_PERFWORKS

bool RangeProfilerEngine::InitPerfworksSession_() {
    if (!ctx_.cuda_ctx) {
        GFL_LOG_ERROR("[RangeProfilerEngine] No CUDA context available");
        return false;
    }

    // Initialize the profiler device
    {
        CUpti_Profiler_Initialize_Params p = {
            CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        if (LogCuptiErrorIfFailed("Perfworks", "cuptiProfilerInitialize",
                                  cuptiProfilerInitialize(&p))) {
            return false;
        }
    }

    if (ctx_.chip_name.empty()) {
        if (LogCuptiErrorIfFailed("Perfworks", "cuptiGetDeviceId",
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

    {
        CUpti_Profiler_Host_Initialize_Params hi = {
            CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
        hi.profilerType              = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
        hi.pChipName                 = ctx_.chip_name.c_str();
        hi.pCounterAvailabilityImage = counterAvailImage.data();
        if (LogCuptiErrorIfFailed("Perfworks", "cuptiProfilerHostInitialize",
                                  cuptiProfilerHostInitialize(&hi))) {
            return false;
        }
        perf_host_object_ = hi.pHostObject;
    }
    {
        CUpti_Profiler_Host_ConfigAddMetrics_Params am = {
            CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
        am.pHostObject    = perf_host_object_;
        am.ppMetricNames  = kPerfMetricNames.data();
        am.numMetrics     = kPerfMetricNames.size();
        if (LogCuptiErrorIfFailed("Perfworks",
                                  "cuptiProfilerHostConfigAddMetrics",
                                  cuptiProfilerHostConfigAddMetrics(&am))) {
            return false;
        }
    }
    {
        CUpti_Profiler_Host_GetConfigImageSize_Params gs = {
            CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
        gs.pHostObject = perf_host_object_;
        if (LogCuptiErrorIfFailed("Perfworks",
                                  "cuptiProfilerHostGetConfigImageSize",
                                  cuptiProfilerHostGetConfigImageSize(&gs))) {
            return false;
        }
        perf_config_image_.resize(gs.configImageSize);
    }
    {
        CUpti_Profiler_Host_GetConfigImage_Params gc = {
            CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
        gc.pHostObject      = perf_host_object_;
        gc.configImageSize  = perf_config_image_.size();
        gc.pConfigImage     = perf_config_image_.data();
        if (LogCuptiErrorIfFailed("Perfworks",
                                  "cuptiProfilerHostGetConfigImage",
                                  cuptiProfilerHostGetConfigImage(&gc))) {
            return false;
        }
    }

    // Build counter data prefix via NVPW_CounterDataBuilder_*.
    // TODO: migrate to cuptiProfilerHostGetCounterDataPrefixImage* once the
    //       CUDA toolkit provides those symbols (not available in CUDA ≤13.x).
    std::vector<uint8_t> counterDataPrefix;
    {
        NVPW_InitializeHost_Params ihp = {NVPW_InitializeHost_Params_STRUCT_SIZE};
        if (NVPW_InitializeHost(&ihp) != NVPA_STATUS_SUCCESS) {
            GFL_LOG_ERROR("[RangeProfilerEngine] NVPW_InitializeHost failed");
            return false;
        }

        NVPW_CounterDataBuilder_Create_Params cbcp = {
            NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE};
        cbcp.pChipName = ctx_.chip_name.c_str();
        if (NVPW_CounterDataBuilder_Create(&cbcp) != NVPA_STATUS_SUCCESS) {
            GFL_LOG_ERROR("[RangeProfilerEngine] NVPW_CounterDataBuilder_Create failed");
            return false;
        }
        NVPA_CounterDataBuilder* builder = cbcp.pCounterDataBuilder;

        {
            std::vector<NVPA_RawMetricRequest> reqs(kPerfMetricNames.size());
            for (size_t i = 0; i < kPerfMetricNames.size(); ++i) {
                reqs[i].structSize    = NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE;
                reqs[i].pMetricName   = kPerfMetricNames[i];
                reqs[i].isolated      = 1;
                reqs[i].keepInstances = 1;
            }
            NVPW_CounterDataBuilder_AddMetrics_Params amp = {
                NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE};
            amp.pCounterDataBuilder = builder;
            amp.pRawMetricRequests  = reqs.data();
            amp.numMetricRequests   = kPerfMetricNames.size();
            NVPW_CounterDataBuilder_AddMetrics(&amp);
        }
        {
            NVPW_CounterDataBuilder_GetCounterDataPrefix_Params gcp = {
                NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE};
            gcp.pCounterDataBuilder = builder;
            gcp.bytesAllocated      = 0;
            gcp.pBuffer             = nullptr;
            NVPW_CounterDataBuilder_GetCounterDataPrefix(&gcp);
            counterDataPrefix.resize(gcp.bytesCopied);
            gcp.bytesAllocated = counterDataPrefix.size();
            gcp.pBuffer        = counterDataPrefix.data();
            NVPW_CounterDataBuilder_GetCounterDataPrefix(&gcp);
        }
        {
            NVPW_CounterDataBuilder_Destroy_Params dp = {
                NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
            dp.pCounterDataBuilder = builder;
            NVPW_CounterDataBuilder_Destroy(&dp);
        }
    }

    CUpti_Profiler_CounterDataImageOptions cdOpts = {
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE};
    cdOpts.pCounterDataPrefix    = counterDataPrefix.data();
    cdOpts.counterDataPrefixSize = counterDataPrefix.size();
    cdOpts.maxNumRanges          = 8;
    cdOpts.maxNumRangeTreeNodes  = 8;
    cdOpts.maxRangeNameLength    = 128;

    {
        CUpti_RangeProfiler_Enable_Params p = {
            CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
        p.ctx = ctx_.cuda_ctx;
        if (LogCuptiErrorIfFailed("RangeProfiler", "cuptiRangeProfilerEnable",
                                  cuptiRangeProfilerEnable(&p))) {
            return false;
        }
        range_profiler_object_ = p.pRangeProfilerObject;
    }
    {
        CUpti_RangeProfiler_GetCounterDataSize_Params p = {
            CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
        p.pRangeProfilerObject = range_profiler_object_;
        p.pMetricNames         = kPerfMetricNames.data();
        p.numMetrics           = kPerfMetricNames.size();
        p.maxNumOfRanges       = cdOpts.maxNumRanges;
        p.maxNumRangeTreeNodes = cdOpts.maxNumRangeTreeNodes;
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
        p.range                  = CUPTI_UserRange;
        p.replayMode             = CUPTI_UserReplay;
        p.maxRangesPerPass       = 1;
        p.numNestingLevels       = 1;
        p.minNestingLevel        = 1;
        p.passIndex              = 0;
        p.targetNestingLevel     = 1;
        if (LogCuptiErrorIfFailed("RangeProfiler", "cuptiRangeProfilerSetConfig",
                                  cuptiRangeProfilerSetConfig(&p))) {
            return false;
        }
    }

    perf_session_active_ = true;
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
        if (LogCuptiErrorIfFailed("RangeProfiler",
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
        if (LogCuptiErrorIfFailed("RangeProfiler",
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

    std::vector<const char*> metricNames = kPerfMetricNames;
    std::vector<double>      values(kPerfMetricNames.size(), -1.0);

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
    if (LogCuptiErrorIfFailed("Perfworks",
                              "cuptiProfilerHostEvaluateToGpuValues",
                              cuptiProfilerHostEvaluateToGpuValues(&p))) {
        return;
    }

    std::lock_guard<std::mutex> lk(perf_mu_);
    perf_last_event_ = PerfMetricEvent{};
    if (!values.empty() && std::isfinite(values[0]))
        perf_last_event_.sm_throughput_pct = values[0];
    if (values.size() > 1)
        perf_last_event_.dram_read_bytes =
            (values[1] >= 0.0) ? static_cast<uint64_t>(values[1]) : 0;
    if (values.size() > 2)
        perf_last_event_.dram_write_bytes =
            (values[2] >= 0.0) ? static_cast<uint64_t>(values[2]) : 0;
    if (values.size() > 3 && std::isfinite(values[3]))
        perf_last_event_.tensor_active_pct = values[3];
    perf_has_event_ = true;
    GFL_LOG_DEBUG("[RangeProfilerEngine] Decoded metrics, perf_has_event_=true");
}

#endif  // GPUFL_HAS_PERFWORKS

}  // namespace gpufl
