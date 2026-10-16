#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpufl/backends/nvidia/cupti_engine_selection.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl {

struct CaptureCapabilityCounters {
    uint64_t kernel_rows = 0;
    uint64_t memory_rows = 0;
    uint64_t mem_transfer_rows = 0;
    uint64_t sync_rows = 0;
    uint64_t nvtx_rows = 0;
    uint64_t graph_rows = 0;
    /** GRAPH_TRACE records discarded because their timing was unusable. */
    uint64_t graph_rows_dropped = 0;
    uint64_t external_rows = 0;
    uint64_t source_rows = 0;
    uint64_t function_rows = 0;
    uint64_t launch_count = 0;
    /**
     * Activity records that reached the timestamp check (intentionally
     * filtered records excluded), and the ones it rejected: start == end == 0
     * (CUPTI's "timestamps unavailable") and any other invalid timing.
     */
    uint64_t kernel_records_eligible = 0;
    uint64_t kernel_records_zero_ts = 0;
    uint64_t kernel_records_invalid_ts = 0;
    uint64_t mem_transfer_records_eligible = 0;
    uint64_t mem_transfer_records_zero_ts = 0;
    uint64_t mem_transfer_records_invalid_ts = 0;
};

enum class ActivityTimestampOutcome {
    /** Nothing was rejected for its timing (or nothing arrived). */
    Complete,
    /** Records arrived, none was emitted, and every one had start == end == 0. */
    Unavailable,
    /** None was emitted, and the rejected ones were not all zero-zero. */
    Invalid,
    /** Some records were rejected for their timing; the rest still count. */
    Partial,
};

ActivityTimestampOutcome ClassifyActivityTimestamps(uint64_t eligible,
                                                    uint64_t emitted,
                                                    uint64_t zero_ts,
                                                    uint64_t invalid_ts);

/** "eligible=N zero_timestamps=Z other_invalid=I" for capability messages and logs. */
std::string DescribeActivityTimestamps(uint64_t eligible, uint64_t zero_ts,
                                       uint64_t invalid_ts);

struct CaptureCapabilityOptions {
    bool enable_memory_tracking = true;
    bool enable_external_correlation = true;
    bool enable_synchronization = true;
    bool enable_cuda_graphs_tracking = false;
};

struct CaptureCapabilityInput {
    std::string session_id;
    int64_t ts_ns = 0;
    ProfilingEngine requested_engine = ProfilingEngine::Monitor;
    bool combo_active = false;

    EngineRequestSet requests;
    EngineRuntimeState engine_state;
    CaptureCapabilityCounters counters;
    CaptureCapabilityOptions options;

    bool kernel_activity = false;
    /** This trace intentionally has no CUDA kernel launch (host-only lesson). */
    bool expect_no_kernel_events = false;
    bool cubin_requested = false;
    bool cubin_capture = false;
    bool sass_metrics_only = false;
    bool allow_sass_memory2_activity = true;
    bool allow_sass_sync_activity = true;
    bool allow_sass_graph_activity = true;
    bool allow_sass_external_correlation = true;

    bool pc_insufficient_privileges = false;
    bool pc_stall_reasons_unavailable = false;
    bool pc_no_cuda_context = false;
};

CaptureCapabilitiesEvent BuildCaptureCapabilitiesEvent(
    const CaptureCapabilityInput& input);

std::vector<std::string> BuildCaptureCapabilityWarnings(
    const CaptureCapabilityInput& input);

}  // namespace gpufl
