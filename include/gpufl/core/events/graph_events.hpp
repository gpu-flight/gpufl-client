#pragma once
#include <cstdint>
#include <string>

namespace gpufl {

/**
 * One CUDA graph launch event captured by CUPTI's
 * CUPTI_ACTIVITY_KIND_GRAPH_TRACE stream.
 *
 * `cudaGraphLaunch` is the launch mechanism torch.compile / CUDA
 * Graphs / Triton-CUDA-graph-mode use to batch many kernels into a
 * single host-side launch call, eliminating per-kernel overhead.
 * This event tells the dashboard that a chunk of GPU work happened
 * as a fused graph rather than as N independent kernel launches.
 *
 * Per-event JSON. Volume is very low - even an inference loop that
 * launches a graph per request typically yields fewer events than
 * any other CUPTI stream we capture. Channel::Scope
 *
 * `corr_id` matches the driver-API call that issued the launch
 * (cuGraphLaunch). `graph_exec_key` is the opaque driver graph-execution
 * handle captured at that call; it links this timing record to the static
 * GraphNodeDefinition rows. It does NOT imply per-node timing.
 */
struct GraphLaunchEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t duration_ns = 0;
    uint32_t graph_id = 0;
    uint32_t device_id = 0;
    uint32_t stream_id = 0;
    uint32_t corr_id = 0;
    uint64_t graph_exec_key = 0;
};

/** One static node in a CUDA graph execution. */
struct GraphNodeDefinitionEvent {
    std::string session_id;
    uint64_t graph_exec_key = 0;
    uint32_t node_index = 0;
    uint32_t node_type = 0;
    uint32_t dependency_count = 0;
};

}  // namespace gpufl
