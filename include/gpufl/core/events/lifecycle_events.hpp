#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "gpufl/core/events/sample_types.hpp"

namespace gpufl {

struct InitEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string log_path;
    int64_t ts_ns = 0;
    HostSample host;
    std::vector<DeviceSample> devices;
    std::vector<GpuStaticDeviceInfo> gpu_static_device_infos;
    // session_kind     : "trace" | "monitor" - vendor-agnostic.
    //                    Pre-Phase-A this drove a Traces / Monitor-
    //                    streams tab split on the frontend; the split
    //                    was removed in May 2026 (the kernel-data
    //                    axis turned out more informative than the
    //                    engine-ran axis), but session_kind is still
    //                    emitted so older deployments survive +
    //                    analytics can still ask "what fraction of
    //                    sessions ran with an engine?".
    // profiling_engine : vendor-namespaced detail like
    //                    "nvidia.pc_sampling" / "nvidia.sass_metrics"
    //                    / "nvidia.none" / "amd.none" / "metal.none".
    //                    The *.none forms identify telemetry-only sessions.
    //                    Stored verbatim by the backend.
    //                    The explicit none string lets the backend
    //                    distinguish "user explicitly disabled
    //                    profiling" from "pre-V40 client that omitted
    //                    the field" - both used to collapse to NULL.
    // Both populated in gpufl::init() from the resolved
    // MonitorOptions::profiling_engine. The C++ enum → string mapping
    // lives next to the InitEvent build site (gpufl.cpp).
    std::string session_kind;
    std::string profiling_engine;
    // Telemetry provider selected after auto-detection: nvidia, amd, or metal.
    // Additive and omitted when no GPU telemetry provider was initialized.
    std::string telemetry_backend;
    // Multi-pass profiling grouping (P1 of the multi-pass workstream).
    // A single "analysis" = N separately-launched passes (one CUPTI engine
    // each, isolated to dodge the SASS/kernel-activity deadlock + cross-
    // perturbation) that the backend stitches back into one kernel view.
    // The launcher's multi-pass driver sets GPUFL_ANALYSIS_ID/PASS_INDEX/
    // PASS_COUNT in each child; gpufl::init() reads them into these fields.
    //   analysis_id : stable id shared by every pass of one analysis
    //                 (empty for an ordinary single-pass run - then
    //                 pass_index/pass_count are NOT emitted).
    //   pass_index  : 0-based position of this pass within the analysis.
    //   pass_count  : total passes planned for the analysis (lets the
    //                 backend detect a missing/failed pass).
    // All three are emitted to job_start only when analysis_id is non-empty,
    // so single runs are byte-identical to pre-P1 (and pass_index==0 is
    // never ambiguous with "unset").
    std::string analysis_id;
    int pass_index = 0;
    int pass_count = 0;

    // Long-running session segmentation. Emitted together only when run_id is
    // non-empty; ordinary sessions remain byte-compatible with the existing
    // job_start wire. This is orthogonal to analysis_id: analysis passes
    // overlay one interval, while segments concatenate adjacent intervals.
    std::string run_id;
    uint32_t segment_index = 0;

    std::string roll_chain_id;
    std::string previous_run_id;
    uint32_t part_index = 0;
};

struct ShutdownEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t ts_ns = 0;
};

// Segment lifecycle records are defined now, but runtime segmentation remains
// disabled until the coordinator and backend contracts are implemented.
// Empty nullable strings and has_requested_boundary=false serialize as JSON
// null, preserving a single exact wire shape for segment zero.
struct SegmentStartEvent {
    std::string session_id;
    std::string run_id;
    uint32_t segment_index = 0;
    int64_t ts_ns = 0;
    int64_t actual_start_ns = 0;
    std::string previous_session_id;
    bool has_requested_boundary = false;
    int64_t requested_boundary_ns = 0;
    int64_t boundary_delay_ns = 0;
    std::string deferred_by;
};

struct SegmentEndEvent {
    std::string session_id;
    std::string run_id;
    uint32_t segment_index = 0;
    int64_t ts_ns = 0;
    int64_t actual_end_ns = 0;
    bool has_requested_boundary = false;
    int64_t requested_boundary_ns = 0;
    int64_t boundary_delay_ns = 0;
    std::string end_reason;
    std::string deferred_by;
    uint64_t records_outside_segment_window = 0;
};

struct RunEndEvent {
    std::string session_id;
    std::string run_id;
    uint32_t final_segment_index = 0;
    int64_t ts_ns = 0;
    int64_t ended_ns = 0;

    std::string end_reason;
    std::string rollover_reason;
    int64_t requested_rollover_ns = 0;
    int64_t actual_rollover_ns = 0;
};

struct SassConfigEvent {
    std::string session_id;
    int64_t ts_ns = 0;
    uint32_t device_id = 0;
    std::vector<std::string> configured_metrics;  // metrics successfully enabled
    std::vector<std::string> skipped_metrics;     // metrics CUPTI rejected for this GPU
};

// Per-scope Execution Signature (P2 multi-pass determinism guard input).
// Accumulated from KERNEL_LAUNCH_META - which fires in EVERY engine mode, so
// every isolated pass (even SASS, where kernel-activity is off) has the full
// per-launch inventory. `signature` hashes the sorted MULTISET of
// (mangled kernel name, grid, block, dyn_smem) -> launch count within the scope
// (mangled is intentional: byte-identical across passes, so no demangle is
// needed here). The backend compares this fingerprint per scope across the
// passes of one analysis: equal => the launch pattern is deterministic and SASS
// metrics from one pass may be merged onto another pass's timing for that
// scope; different (e.g. cuDNN autotune changed grid/block/count) => abort the
// SASS merge for that scope. Emitted once per scope at session end.
struct ExecutionSignatureEvent {
    std::string session_id;
    int64_t     ts_ns = 0;
    std::string scope_name;        // full user-scope path; "" = global / no scope
    uint64_t    signature = 0;     // FNV-1a 64 over the sorted launch multiset
    uint64_t    launch_count = 0;  // total kernel launches attributed to the scope
    uint32_t    distinct_kernels = 0;  // distinct (name,grid,block,smem) keys
};

struct CaptureCapability {
    std::string feature;
    bool requested = false;
    std::string status;
    std::string mode;
    std::string reason_code;
    std::string message;
};

struct CaptureCapabilitiesEvent {
    std::string session_id;
    int64_t ts_ns = 0;
    std::string requested_engine;
    std::string selected_engine;
    std::vector<CaptureCapability> capabilities;
};

}  // namespace gpufl
