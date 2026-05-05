#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gpufl {
struct HostSample {
    double cpu_util_percent = 0.0;  // System-wide CPU usage (0.0 - 100.0)
    uint64_t ram_used_mib = 0;
    uint64_t ram_total_mib = 0;
};

struct GpuStaticDeviceInfo {
    int id = 0;
    std::string name;
    std::string uuid;
    std::string vendor;
    std::string architecture;
    int compute_major = 0;
    int compute_minor = 0;
    int l2_cache_size = 0;
    int shared_mem_per_block = 0;
    int regs_per_block = 0;
    int multi_processor_count = 0;
    int warp_size = 0;
};

struct DeviceSample {
    int device_id = 0;
    std::string name;
    std::string uuid;
    std::string vendor;
    int pci_bus_id = 0;

    size_t free_mib = 0;
    size_t total_mib = 0;
    size_t used_mib = 0;

    unsigned int gpu_util = 0;   // %
    unsigned int mem_util = 0;   // %
    unsigned int temp_c = 0;     // Celsius
    unsigned int power_mw = 0;   // Milliwatts
    unsigned int clock_gfx = 0;  // MHz
    unsigned int clock_sm = 0;   // MHz
    unsigned int clock_mem = 0;  // MHz

    // Extended metrics (AMD ROCm SMI)
    unsigned int fan_speed_pct = 0;    // Fan speed 0-100%
    unsigned int temp_mem_c = 0;       // Memory temperature, Celsius
    unsigned int temp_junction_c = 0;  // Junction temperature, Celsius
    unsigned int voltage_mv = 0;       // GFX voltage, millivolts
    uint64_t energy_uj = 0;            // Cumulative energy, microjoules
    uint64_t ecc_corrected = 0;        // Correctable ECC error count
    uint64_t ecc_uncorrected = 0;      // Uncorrectable ECC error count

    bool throttle_power;    // True if hitting Power CAp
    bool throttle_thermal;  // True if slowing down due to Heat

    unsigned long long nvlink_rx_bps;  // Receive Speed
    unsigned long long nvlink_tx_bps;  // Transmit Speed

    unsigned long long pcie_rx_bps;  // Host -> Device (Upload)
    unsigned long long pcie_tx_bps;  // Device -> Host (Download)
};

struct InitEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string log_path;
    int64_t ts_ns = 0;
    HostSample host;
    std::vector<DeviceSample> devices;
    std::vector<GpuStaticDeviceInfo> gpu_static_device_infos;
};

struct ShutdownEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t ts_ns = 0;
};

struct SassConfigEvent {
    std::string session_id;
    int64_t ts_ns = 0;
    uint32_t device_id = 0;
    std::vector<std::string> configured_metrics;  // metrics successfully enabled
    std::vector<std::string> skipped_metrics;     // metrics CUPTI rejected for this GPU
};

struct KernelEvent {
    int pid = 0;
    std::string app;
    std::string name;
    std::string platform;
    std::string session_id;
    uint32_t device_id = 0;
    uint32_t stream_id = 0;

    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t api_start_ns = 0;
    int64_t api_exit_ns = 0;

    std::string grid;
    std::string block;
    bool has_details = false;
    int dyn_shared_bytes = 0;
    int num_regs = 0;
    std::size_t static_shared_bytes = 0;
    std::size_t local_bytes = 0;
    std::size_t const_bytes = 0;
    float occupancy = 0.0f;
    float reg_occupancy = 0.0f;
    float smem_occupancy = 0.0f;
    float warp_occupancy = 0.0f;
    float block_occupancy = 0.0f;
    std::string limiting_resource;
    int max_active_blocks = 0;
    unsigned int corr_id = 0;

    uint32_t local_mem_total = 0;        // total local mem across all threads (bytes)
    uint32_t local_mem_per_thread = 0;  // bytes spilled per thread (0 = no spill)

    uint8_t cache_config_requested = 0;
    uint8_t cache_config_executed = 0;
    uint32_t shared_mem_executed = 0;

    std::string user_scope;
    int scope_depth = 0;

    std::string stack_trace;

    // External correlation stamped onto this kernel by the framework
    // (PyTorch / TF / JAX). external_id == 0 means no framework tracked
    // this launch; kernel_event_model.cpp omits the columns when zero.
    uint8_t  external_kind = 0;
    uint64_t external_id   = 0;
};

struct MemcpyEvent {
    int pid = 0;
    std::string app;
    std::string name;
    std::string platform;
    std::string session_id;
    uint32_t device_id = 0;
    uint32_t stream_id = 0;

    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t api_start_ns = 0;
    int64_t api_exit_ns = 0;

    unsigned int corr_id = 0;
    std::string user_scope;
    int scope_depth = 0;
    std::string stack_trace;

    uint64_t bytes = 0;
    std::string copy_kind;
    std::string src_kind;
    std::string dst_kind;
};

struct MemsetEvent {
    int pid = 0;
    std::string app;
    std::string name;
    std::string platform;
    std::string session_id;
    uint32_t device_id = 0;
    uint32_t stream_id = 0;

    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t api_start_ns = 0;
    int64_t api_exit_ns = 0;

    unsigned int corr_id = 0;
    std::string user_scope;
    int scope_depth = 0;
    std::string stack_trace;

    uint64_t bytes = 0;
};

struct ProfileSampleEvent {
    int pid = 0;
    std::string app;
    std::string session_id;

    int64_t ts_ns = 0;
    uint32_t device_id = 0;
    uint32_t corr_id = 0;
    uint32_t samples_count = 0;
    uint32_t stall_reason = 0;
    std::string reason_name;
    std::string sample_kind;  // "pc_sampling" | "sass_metric"

    std::string source_file;
    std::string function_name;
    uint32_t source_line = 0;

    // SASS Metrics
    std::string metric_name;
    uint64_t metric_value = 0;
    uint32_t pc_offset = 0;
};

struct ScopeBeginEvent {
    uint64_t scope_id = 0;
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;
    std::string tag;
    int64_t ts_ns = 0;

    HostSample host;
    std::vector<DeviceSample> devices;

    std::string user_scope;
    int scope_depth = 0;
};

struct ScopeEndEvent {
    uint64_t scope_id = 0;
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;
    std::string tag;
    int64_t ts_ns = 0;

    HostSample host;
    std::vector<DeviceSample> devices;

    std::string user_scope;
    int scope_depth = 0;
};

struct SystemStartEvent {
    int pid{};
    std::string app;
    std::string name;
    std::string session_id;
    int64_t ts_ns{};

    HostSample host;
    std::vector<DeviceSample> devices;
};

struct SystemSampleEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;
    int64_t ts_ns = 0;

    HostSample host;
    std::vector<DeviceSample> devices;
};

struct SystemStopEvent {
    int pid{};
    std::string app;
    std::string session_id;
    std::string name;
    int64_t ts_ns{};

    HostSample host;
    std::vector<DeviceSample> devices;
};

// ── Batch row types (used by BatchBuffer, no heap strings) ────────────────

// One synchronization API call — `cudaStreamSynchronize` /
// `cudaDeviceSynchronize` / `cudaEventSynchronize` / `cuStreamWaitEvent`.
// Replaces the per-event `SynchronizationEvent` JSON with a packed row
// inside `synchronization_event_batch`. Cuts wire bytes ~14× on real
// workloads where the same call site fires repeatedly:
//   - The per-event envelope (type/pid/app/session_id) amortizes across
//     up to kMaxRows rows in the batch.
//   - `stack_trace` (typically 250+ bytes of nearly-identical text per
//     event in a hot loop) becomes a `function_id` interned via
//     `DictionaryManager::internFunction` and shipped exactly once per
//     unique stack via the existing `dictionary_update` flush.
struct SynchronizationEventBatchRow {
    int64_t  start_ns    = 0;   // absolute wall clock
    int64_t  duration_ns = 0;
    uint8_t  sync_type   = 0;   // CUpti_ActivitySynchronizationType (1..4)
    uint32_t stream_id   = 0;   // 0 = device-wide / context sync
    uint32_t event_id    = 0;   // 0 = no event handle
    uint32_t context_id  = 0;
    uint32_t corr_id     = 0;
    uint32_t function_id = 0;   // DictionaryManager::internFunction(stack_trace); 0 = no stack
};

// One CUPTI MEMORY2 record — `cudaMalloc` / `cudaFree` / `cudaMallocAsync` /
// etc. Replaces per-event `memory_alloc_event` JSON with a packed row
// inside `memory_alloc_event_batch`. Pure-numeric fields → no dictionary
// encoding, just envelope amortization. Saves ~85% on alloc-heavy
// workloads.
struct MemoryAllocEventBatchRow {
    int64_t  start_ns    = 0;
    int64_t  duration_ns = 0;   // 0 in v1 — CUPTI doesn't emit alloc duration
    uint8_t  memory_op   = 0;   // 1=ALLOC, 2=FREE
    uint8_t  memory_kind = 0;   // CUpti_ActivityMemoryKind
    uint64_t address     = 0;   // GPU virtual address
    uint64_t bytes       = 0;
    uint32_t device_id   = 0;
    uint32_t stream_id   = 0;
    uint32_t corr_id     = 0;
};

struct KernelBatchRow {
    int64_t  start_ns    = 0;  // absolute GPU execution start
    uint32_t kernel_id   = 0;  // name dictionary ID
    uint32_t stream_id   = 0;  // raw CUDA stream ID
    int64_t  duration_ns = 0;
    unsigned corr_id     = 0;
    int      dyn_shared  = 0;
    int      num_regs    = 0;
    uint8_t  has_details = 0;  // 1 → a kernel_detail event follows with same corr_id

    // Framework-emitted external correlation, sourced from
    // CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION records. Stamped onto
    // the kernel by KernelLaunchHandler::handleActivityRecord; ferried
    // through the ActivityRecord into this row by CollectorLoop.
    // external_id == 0 means "no framework was tracking this kernel"
    // and the column is omitted from the JSON to keep the wire compact.
    uint8_t  external_kind = 0;
    uint64_t external_id   = 0;
};

struct KernelDetailRow {
    unsigned     corr_id = 0;
    std::string  session_id;
    int          pid = 0;
    std::string  app;
    int grid_x = 0, grid_y = 0, grid_z = 0;
    int block_x = 0, block_y = 0, block_z = 0;
    int  static_shared = 0;
    int  local_bytes   = 0;
    int  const_bytes   = 0;
    float occupancy       = 0.0f;
    float reg_occupancy   = 0.0f;
    float smem_occupancy  = 0.0f;
    float warp_occupancy  = 0.0f;
    float block_occupancy = 0.0f;
    char  limiting_resource[16]{};
    int   max_active_blocks      = 0;
    uint32_t local_mem_total     = 0;
    uint32_t local_mem_per_thread = 0;
    uint8_t  cache_config_requested = 0;
    uint8_t  cache_config_executed  = 0;
    uint32_t shared_mem_executed    = 0;
    std::string user_scope;
    std::string stack_trace;
};

struct MemcpyBatchRow {
    int64_t  start_ns    = 0;
    uint32_t stream_id   = 0;
    int64_t  duration_ns = 0;
    uint64_t bytes       = 0;
    uint32_t copy_kind   = 0;  // numeric CUpti kind value
    unsigned corr_id     = 0;
};

struct DeviceMetricBatchRow {
    int64_t  ts_ns     = 0;  // absolute timestamp
    int      device_id = 0;
    unsigned gpu_util  = 0;  // %
    unsigned mem_util  = 0;  // %
    unsigned temp_c    = 0;
    unsigned power_mw  = 0;
    uint64_t used_mib  = 0;
    uint64_t total_mib = 0;
    unsigned clock_sm  = 0;  // MHz
    // Extended metrics
    unsigned fan_speed_pct   = 0;  // %
    unsigned temp_mem_c      = 0;  // Celsius
    unsigned temp_junction_c = 0;  // Celsius
    unsigned voltage_mv      = 0;  // millivolts
    uint64_t energy_uj       = 0;  // cumulative microjoules
    unsigned clock_mem       = 0;  // MHz
    uint64_t pcie_bw_bps     = 0;  // bytes/sec (rx+tx combined)
    uint64_t ecc_corrected   = 0;
    uint64_t ecc_uncorrected = 0;
};

struct HostMetricBatchRow {
    int64_t  ts_ns         = 0;   // absolute timestamp
    uint32_t cpu_pct_x100  = 0;   // cpu_util_percent × 100 (2 decimal places)
    uint64_t ram_used_mib  = 0;
    uint64_t ram_total_mib = 0;
};

struct ScopeBatchRow {
    int64_t  ts_ns             = 0;  // absolute timestamp
    uint64_t scope_instance_id = 0;  // monotonic ID shared by begin/end pair
    uint32_t name_id           = 0;  // scope name dictionary ID
    uint8_t  event_type        = 0;  // 0 = begin, 1 = end
    int      depth             = 0;
};

struct ProfileSampleBatchRow {
    int64_t  ts_ns           = 0;
    uint32_t corr_id         = 0;
    uint32_t device_id       = 0;
    uint32_t function_id     = 0;   // function_dict ID
    uint32_t pc_offset       = 0;
    uint32_t metric_id       = 0;   // metric_dict ID (0 for pc_sampling)
    uint64_t metric_value    = 0;   // metric value (sass) or sample_count (pc)
    uint32_t stall_reason    = 0;   // pc_sampling only (0 for sass)
    uint8_t  sample_kind     = 0;   // 0 = pc_sampling, 1 = sass_metric
    uint32_t scope_name_id   = 0;   // scope_name_dict ID (0 = no scope)
    uint32_t source_file_id  = 0;   // source_file_dict ID (0 = unknown)
    uint32_t source_line     = 0;   // source line number (0 = unknown)
};

struct PerfMetricEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;      // scope name
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int device_id = 0;

    // Hardware counters (-1.0 = not available for this GPU/metric)
    double sm_throughput_pct = -1.0;   // SM active % of peak
    double l1_hit_rate_pct = -1.0;     // L1 global load hit rate
    double l2_hit_rate_pct = -1.0;     // L2 read hit rate
    uint64_t dram_read_bytes = 0;      // DRAM read bytes
    uint64_t dram_write_bytes = 0;     // DRAM write bytes
    double tensor_active_pct = -1.0;   // Tensor core active % (-1 if N/A)

    std::string user_scope;
    int scope_depth = 0;
};

/**
 * NVTX range captured via CUPTI_ACTIVITY_KIND_MARKER. Sources include:
 *   - GFL_SCOPE (which auto-emits nvtxRangePushA/Pop as of the PyTorch
 *     integration work)
 *   - PyTorch's automatic NVTX annotations (via torch.cuda.nvtx or our
 *     gpufl.torch.TorchDispatchMode wrapping)
 *   - cuDNN / cuBLAS / NCCL / TensorRT which emit NVTX internally
 *   - User-emitted nvtxRangePush / nvtxMarkA calls
 *
 * Paired START/END records from CUPTI are merged in the client before
 * emitting; one NvtxMarkerEvent per completed range.
 */
struct NvtxMarkerEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;           // Range name (NVTX push argument)
    std::string domain;         // NVTX domain, "" for default
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t duration_ns = 0;    // Redundant with end-start; kept for convenience
    uint32_t marker_id = 0;     // CUPTI marker ID (for debug / dedup)
};

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
 * Per-event JSON. Volume is very low — even an inference loop that
 * launches a graph per request typically yields fewer events than
 * any other CUPTI stream we capture. Channel::Scope
 *
 * `corr_id` matches the driver-API call that issued the launch
 * (cuGraphLaunch). It does NOT match the per-node kernel records —
 * each kernel inside the graph keeps its own correlationId. To pair
 * "kernel K was part of graph G", the backend (or dashboard) needs
 * a temporal join on [start_ns, end_ns] + same stream — that's
 * deliberate v2 work, out of scope here.
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
};

/**
 * One CUDA memory-management event captured by CUPTI's
 * CUPTI_ACTIVITY_KIND_MEMORY2 stream.
 *
 * Covers cudaMalloc / cudaFree / cudaMallocAsync / cudaFreeAsync /
 * cudaMallocManaged / cudaMallocHost (and their driver-API cousins).
 * One event per call. Note that cudaMallocAsync is associated with a
 * stream and the reported {@code start_ns} is the host call time
 * (not the GPU completion time) — the host-side cost is what users
 * actually pay for in their python/c++ code.
 *
 * Per-event JSON. Volume in PyTorch workloads is typically <1k events
 * per session because torch's caching allocator absorbs most python-
 * level allocations; only large-block CUDA-level mallocs reach this
 * stream. TensorFlow eager mode is the high-volume edge case — if it
 * becomes a problem the gating flag {@code enable_memory_tracking}
 * lets users opt out without losing other CUPTI streams.
 *
 * The {@code address} field is the VA returned by cudaMalloc (or
 * being freed by cudaFree). Pairing alloc → free across the session
 * for leak / fragmentation analysis is a v2 follow-up; v1 just
 * stores raw events.
 */
struct MemoryAllocEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t start_ns = 0;
    int64_t duration_ns = 0;     // host-side; usually tiny but non-zero
    uint8_t  memory_op = 0;       // 1 = ALLOC, 2 = FREE
    uint8_t  memory_kind = 0;     // CUpti_ActivityMemoryKind
    uint64_t address = 0;
    uint64_t bytes = 0;
    uint32_t device_id = 0;
    uint32_t stream_id = 0;       // for cudaMallocAsync; 0 otherwise
    uint32_t corr_id = 0;
};

/**
 * CUDA synchronization event captured by CUPTI.
 *
 * One event per cudaStreamSynchronize / cudaDeviceSynchronize /
 * cudaEventSynchronize / cuStreamWaitEvent call (and their driver-API
 * cousins). The wall-clock duration here is the CPU-side time the
 * thread was blocked — which is the exact metric that explains GPU
 * underutilization on workloads that interleave host-side python with
 * synchronous waits (PyTorch's `torch.cuda.synchronize()` between
 * forward / backward; eager-mode TF; manual debugging code).
 *
 * Per-event JSON (not batched). Volume is hundreds-to-thousands per
 * session in typical workloads — well within per-event capacity. If a
 * user runs a stress test that produces millions of syncs, switching
 * to a batched columnar format is a one-file change (mirrors the
 * KernelEventBatch pattern).
 *
 * `sync_type` is the integer from CUPTI's CUpti_ActivitySynchronizationType
 * enum; the dashboard renders it as a human label
 * (EventSynchronize / StreamWaitEvent / StreamSynchronize / ContextSynchronize).
 *
 * `corr_id` joins to KernelEvent.corr_id, letting the dashboard
 * answer questions like "this matmul kernel finished at T1; the
 * `cudaStreamSynchronize` waiting for it returned at T2 — that
 * (T2 - kernel_end) gap is host-side overhead, not GPU work."
 */
struct SynchronizationEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t duration_ns = 0;
    uint8_t  sync_type = 0;       // CUpti_ActivitySynchronizationType
    uint32_t stream_id = 0;       // 0 for context-wide / device sync
    uint32_t event_id = 0;        // 0 for non-event syncs
    uint32_t corr_id = 0;         // links to KernelEvent.corr_id
    uint32_t context_id = 0;
    // User call stack at the moment cudaStreamSynchronize / etc. fired.
    // Captured by SynchronizationHandler on the API_ENTER callback when
    // opts.enable_stack_trace is on, joined to the activity record by
    // correlationId. Mirrors KernelEvent.stack_trace — same string
    // format, same downstream wiring (backend stores as inline VARCHAR).
    // Empty when stack capture is disabled OR the launch API isn't in
    // SynchronizationHandler's CBID set.
    std::string stack_trace;
};

}  // namespace gpufl
