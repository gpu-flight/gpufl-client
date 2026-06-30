#include "gpufl/core/monitor_record_builders.hpp"

#include <cstring>

#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/runtime.hpp"

namespace gpufl::detail {

KernelBatchRow MakeKernelBatchRow(const ActivityRecord& rec,
                                  const uint32_t kernel_id) {
    KernelBatchRow row;
    row.start_ns = rec.cpu_start_ns;
    row.kernel_id = kernel_id;
    row.stream_id = static_cast<uint32_t>(rec.stream);
    row.duration_ns = rec.duration_ns;
    row.corr_id = rec.corr_id;
    row.dyn_shared = rec.dyn_shared;
    row.num_regs = rec.num_regs;
    row.has_details = rec.has_details ? 1 : 0;
    row.external_kind = rec.external_kind;
    row.external_id = rec.external_id;
    return row;
}

KernelDetailRow MakeKernelDetailRow(const ActivityRecord& rec,
                                    const std::string& stack_trace,
                                    const Runtime& rt) {
    KernelDetailRow detail;
    detail.corr_id = rec.corr_id;
    detail.session_id = rt.session_id;
    detail.pid = GetPid();
    detail.app = rt.app_name;
    detail.grid_x = rec.grid_x; detail.grid_y = rec.grid_y; detail.grid_z = rec.grid_z;
    detail.block_x = rec.block_x; detail.block_y = rec.block_y; detail.block_z = rec.block_z;
    detail.static_shared = rec.static_shared;
    detail.local_bytes = rec.local_bytes;
    detail.const_bytes = rec.const_bytes;
    detail.occupancy = rec.occupancy;
    detail.reg_occupancy = rec.reg_occupancy;
    detail.smem_occupancy = rec.smem_occupancy;
    detail.warp_occupancy = rec.warp_occupancy;
    detail.block_occupancy = rec.block_occupancy;
    std::memcpy(detail.limiting_resource, rec.limiting_resource, sizeof(detail.limiting_resource));
    detail.max_active_blocks = rec.max_active_blocks;
    detail.local_mem_total = rec.local_mem_total;
    detail.local_mem_per_thread = rec.local_mem_per_thread;
    detail.cache_config_requested = rec.cache_config_requested;
    detail.cache_config_executed = rec.cache_config_executed;
    detail.shared_mem_executed = rec.shared_mem_executed;
    detail.user_scope = rec.user_scope;
    detail.stack_trace = stack_trace;
    return detail;
}

MemcpyBatchRow MakeMemcpyBatchRow(const ActivityRecord& rec) {
    MemcpyBatchRow row;
    row.start_ns = rec.cpu_start_ns;
    row.stream_id = static_cast<uint32_t>(rec.stream);
    row.duration_ns = rec.duration_ns;
    row.bytes = rec.bytes;
    row.copy_kind = rec.copy_kind;
    row.corr_id = rec.corr_id;
    return row;
}

ScopeBatchRow MakeScopeBatchRow(const int64_t ts_ns, const uint64_t instance_id,
                                const uint32_t name_id,
                                const uint8_t event_type, const int depth) {
    ScopeBatchRow row;
    row.ts_ns = ts_ns;
    row.scope_instance_id = instance_id;
    row.name_id = name_id;
    row.event_type = event_type;
    row.depth = depth;
    return row;
}

ProfileSampleBatchRow MakeProfileSampleBatchRow(const ActivityRecord& rec, const uint8_t sample_kind,
                                                const uint32_t function_id,
                                                const uint32_t metric_id,
                                                const uint32_t scope_name_id,
                                                const uint32_t source_file_id) {
    ProfileSampleBatchRow row;
    row.ts_ns = rec.cpu_start_ns;
    row.corr_id = rec.corr_id;
    row.device_id = rec.device_id;
    row.function_id = function_id;
    row.pc_offset = rec.pc_offset;
    row.metric_id = metric_id;
    row.metric_value = (sample_kind == 1) ? rec.metric_value : rec.samples_count;
    row.stall_reason = rec.stall_reason;
    row.sample_kind = sample_kind;
    row.scope_name_id = scope_name_id;
    row.source_file_id = source_file_id;
    row.source_line = rec.source_line;
    return row;
}

MemoryAllocEventBatchRow MakeMemoryAllocBatchRow(const ActivityRecord& rec) {
    MemoryAllocEventBatchRow row;
    row.start_ns = rec.cpu_start_ns;
    row.duration_ns = rec.duration_ns;
    row.memory_op = rec.memory_op;
    row.memory_kind = rec.memory_kind;
    row.address = rec.address;
    row.bytes = rec.bytes;
    row.device_id = rec.device_id;
    row.stream_id = static_cast<uint32_t>(rec.stream);
    row.corr_id = rec.corr_id;
    return row;
}

SynchronizationEventBatchRow MakeSynchronizationBatchRow(const ActivityRecord& rec, const uint32_t function_id) {
    SynchronizationEventBatchRow row;
    row.start_ns = rec.cpu_start_ns;
    row.duration_ns = rec.duration_ns;
    row.sync_type = rec.sync_type;
    row.stream_id = static_cast<uint32_t>(rec.stream);
    row.event_id = rec.sync_event_id;
    row.context_id = rec.context_id;
    row.corr_id = rec.corr_id;
    row.function_id = function_id;
    return row;
}

}  // namespace gpufl::detail
