#pragma once

#include <cstdint>
#include <string>

#include "gpufl/core/events.hpp"

namespace gpufl {

struct ActivityRecord;
struct Runtime;

namespace detail {

KernelBatchRow MakeKernelBatchRow(const ActivityRecord& rec, uint32_t kernel_id);
KernelDetailRow MakeKernelDetailRow(const ActivityRecord& rec,
                                     const std::string& stack_trace,
                                     const Runtime& rt);
MemcpyBatchRow MakeMemcpyBatchRow(const ActivityRecord& rec);
ScopeBatchRow MakeScopeBatchRow(int64_t ts_ns,
                                uint64_t instance_id,
                                uint32_t name_id,
                                uint8_t event_type,
                                int depth);
ProfileSampleBatchRow MakeProfileSampleBatchRow(const ActivityRecord& rec,
                                                uint8_t sample_kind,
                                                uint32_t function_id,
                                                uint32_t metric_id,
                                                uint32_t scope_name_id,
                                                uint32_t source_file_id);
MemoryAllocEventBatchRow MakeMemoryAllocBatchRow(const ActivityRecord& rec);
SynchronizationEventBatchRow MakeSynchronizationBatchRow(const ActivityRecord& rec,
                                                         uint32_t function_id);

}  // namespace detail
}  // namespace gpufl
