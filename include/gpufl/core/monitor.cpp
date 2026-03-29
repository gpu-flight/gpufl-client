#include "gpufl/core/monitor.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stack>
#include <thread>
#include <vector>

#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/batch_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/dictionary_manager.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/batch_models.hpp"
#include "gpufl/core/model/memcpy_event_model.hpp"
#include "gpufl/core/model/profile_sample_model.hpp"
#include "gpufl/core/model/scope_event_model.hpp"
#include "gpufl/core/monitor_adapter.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/stack_registry.hpp"

namespace gpufl {

RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

static std::unique_ptr<IMonitorAdapter> g_adapter;
static std::atomic<bool> g_initialized{false};
static std::thread g_collectorThread;
static std::atomic<bool> g_collectorRunning{false};
static thread_local std::stack<void*> g_rangeStack;

// Batch state — kernel/memcpy accessed only from CollectorLoop thread;
// scope/profile may be pushed from any thread (guarded by their own mutex)
static DictionaryManager      g_dictManager;
static BatchBuffer<KernelBatchRow>  g_kernelBatch;
static BatchBuffer<MemcpyBatchRow>  g_memcpyBatch;
static uint64_t g_kernelBatchId = 0;
static uint64_t g_memcpyBatchId = 0;
// Deferred kernel detail rows — written after the kernel batch so the
// backend's UPDATE (match by corr_id) always finds the INSERT first.
static std::vector<KernelDetailRow> g_pendingDetails;

// Tracks the most recently begun scope name ID.
// Updated by PushScopeRow (user thread) when a scope begin row is pushed.
// PC_SAMPLE records arrive in the ring buffer after EndPerfScope() is called
// (which happens in ScopedMonitor dtor), but KERNEL activity records arrive
// later via flushActivities().  By the time PC samples are processed the
// active scope ID is already written here, so we can assign it directly.
static std::atomic<uint32_t> g_activeScopeNameId{0};

static BatchBuffer<ScopeBatchRow>         g_scopeBatch;
static BatchBuffer<ProfileSampleBatchRow> g_profileBatch;
static uint64_t g_scopeBatchId   = 0;
static uint64_t g_profileBatchId = 0;
static std::mutex g_scopeBatchMu;
static std::atomic<uint64_t> g_nextScopeInstanceId{1};

static void flushBatches(Logger& logger, const std::string& session_id) {
    g_dictManager.flushDictionary(logger, session_id);
    g_dictManager.flushSourceContent(logger, session_id);
    g_dictManager.flushDisassembly(logger, session_id);
    if (!g_kernelBatch.empty()) {
        logger.write(model::KernelEventBatchModel(
            g_kernelBatch, session_id, ++g_kernelBatchId));
        g_kernelBatch.clear();
        // Write deferred details AFTER their batch so the backend UPDATE
        // always finds the INSERT row that was just written.
        for (const auto& d : g_pendingDetails) {
            logger.write(model::KernelDetailModel(d));
        }
        g_pendingDetails.clear();
    }
    if (!g_memcpyBatch.empty()) {
        logger.write(model::MemcpyEventBatchModel(
            g_memcpyBatch, session_id, ++g_memcpyBatchId));
        g_memcpyBatch.clear();
    }
    {
        std::lock_guard lk(g_scopeBatchMu);
        if (!g_scopeBatch.empty()) {
            logger.write(model::ScopeEventBatchModel(
                g_scopeBatch, session_id, ++g_scopeBatchId));
            g_scopeBatch.clear();
        }
        if (!g_profileBatch.empty()) {
            logger.write(model::ProfileSampleBatchModel(
                g_profileBatch, session_id, ++g_profileBatchId));
            g_profileBatch.clear();
        }
    }
}

void CollectorLoop() {
    auto processNext = []() -> bool {
        ActivityRecord rec{};
        if (!g_monitorBuffer.Consume(rec)) return false;

        const int64_t duration_ns = rec.duration_ns;
        Runtime* rt = runtime();
        if (!(rt && rt->logger)) return true;

        if (rec.type == TraceType::KERNEL || rec.type == TraceType::MEMCPY ||
            rec.type == TraceType::MEMSET) {
            const std::string stack_trace =
                (rec.stack_id != 0) ? StackRegistry::instance().get(rec.stack_id)
                                    : "";
            const char* platform =
                g_adapter ? g_adapter->platformName() : "unknown";

            if (rec.type == TraceType::KERNEL) {
                const uint32_t kernel_id = g_dictManager.internKernel(rec.name);

                KernelBatchRow row;
                row.start_ns    = rec.cpu_start_ns;
                row.kernel_id   = kernel_id;
                row.stream_id   = static_cast<uint32_t>(rec.stream);
                row.duration_ns = duration_ns;
                row.corr_id     = rec.corr_id;
                row.dyn_shared  = rec.dyn_shared;
                row.num_regs    = rec.num_regs;
                row.has_details = rec.has_details ? 1 : 0;
                g_kernelBatch.push(row);

                if (rec.has_details) {
                    KernelDetailRow detail;
                    detail.corr_id    = rec.corr_id;
                    detail.session_id = rt->session_id;
                    detail.pid        = detail::GetPid();
                    detail.app        = rt->app_name;
                    detail.grid_x = rec.grid_x; detail.grid_y = rec.grid_y;
                    detail.grid_z = rec.grid_z;
                    detail.block_x = rec.block_x; detail.block_y = rec.block_y;
                    detail.block_z = rec.block_z;
                    detail.static_shared         = rec.static_shared;
                    detail.local_bytes           = rec.local_bytes;
                    detail.const_bytes           = rec.const_bytes;
                    detail.occupancy             = rec.occupancy;
                    detail.reg_occupancy         = rec.reg_occupancy;
                    detail.smem_occupancy        = rec.smem_occupancy;
                    detail.warp_occupancy        = rec.warp_occupancy;
                    detail.block_occupancy       = rec.block_occupancy;
                    std::memcpy(detail.limiting_resource, rec.limiting_resource,
                                sizeof(detail.limiting_resource));
                    detail.max_active_blocks      = rec.max_active_blocks;
                    detail.local_mem_total        = rec.local_mem_total;
                    detail.local_mem_per_thread   = rec.local_mem_per_thread;
                    detail.cache_config_requested = rec.cache_config_requested;
                    detail.cache_config_executed  = rec.cache_config_executed;
                    detail.shared_mem_executed    = rec.shared_mem_executed;
                    detail.user_scope  = rec.user_scope;
                    detail.stack_trace = stack_trace;
                    // Defer: written after the kernel batch by flushBatches().
                    g_pendingDetails.push_back(std::move(detail));
                }

                if (g_kernelBatch.needsFlush()) {
                    flushBatches(*rt->logger, rt->session_id);
                }

            } else if (rec.type == TraceType::MEMCPY) {
                MemcpyBatchRow row;
                row.start_ns    = rec.cpu_start_ns;
                row.stream_id   = static_cast<uint32_t>(rec.stream);
                row.duration_ns = duration_ns;
                row.bytes       = rec.bytes;
                row.copy_kind   = rec.copy_kind;
                row.corr_id     = rec.corr_id;
                g_memcpyBatch.push(row);

                if (g_memcpyBatch.needsFlush()) {
                    flushBatches(*rt->logger, rt->session_id);
                }

            } else {
                // MEMSET — infrequent, keep as immediate verbose write
                MemsetEvent be;
                be.platform    = platform;
                be.device_id   = rec.device_id;
                be.stream_id   = static_cast<uint32_t>(rec.stream);
                be.session_id  = rt->session_id;
                be.pid         = detail::GetPid();
                be.app         = rt->app_name;
                be.name        = rec.name;
                be.start_ns    = rec.cpu_start_ns;
                be.end_ns      = rec.cpu_start_ns + duration_ns;
                be.api_start_ns = rec.api_start_ns;
                be.api_exit_ns  = rec.api_exit_ns;
                be.user_scope  = rec.user_scope;
                be.scope_depth = rec.scope_depth;
                be.corr_id     = rec.corr_id;
                be.stack_trace = stack_trace;
                be.bytes       = rec.bytes;
                rt->logger->write(model::MemsetEventModel(be));
            }
        } else if (rec.type == TraceType::RANGE) {
            const uint32_t name_id =
                g_dictManager.internScopeName(rec.name);
            const uint64_t instance_id =
                g_nextScopeInstanceId.fetch_add(1, std::memory_order_relaxed);

            ScopeBatchRow begin_row;
            begin_row.ts_ns             = rec.cpu_start_ns;
            begin_row.scope_instance_id = instance_id;
            begin_row.name_id           = name_id;
            begin_row.event_type        = 0;  // begin
            begin_row.depth             = rec.scope_depth;

            ScopeBatchRow end_row;
            end_row.ts_ns             = rec.cpu_start_ns + duration_ns;
            end_row.scope_instance_id = instance_id;
            end_row.name_id           = name_id;
            end_row.event_type        = 1;  // end
            end_row.depth             = rec.scope_depth;

            {
                std::lock_guard lk(g_scopeBatchMu);
                g_scopeBatch.push(begin_row);
                g_scopeBatch.push(end_row);
            }
        } else if (rec.type == TraceType::PC_SAMPLE) {
            // Determine sample kind: 0=pc_sampling, 1=sass_metric
            uint8_t kind = 0;
            if (rec.metric_name[0] != '\0') {
                kind = 1;
            } else if (rec.sample_kind[0] != '\0' &&
                       rec.sample_kind[0] == 's') {
                kind = 1;  // "sass_metric"
            }

            const std::string func_key =
                std::string(rec.function_name) + "@" + rec.source_file;
            const uint32_t function_id =
                g_dictManager.internFunction(func_key);
            // For PC sampling rows metric_name is empty; use reason_name so the
            // stall reason string is interned into metric_dict and reachable via
            // metric_id on the backend.  For SASS rows metric_name is always set.
            const std::string metric_key = (rec.metric_name[0] != '\0')
                ? std::string(rec.metric_name)
                : rec.reason_name;
            const uint32_t metric_id =
                g_dictManager.internMetric(metric_key);
            // Use the most recently begun scope.  PC samples are pushed to the
            // ring buffer by EndPerfScope() (inside ScopedMonitor dtor), which
            // runs after the scope begin row is pushed via PushScopeRow().
            // g_activeScopeNameId is therefore already set to the correct ID
            // by the time the collector loop processes these records.
            const uint32_t scope_name_id =
                g_activeScopeNameId.load(std::memory_order_relaxed);

            const uint32_t source_file_id =
                g_dictManager.internSourceFile(rec.source_file);

            ProfileSampleBatchRow row;
            row.ts_ns           = rec.cpu_start_ns;
            row.corr_id         = rec.corr_id;
            row.device_id       = rec.device_id;
            row.function_id     = function_id;
            row.pc_offset       = rec.pc_offset;
            row.metric_id       = metric_id;
            row.metric_value    = (kind == 1) ? rec.metric_value
                                              : rec.samples_count;
            row.stall_reason    = rec.stall_reason;
            row.sample_kind     = kind;
            row.scope_name_id   = scope_name_id;
            row.source_file_id  = source_file_id;
            row.source_line     = rec.source_line;

            bool needs_flush = false;
            {
                std::lock_guard lk(g_scopeBatchMu);
                g_profileBatch.push(row);
                needs_flush = g_profileBatch.needsFlush();
            }
            if (needs_flush) {
                flushBatches(*rt->logger, rt->session_id);
            }
        }

        return true;
    };

    while (g_collectorRunning.load()) {
        if (!processNext()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        static auto lastFlush = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - lastFlush >
            std::chrono::milliseconds(250)) {
            if (g_initialized.load() && g_adapter) g_adapter->flushActivities();
            if (Runtime* rt = runtime(); rt && rt->logger) {
                flushBatches(*rt->logger, rt->session_id);
            }
            lastFlush = std::chrono::steady_clock::now();
        }
    }

    // Drain remaining ring buffer entries
    while (processNext()) {
    }

    // Final flush of any partially-filled batches
    if (Runtime* rt = runtime(); rt && rt->logger) {
        flushBatches(*rt->logger, rt->session_id);
    }
}

void Monitor::Initialize(const MonitorOptions& opts) {
    if (g_initialized.exchange(true)) return;

    // Reset batch state for this session
    g_dictManager.reset();
    g_kernelBatch.clear();
    g_memcpyBatch.clear();
    g_kernelBatchId  = 0;
    g_memcpyBatchId  = 0;
    g_scopeBatch.clear();
    g_profileBatch.clear();
    g_scopeBatchId   = 0;
    g_profileBatchId = 0;
    g_nextScopeInstanceId.store(1);
    g_activeScopeNameId.store(0);

    DebugLogger::setEnabled(opts.enable_debug_output);
    g_adapter = CreateMonitorAdapter();
    if (g_adapter) g_adapter->initialize(opts);

    g_collectorRunning.store(true);
    g_collectorThread = std::thread(CollectorLoop);
}

void Monitor::Shutdown() {
    if (!g_initialized.exchange(false)) return;

    if (g_adapter) g_adapter->shutdown();
    g_collectorRunning.store(false);
    if (g_collectorThread.joinable()) g_collectorThread.join();
    g_adapter.reset();
}

void Monitor::Start() {
    if (g_adapter) g_adapter->start();
}

void Monitor::Stop() {
    if (g_adapter) g_adapter->stop();
}

void Monitor::PushRange(const char* name) {
    void* handle = nullptr;
    RecordStart(name, 0, TraceType::RANGE, &handle);
    g_rangeStack.push(handle);
}

void Monitor::PopRange() {
    if (g_rangeStack.empty()) return;
    void* handle = g_rangeStack.top();
    g_rangeStack.pop();
    RecordStop(handle, 0);
}

void Monitor::RecordStart(const char* name, const StreamHandle stream,
                          const TraceType type, void** outHandle) {
    auto* rec = new ActivityRecord();
    strncpy(rec->name, name, 127);
    rec->type       = type;
    rec->stream     = stream;
    rec->cpu_start_ns = detail::GetTimestampNs();
    rec->duration_ns  = 0;
    rec->has_details  = false;
    *outHandle = rec;
}

void Monitor::RecordStop(void* handle, StreamHandle) {
    auto* rec = static_cast<ActivityRecord*>(handle);
    rec->duration_ns = detail::GetTimestampNs() - rec->cpu_start_ns;
    g_monitorBuffer.Push(*rec);
    delete rec;
}

void Monitor::BeginProfilerScope(const char* name) {
    if (auto* b = GetBackend()) b->OnScopeStart(name);
}

void Monitor::EndProfilerScope(const char* name) {
    if (auto* b = GetBackend()) b->OnScopeStop(name);
}

void Monitor::BeginPerfScope(const char* name) {
    if (auto* b = GetBackend()) b->OnPerfScopeStart(name);
}

void Monitor::EndPerfScope(const char* name) {
    if (auto* b = GetBackend()) b->OnPerfScopeStop(name);
}

IMonitorBackend* Monitor::GetBackend() {
    return g_adapter ? g_adapter->backend() : nullptr;
}

uint32_t Monitor::InternScopeName(const std::string& name) {
    return g_dictManager.internScopeName(name);
}

void Monitor::EnqueueCubinForDisassembly(uint64_t crc, const uint8_t* data,
                                         size_t size) {
    g_dictManager.enqueueDisassembly(crc, data, size);
}

void Monitor::PushScopeRow(const ScopeBatchRow& row) {
    if (row.event_type == 0) {  // begin: update active scope for PC sample association
        g_activeScopeNameId.store(row.name_id, std::memory_order_relaxed);
    }
    std::lock_guard lk(g_scopeBatchMu);
    g_scopeBatch.push(row);
}

}  // namespace gpufl
