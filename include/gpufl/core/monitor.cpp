#include "gpufl/core/monitor.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stack>
#include <thread>

#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/kernel_event_model.hpp"
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
                KernelEvent be;
                be.platform = platform;
                be.has_details = rec.has_details;
                be.device_id = rec.device_id;
                be.stream_id = static_cast<uint32_t>(rec.stream);
                be.session_id = rt->session_id;
                be.pid = detail::GetPid();
                be.app = rt->app_name;
                be.name = rec.name;
                be.start_ns = rec.cpu_start_ns;
                be.end_ns = rec.cpu_start_ns + duration_ns;
                be.api_start_ns = rec.api_start_ns;
                be.api_exit_ns = rec.api_exit_ns;
                be.user_scope = rec.user_scope;
                be.scope_depth = rec.scope_depth;
                be.corr_id = rec.corr_id;
                be.stack_trace = stack_trace;
                be.local_mem_total = rec.local_mem_total;
                be.local_mem_per_thread = rec.local_mem_per_thread;
                be.cache_config_requested = rec.cache_config_requested;
                be.cache_config_executed = rec.cache_config_executed;
                be.shared_mem_executed = rec.shared_mem_executed;

                if (rec.has_details) {
                    be.grid = "(" + std::to_string(rec.grid_x) + "," +
                              std::to_string(rec.grid_y) + "," +
                              std::to_string(rec.grid_z) + ")";
                    be.block = "(" + std::to_string(rec.block_x) + "," +
                               std::to_string(rec.block_y) + "," +
                               std::to_string(rec.block_z) + ")";
                    be.dyn_shared_bytes = rec.dyn_shared;
                    be.static_shared_bytes = rec.static_shared;
                    be.num_regs = rec.num_regs;
                    be.local_bytes = rec.local_bytes;
                    be.const_bytes = rec.const_bytes;
                    be.occupancy = rec.occupancy;
                    be.reg_occupancy = rec.reg_occupancy;
                    be.smem_occupancy = rec.smem_occupancy;
                    be.warp_occupancy = rec.warp_occupancy;
                    be.block_occupancy = rec.block_occupancy;
                    be.limiting_resource = rec.limiting_resource;
                    be.max_active_blocks = rec.max_active_blocks;
                }
                rt->logger->write(model::KernelEventModel(be));
            } else if (rec.type == TraceType::MEMCPY) {
                MemcpyEvent be;
                be.platform = platform;
                be.device_id = rec.device_id;
                be.stream_id = static_cast<uint32_t>(rec.stream);
                be.session_id = rt->session_id;
                be.pid = detail::GetPid();
                be.app = rt->app_name;
                be.name = rec.name;
                be.start_ns = rec.cpu_start_ns;
                be.end_ns = rec.cpu_start_ns + duration_ns;
                be.api_start_ns = rec.api_start_ns;
                be.api_exit_ns = rec.api_exit_ns;
                be.user_scope = rec.user_scope;
                be.scope_depth = rec.scope_depth;
                be.corr_id = rec.corr_id;
                be.stack_trace = stack_trace;
                be.bytes = rec.bytes;
                be.copy_kind = g_adapter ? g_adapter->memcpyKindToString(rec.copy_kind)
                                         : "Unknown";
                be.src_kind = g_adapter ? g_adapter->memoryKindToString(rec.src_kind)
                                        : "Unknown";
                be.dst_kind = g_adapter ? g_adapter->memoryKindToString(rec.dst_kind)
                                        : "Unknown";
                rt->logger->write(model::MemcpyEventModel(be));
            } else {
                MemsetEvent be;
                be.platform = platform;
                be.device_id = rec.device_id;
                be.stream_id = static_cast<uint32_t>(rec.stream);
                be.session_id = rt->session_id;
                be.pid = detail::GetPid();
                be.app = rt->app_name;
                be.name = rec.name;
                be.start_ns = rec.cpu_start_ns;
                be.end_ns = rec.cpu_start_ns + duration_ns;
                be.api_start_ns = rec.api_start_ns;
                be.api_exit_ns = rec.api_exit_ns;
                be.user_scope = rec.user_scope;
                be.scope_depth = rec.scope_depth;
                be.corr_id = rec.corr_id;
                be.stack_trace = stack_trace;
                be.bytes = rec.bytes;
                rt->logger->write(model::MemsetEventModel(be));
            }
        } else if (rec.type == TraceType::RANGE) {
            ScopeBeginEvent be;
            be.pid = detail::GetPid();
            be.app = rt->app_name;
            be.session_id = rt->session_id;
            be.name = rec.name;
            be.ts_ns = rec.cpu_start_ns;
            if (rt->collector) be.devices = rt->collector->sampleAll();
            if (rt->host_collector) be.host = rt->host_collector->sample();
            rt->logger->write(model::ScopeBeginModel(be));

            ScopeEndEvent ee;
            ee.pid = detail::GetPid();
            ee.app = rt->app_name;
            ee.name = rec.name;
            ee.session_id = rt->session_id;
            ee.ts_ns = rec.cpu_start_ns + duration_ns;
            if (rt->collector) ee.devices = rt->collector->sampleAll();
            if (rt->host_collector) ee.host = rt->host_collector->sample();
            rt->logger->write(model::ScopeEndModel(ee));
        } else if (rec.type == TraceType::PC_SAMPLE) {
            ProfileSampleEvent pe;
            pe.pid = detail::GetPid();
            pe.app = rt->app_name;
            pe.session_id = rt->session_id;
            pe.ts_ns = rec.cpu_start_ns;
            pe.samples_count = rec.samples_count;
            pe.stall_reason = rec.stall_reason;
            pe.device_id = rec.device_id;
            pe.corr_id = rec.corr_id;
            if (rec.sample_kind[0] != '\0') {
                pe.sample_kind = rec.sample_kind;
            } else if (rec.metric_name[0] != '\0') {
                pe.sample_kind = "sass_metric";
            } else if (rec.samples_count > 0) {
                pe.sample_kind = "pc_sampling";
            } else {
                pe.sample_kind = "unknown";
            }
            pe.source_file = rec.source_file;
            pe.reason_name = rec.reason_name;
            pe.function_name = rec.function_name;
            pe.source_line = rec.source_line;
            pe.metric_name = rec.metric_name;
            pe.metric_value = rec.metric_value;
            pe.pc_offset = rec.pc_offset;
            rt->logger->write(model::ProfileSampleModel(pe));
        }

        return true;
    };

    while (g_collectorRunning.load()) {
        if (!processNext()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        static auto lastFlush = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - lastFlush >
            std::chrono::milliseconds(100)) {
            if (g_initialized.load() && g_adapter) g_adapter->flushActivities();
            lastFlush = std::chrono::steady_clock::now();
        }
    }

    while (processNext()) {
    }
}

void Monitor::Initialize(const MonitorOptions& opts) {
    if (g_initialized.exchange(true)) return;

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
    rec->type = type;
    rec->stream = stream;
    rec->cpu_start_ns = detail::GetTimestampNs();
    rec->duration_ns = 0;
    rec->has_details = false;
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

}  // namespace gpufl
