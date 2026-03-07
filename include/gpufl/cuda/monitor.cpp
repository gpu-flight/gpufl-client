#include "gpufl/core/monitor.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <stack>
#include <thread>
#include <vector>

#include "gpufl/backends/nvidia/cupti_backend.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/kernel_event_model.hpp"
#include "gpufl/core/model/memcpy_event_model.hpp"
#include "gpufl/core/model/profile_sample_model.hpp"
#include "gpufl/core/model/scope_event_model.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/stack_registry.hpp"

namespace gpufl {

// Global Ring Buffer for MPSC trace delivery
RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

// Backend implementations are in separate files
static std::unique_ptr<IMonitorBackend> g_backend;
static std::atomic<bool> g_initialized{false};
static std::thread g_collectorThread;
static std::atomic<bool> g_collectorRunning{false};
static thread_local std::stack<void*> g_rangeStack;

static std::string MemcpyKindToString(uint32_t kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            return "HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            return "DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
            return "HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
            return "AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
            return "AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
            return "AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
            return "DtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            return "DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            return "HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
            return "PtoP";
        default:
            return "Unknown";
    }
}

static std::string MemoryKindToString(uint32_t kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
            return "Unknown";
        case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
            return "Pageable";
        case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
            return "Pinned";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
            return "Device";
        case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
            return "Array";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
            return "Managed";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
            return "DeviceStatic";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
            return "ManagedStatic";
        default:
            return "Unknown";
    }
}

void CollectorLoop() {
    auto processNext = []() -> bool {
        ActivityRecord rec{};
        if (g_monitorBuffer.Consume(rec)) {
            int64_t duration_ns = rec.duration_ns;

            if (rec.start_event != nullptr && rec.stop_event != nullptr) {
                // Wait for GPU events with a timeout to avoid infinite loop if
                // something goes wrong
                auto start_wait = std::chrono::steady_clock::now();
                while (cudaEventQuery(rec.stop_event) == cudaErrorNotReady) {
                    if (std::chrono::steady_clock::now() - start_wait >
                        std::chrono::seconds(5)) {
                        break;
                    }
                    std::this_thread::yield();
                }

                float durationMs = 0.0f;
                cudaEventElapsedTime(&durationMs, rec.start_event,
                                     rec.stop_event);
                duration_ns = static_cast<int64_t>(durationMs * 1e6);
            }

            Runtime* rt = runtime();
            if (rt && rt->logger) {
                if (rec.type == TraceType::KERNEL ||
                    rec.type == TraceType::MEMCPY ||
                    rec.type == TraceType::MEMSET) {
                    std::string stack_trace =
                        (rec.stack_id != 0)
                            ? StackRegistry::instance().get(rec.stack_id)
                            : "";

                    if (rec.type == TraceType::KERNEL) {
                        KernelEvent be;
                        be.platform = "cuda";
                        be.has_details = rec.has_details;
                        be.device_id = rec.device_id;
                        be.stream_id = static_cast<uint32_t>(
                            reinterpret_cast<uintptr_t>(rec.stream));
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

                        // Phase 1a: always-on CUPTI fields (from
                        // CUpti_ActivityKernel11, no details required)
                        be.local_mem_total = rec.local_mem_total;
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
                        be.platform = "cuda";
                        be.device_id = rec.device_id;
                        be.stream_id = static_cast<uint32_t>(
                            reinterpret_cast<uintptr_t>(rec.stream));
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
                        be.copy_kind = MemcpyKindToString(rec.copy_kind);
                        be.src_kind = MemoryKindToString(rec.src_kind);
                        be.dst_kind = MemoryKindToString(rec.dst_kind);
                        rt->logger->write(model::MemcpyEventModel(be));
                    } else if (rec.type == TraceType::MEMSET) {
                        MemsetEvent be;
                        be.platform = "cuda";
                        be.device_id = rec.device_id;
                        be.stream_id = static_cast<uint32_t>(
                            reinterpret_cast<uintptr_t>(rec.stream));
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
                    if (rt->host_collector)
                        be.host = rt->host_collector->sample();
                    rt->logger->write(model::ScopeBeginModel(be));

                    ScopeEndEvent ee;
                    ee.pid = detail::GetPid();
                    ee.app = rt->app_name;
                    ee.name = rec.name;
                    ee.session_id = rt->session_id;
                    ee.ts_ns = rec.cpu_start_ns + duration_ns;
                    if (rt->collector) ee.devices = rt->collector->sampleAll();
                    if (rt->host_collector)
                        ee.host = rt->host_collector->sample();
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
                    pe.corr_id = static_cast<int32_t>(rec.corr_id);
                    pe.source_file = rec.source_file;
                    pe.reason_name = rec.reason_name;
                    pe.function_name = rec.function_name;
                    pe.source_line = rec.source_line;
                    pe.metric_name = rec.metric_name;
                    pe.metric_value = rec.metric_value;
                    pe.pc_offset = rec.pc_offset;
                    rt->logger->write(model::ProfileSampleModel(pe));
                }
            }

            if (rec.start_event) cudaEventDestroy(rec.start_event);
            if (rec.stop_event) cudaEventDestroy(rec.stop_event);
            return true;
        }
        return false;
    };

    while (g_collectorRunning.load()) {
        if (!processNext()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Periodic activity flush
        static auto lastFlush = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - lastFlush >
            std::chrono::milliseconds(100)) {
            if (g_initialized.load() && g_backend) {
                cuptiActivityFlushAll(0);
            }
            lastFlush = std::chrono::steady_clock::now();
        }
    }

    // Process remaining events after shutdown signal
    while (processNext()) {
    }
}

void Monitor::Initialize(const MonitorOptions& opts) {
    if (g_initialized.exchange(true)) return;

    DebugLogger::setEnabled(opts.enable_debug_output);

#if defined(GPUFL_HAS_CUDA)
    g_backend = std::make_unique<CuptiBackend>();
#else
    // Future AMD backend here
#endif

    if (g_backend) {
        g_backend->initialize(opts);
    }

    g_collectorRunning.store(true);
    g_collectorThread = std::thread(CollectorLoop);
}

void Monitor::Shutdown() {
    if (!g_initialized.exchange(false)) return;

    if (g_backend) {
        g_backend->shutdown();
    }

    g_collectorRunning.store(false);

    if (g_collectorThread.joinable()) {
        g_collectorThread.join();
    }

    g_backend.reset();
}

void Monitor::Start() {
    if (g_backend) g_backend->start();
}

void Monitor::Stop() {
    if (g_backend) g_backend->stop();
}

void Monitor::PushRange(const char* name) {
    void* handle = nullptr;
    RecordStart(name, nullptr, TraceType::RANGE, &handle);
    g_rangeStack.push(handle);
}

void Monitor::PopRange() {
    if (g_rangeStack.empty()) return;
    void* handle = g_rangeStack.top();
    g_rangeStack.pop();
    RecordStop(handle, nullptr);
}

void Monitor::RecordStart(const char* name, const cudaStream_t stream,
                          const TraceType type, void** outHandle) {
    auto* rec = new ActivityRecord();
    strncpy(rec->name, name, 127);
    rec->type = type;
    rec->stream = stream;
    rec->cpu_start_ns = detail::GetTimestampNs();
    rec->duration_ns = 0;
    rec->has_details = false;

    cudaEventCreate(&rec->start_event);
    cudaEventCreate(&rec->stop_event);
    cudaEventRecord(rec->start_event, stream);

    *outHandle = rec;
}

void Monitor::RecordStop(void* handle, cudaStream_t stream) {
    auto* rec = static_cast<ActivityRecord*>(handle);
    cudaEventRecord(rec->stop_event, stream);

    bool pushed = g_monitorBuffer.Push(*rec);

    if (!pushed) {
        // Buffer full, cleanup immediately
        if (rec->start_event) cudaEventDestroy(rec->start_event);
        if (rec->stop_event) cudaEventDestroy(rec->stop_event);
    }
    delete rec;
}

void Monitor::BeginProfilerScope(const char* name) {
    if (g_backend) {
        g_backend->OnScopeStart(name);
    }
}

void Monitor::EndProfilerScope(const char* name) {
    if (g_backend) {
        g_backend->OnScopeStop(name);
    }
}

void Monitor::BeginPerfScope(const char* name) {
    if (auto* b = g_backend.get()) b->OnPerfScopeStart(name);
}

void Monitor::EndPerfScope(const char* name) {
    if (auto* b = g_backend.get()) b->OnPerfScopeStop(name);
}

IMonitorBackend* Monitor::GetBackend() {
    return g_backend.get();
}
}  // namespace gpufl
