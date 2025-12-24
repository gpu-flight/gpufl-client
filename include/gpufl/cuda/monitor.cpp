#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/cuda/cupti_backend.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/logger.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/runtime.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <stack>
#include <chrono>

namespace gpufl {

    // Global Ring Buffer for MPSC trace delivery
    RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

    // Backend implementations are in separate files
    static std::unique_ptr<IMonitorBackend> g_backend;
    static std::atomic<bool> g_initialized{false};
    static std::thread g_collectorThread;
    static std::atomic<bool> g_collectorRunning{false};
    static thread_local std::stack<void*> g_rangeStack;

    void CollectorLoop() {
        auto processNext = []() -> bool {
            ActivityRecord rec{};
            if (g_monitorBuffer.Consume(rec)) {
                int64_t durationNs = rec.durationNs;

                if (rec.startEvent != nullptr && rec.stopEvent != nullptr) {
                    // Wait for GPU events with a timeout to avoid infinite loop if something goes wrong
                    auto start_wait = std::chrono::steady_clock::now();
                    while (cudaEventQuery(rec.stopEvent) == cudaErrorNotReady) {
                        if (std::chrono::steady_clock::now() - start_wait > std::chrono::seconds(5)) {
                            break;
                        }
                        std::this_thread::yield();
                    }

                    float durationMs = 0.0f;
                    cudaEventElapsedTime(&durationMs, rec.startEvent, rec.stopEvent);
                    durationNs = static_cast<int64_t>(durationMs * 1e6);
                }

                // Log to system
                Runtime* rt = runtime();
                if (rt && rt->logger) {
                    if (rec.type == TraceType::KERNEL) {
                        KernelBeginEvent be;
                        be.pid = detail::getPid();
                        be.app = rt->appName;
                        be.name = rec.name;
                        be.tsNs = rec.cpuStartNs;
                        be.corrId = rec.corrId;
                        if (rec.hasDetails) {
                            be.grid = "(" + std::to_string(rec.gridX) + "," + std::to_string(rec.gridY) + "," + std::to_string(rec.gridZ) + ")";
                            be.block = "(" + std::to_string(rec.blockX) + "," + std::to_string(rec.blockY) + "," + std::to_string(rec.blockZ) + ")";
                            be.dynSharedBytes = rec.dynShared;
                            be.staticSharedBytes = rec.staticShared;
                            be.numRegs = rec.numRegs;
                            be.localBytes = rec.localBytes;
                            be.constBytes = rec.constBytes;
                            be.occupancy = rec.occupancy;
                            be.maxActiveBlocks = rec.maxActiveBlocks;
                        }
                        rt->logger->logKernelBegin(be);

                        KernelEndEvent ee;
                        ee.pid = detail::getPid();
                        ee.app = rt->appName;
                        ee.corrId = rec.corrId;
                        ee.name = rec.name;
                        ee.tsNs = rec.cpuStartNs + durationNs;
                        rt->logger->logKernelEnd(ee);
                    } else if (rec.type == TraceType::RANGE) {
                        ScopeBeginEvent be;
                        be.pid = detail::getPid();
                        be.app = rt->appName;
                        be.name = rec.name;
                        be.tsNs = rec.cpuStartNs;
                        if (rt->collector) be.devices = rt->collector->sampleAll();
                        if (rt->hostCollector) be.host = rt->hostCollector->sample();
                        rt->logger->logScopeBegin(be);

                        ScopeEndEvent ee;
                        ee.pid = detail::getPid();
                        ee.app = rt->appName;
                        ee.name = rec.name;
                        ee.tsNs = rec.cpuStartNs + durationNs;
                        if (rt->collector) ee.devices = rt->collector->sampleAll();
                        if (rt->hostCollector) ee.host = rt->hostCollector->sample();
                        rt->logger->logScopeEnd(ee);
                    }
                }

                if (rec.startEvent) cudaEventDestroy(rec.startEvent);
                if (rec.stopEvent) cudaEventDestroy(rec.stopEvent);
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
            if (std::chrono::steady_clock::now() - lastFlush > std::chrono::milliseconds(100)) {
                cuptiActivityFlushAll(0);
                lastFlush = std::chrono::steady_clock::now();
            }
        }

        // Process remaining events after shutdown signal
        while (processNext()) {}
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
            g_backend->Initialize(opts);
        }

        g_collectorRunning.store(true);
        g_collectorThread = std::thread(CollectorLoop);
    }

    void Monitor::Shutdown() {
        if (!g_initialized.exchange(false)) return;

        if (g_backend) {
            g_backend->Shutdown();
        }

        g_collectorRunning.store(false);

        if (g_collectorThread.joinable()) {
            g_collectorThread.join();
        }

        g_backend.reset();
    }

    void Monitor::Start() {
        if (g_backend) g_backend->Start();
    }

    void Monitor::Stop() {
        if (g_backend) g_backend->Stop();
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

    void Monitor::RecordStart(const char* name, cudaStream_t stream, TraceType type, void** outHandle) {
        auto* rec = new ActivityRecord();
        strncpy(rec->name, name, 127);
        rec->type = type;
        rec->stream = stream;
        rec->cpuStartNs = detail::getTimestampNs();
        rec->durationNs = 0;
        rec->hasDetails = false;

        cudaEventCreate(&rec->startEvent);
        cudaEventCreate(&rec->stopEvent);
        cudaEventRecord(rec->startEvent, stream);

        *outHandle = rec;
    }

    void Monitor::RecordStop(void* handle, cudaStream_t stream) {
        auto* rec = static_cast<ActivityRecord*>(handle);
        cudaEventRecord(rec->stopEvent, stream);

        bool pushed = g_monitorBuffer.Push(*rec);

        if (!pushed) {
            // Buffer full, cleanup immediately
            if (rec->startEvent) cudaEventDestroy(rec->startEvent);
            if (rec->stopEvent) cudaEventDestroy(rec->stopEvent);
        }
        delete rec;
    }

}
