#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_backend.hpp"
#include "gpufl/cuda/cupti_backend.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/logger.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/runtime.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <stack>
#include <chrono>

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

                        std::string stackTrace = StackRegistry::instance().get(rec.stackId);

                        KernelEvent be;
                        be.platform = "cuda";
                        be.hasDetails = rec.hasDetails;
                        be.deviceId = rec.deviceId;
                        be.sessionId = rt->sessionId;
                        be.pid = detail::getPid();
                        be.app = rt->appName;
                        be.name = rec.name;
                        be.startNs = rec.cpuStartNs;
                        be.endNs = rec.cpuStartNs + durationNs;
                        be.apiStartNs = rec.apiStartNs;
                        be.apiExitNs = rec.apiExitNs;
                        be.userScope = rec.userScope;
                        be.scopeDepth = rec.scopeDepth;
                        be.corrId = rec.corrId;
                        be.stackTrace = stackTrace;
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
                        rt->logger->logKernelEvent(be);

                    } else if (rec.type == TraceType::RANGE) {
                        ScopeBeginEvent be;
                        be.pid = detail::getPid();
                        be.app = rt->appName;
                        be.sessionId = rt->sessionId;
                        be.name = rec.name;
                        be.tsNs = rec.cpuStartNs;
                        if (rt->collector) be.devices = rt->collector->sampleAll();
                        if (rt->hostCollector) be.host = rt->hostCollector->sample();
                        rt->logger->logScopeBegin(be);

                        ScopeEndEvent ee;
                        ee.pid = detail::getPid();
                        ee.app = rt->appName;
                        ee.name = rec.name;
                        ee.sessionId = rt->sessionId;
                        ee.tsNs = rec.cpuStartNs + durationNs;
                        if (rt->collector) ee.devices = rt->collector->sampleAll();
                        if (rt->hostCollector) ee.host = rt->hostCollector->sample();
                        rt->logger->logScopeEnd(ee);
                    } else if (rec.type == TraceType::PC_SAMPLE) {
                        ProfileSampleEvent pe;
                        pe.pid = detail::getPid();
                        pe.app = rt->appName;
                        pe.sessionId = rt->sessionId;
                        pe.tsNs = rec.cpuStartNs;
                        pe.samplesCount = rec.samplesCount;
                        pe.stallReason = rec.stallReason;
                        pe.deviceId = rec.deviceId;
                        pe.sourceFile = rec.sourceFile;
                        pe.sourceLine = rec.sourceLine;
                        rt->logger->logProfileSample(pe);
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
                if (g_initialized.load() && g_backend) {
                    cuptiActivityFlushAll(0);
                }
                lastFlush = std::chrono::steady_clock::now();
            }
        }

        // Process remaining events after shutdown signal
        while (processNext()) {}
    }

    void Monitor::Initialize(const MonitorOptions& opts) {
        if (g_initialized.exchange(true)) return;

        DebugLogger::setEnabled(opts.enableDebugOutput);

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

    void Monitor::RecordStart(const char* name, const cudaStream_t stream, const TraceType type, void** outHandle) {
        auto* rec = new ActivityRecord();
        std::snprintf(rec->name, sizeof(rec->name), "%s", name ? name : "");
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

    void Monitor::BeginProfilerScope(const char *name) {
        if (g_backend) {
            g_backend->onScopeStart(name);
        }
    }

    void Monitor::EndProfilerScope(const char *name) {
        if (g_backend) {
            g_backend->onScopeStop(name);
        }
    }
}
