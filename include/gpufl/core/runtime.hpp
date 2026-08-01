#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/backend_interfaces.hpp"
#include "gpufl/core/sampler.hpp"
#include "gpufl/core/segment_context.hpp"

namespace gpufl {
class Logger;
class SegmentRuntime;

struct Runtime {
    std::string app_name;
    std::string session_id;
    // Present only for launcher-owned long-running segmentation. The
    // coordinator is introduced in a later slice; segment zero still carries
    // these values in job_start so the wire contract is testable end to end.
    std::string run_id;
    uint32_t segment_index = 0;
    int64_t segment_every_ms = 0;
    uint64_t segment_max_rows = 0;
    int64_t run_roll_every_ms = 0;
    uint64_t run_roll_max_bytes = 0;

    std::shared_ptr<Logger> logger;
    // Transition bridge: producers move from the aliases above to this
    // immutable context one complete write path at a time. C++17 provides
    // atomic_load/store overloads for shared_ptr; do not access this member
    // directly outside Runtime's methods.
    std::shared_ptr<const SegmentContext> active_segment_context;
    std::shared_ptr<SegmentRuntime> segment_runtime;

    SegmentWriteLease acquireSegmentContext(
        const char* owner = "general") const noexcept;
    /** Liveness check only; does not participate in writer drainage. */
    bool hasSegmentContext() const noexcept;
    /** Coordinator-only read. This does not protect a write operation. */
    std::shared_ptr<const SegmentContext> peekSegmentContext() const noexcept;
    bool publishSegmentContext(
        std::shared_ptr<const SegmentContext> context) noexcept;
    /** Stop new writers without publishing a replacement. Shutdown only. */
    std::shared_ptr<const SegmentContext>
    sealActiveSegmentContext() noexcept;

    std::shared_ptr<IUnifiedGpuCollector> unified_gpu_collector;
    std::shared_ptr<ISystemCollector<DeviceSample>> collector;
    std::unique_ptr<HostCollector> host_collector;
    std::shared_ptr<IGpuStaticInfoCollector> static_info_collector;

    // background system sampling
    std::atomic<bool> system_sampling{false};
    Sampler sampler;
    std::mutex system_mu;
    std::thread system_thread;
    int system_interval_ms{0};
};

Runtime* runtime();
void set_runtime(std::unique_ptr<Runtime> rt);
}  // namespace gpufl
