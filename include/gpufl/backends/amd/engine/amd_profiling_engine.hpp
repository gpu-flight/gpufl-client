#pragma once

#include <rocprofiler-sdk/fwd.h>

namespace gpufl {
struct MonitorOptions;
}

namespace gpufl::amd {

/// Abstract base for AMD profiling engines (PC sampling, dispatch counters).
/// Mirrors the lifecycle of IMonitorBackend but scoped to profiling concerns.
/// Only one engine may be active per rocprofiler context.
class AmdProfilingEngine {
   public:
    virtual ~AmdProfilingEngine() = default;

    /// Configure the profiling service on the given context/agent.
    /// Called from toolInitialize() after context and agents are ready.
    /// Returns false if the hardware/driver doesn't support this engine.
    virtual bool initialize(rocprofiler_context_id_t context,
                            rocprofiler_agent_id_t gpu_agent,
                            const MonitorOptions& opts) = 0;

    /// Begin profiling (context is already started).
    virtual void start() = 0;

    /// Stop profiling (before context stop).
    virtual void stop() = 0;

    /// Periodically drain buffered profiling data into the monitor ring buffer.
    virtual void drain() = 0;

    /// Release resources.
    virtual void shutdown() = 0;

    /// Scope hooks — engines may filter collection to scoped regions.
    virtual void onScopeStart(const char* /*name*/) {}
    virtual void onScopeStop(const char* /*name*/) {}
};

}  // namespace gpufl::amd
