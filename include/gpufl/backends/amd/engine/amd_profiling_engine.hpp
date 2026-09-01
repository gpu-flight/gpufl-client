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
                            uint32_t gpu_device_id,
                            const MonitorOptions& opts) = 0;

    /// Begin profiling (context is already started).
    virtual void start() = 0;

    /// Stop profiling (before context stop).
    virtual void stop() = 0;

    /// Periodically drain buffered profiling data into the monitor ring buffer.
    virtual void drain() = 0;

    /// Cheap collector-loop service tick. Pull-based engines use this to
    /// honor their sampling cadence; callback/buffer engines leave it idle.
    virtual void service() {}

    /// Release resources.
    virtual void shutdown() = 0;

    /// True once this engine has emitted at least one profiling sample.
    virtual bool hasData() const = 0;

    /// True once the context-bound service and its counter configuration are
    /// ready for a deep window to arm.
    virtual bool isPrepared() const = 0;

    /// Point-in-time state used by deep-window audit rows before disarming.
    virtual bool isArmed() const = 0;

    /// Scope hooks - engines may filter collection to scoped regions.
    virtual void onScopeStart(const char* /*name*/) {}
    virtual void onScopeStop(const char* /*name*/) {}
};

}  // namespace gpufl::amd
