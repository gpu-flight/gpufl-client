#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "gpufl/core/monitor.hpp"

namespace gpufl {

struct ProfilingRequest {
    ProfilingEngine engine = ProfilingEngine::Monitor;
    bool enable_memory_tracking = false;
    bool enable_external_correlation = true;
    bool enable_synchronization = true;
    bool enable_cuda_graphs_tracking = false;
};

struct DeviceFacts {
    int compute_major = 0;
    int compute_minor = 0;
    uint32_t cupti_version = 0;
};

struct EnvOverrides {
    bool sass_metrics_only = false;
    bool sass_force_safe_activity = false;
    bool sass_allow_full_activity = false;
    bool sass_allow_kernel_activity = false;
    bool sass_allow_marker_activity = false;
    bool sass_allow_mem_transfer_activity = false;
    bool sass_allow_memory2_activity = false;
    bool sass_allow_memory_activity = false;
    bool sass_allow_sync_activity = false;
    bool sass_allow_graph_activity = false;
    bool sass_allow_external_correlation = false;
    bool disable_cubin_capture = false;
    bool sass_disable_cubin_capture = false;

    static bool Enabled(const char* name) {
        const char* v = std::getenv(name);
        return v && v[0] != '\0' && v[0] != '0' &&
               std::strcmp(v, "false") != 0 && std::strcmp(v, "FALSE") != 0 &&
               std::strcmp(v, "off") != 0 && std::strcmp(v, "OFF") != 0;
    }

    static EnvOverrides FromProcess() {
        EnvOverrides env;
        env.sass_metrics_only = Enabled("GPUFL_SASS_METRICS_ONLY");
        env.sass_force_safe_activity = Enabled("GPUFL_SASS_FORCE_SAFE_ACTIVITY");
        env.sass_allow_full_activity = Enabled("GPUFL_SASS_ALLOW_FULL_ACTIVITY");
        env.sass_allow_kernel_activity = Enabled("GPUFL_SASS_ALLOW_KERNEL_ACTIVITY");
        env.sass_allow_marker_activity = Enabled("GPUFL_SASS_ALLOW_MARKER_ACTIVITY");
        env.sass_allow_mem_transfer_activity =
            Enabled("GPUFL_SASS_ALLOW_MEM_TRANSFER_ACTIVITY");
        env.sass_allow_memory2_activity = Enabled("GPUFL_SASS_ALLOW_MEMORY2_ACTIVITY");
        env.sass_allow_memory_activity = Enabled("GPUFL_SASS_ALLOW_MEMORY_ACTIVITY");
        env.sass_allow_sync_activity = Enabled("GPUFL_SASS_ALLOW_SYNC_ACTIVITY");
        env.sass_allow_graph_activity = Enabled("GPUFL_SASS_ALLOW_GRAPH_ACTIVITY");
        env.sass_allow_external_correlation =
            Enabled("GPUFL_SASS_ALLOW_EXTERNAL_CORRELATION");
        env.disable_cubin_capture = Enabled("GPUFL_DISABLE_CUBIN_CAPTURE");
        env.sass_disable_cubin_capture = Enabled("GPUFL_SASS_DISABLE_CUBIN_CAPTURE");
        return env;
    }
};

struct ResolvedProfilingPlan {
    ProfilingEngine requested_engine = ProfilingEngine::Monitor;
    bool is_sass_profiler = false;
    bool sass_metrics_only = false;
    bool safe_sass_activity_defaults = false;
    bool allow_sass_kernel_activity = true;
    bool allow_sass_marker_activity = true;
    bool allow_sass_mem_transfer_activity = true;
    bool allow_sass_memory2_activity = true;
    bool allow_sass_sync_activity = true;
    bool allow_sass_graph_activity = true;
    bool allow_sass_external_correlation = true;
    bool needs_cubin_capture = false;
};

inline ProfilingRequest MakeProfilingRequest(const MonitorOptions& opts) {
    ProfilingRequest request;
    request.engine = opts.profiling_engine;
    request.enable_memory_tracking = opts.enable_memory_tracking;
    request.enable_external_correlation = opts.enable_external_correlation;
    request.enable_synchronization = opts.enable_synchronization;
    request.enable_cuda_graphs_tracking = opts.enable_cuda_graphs_tracking;
    return request;
}

class NvidiaProfilingPolicy {
   public:
    static ResolvedProfilingPlan Resolve(const ProfilingRequest& request,
                                         const DeviceFacts& /*device*/,
                                         const EnvOverrides& env) {
        ResolvedProfilingPlan plan;
        plan.requested_engine = request.engine;
        plan.is_sass_profiler =
            request.engine == ProfilingEngine::SassMetrics ||
            request.engine == ProfilingEngine::Deep;
        plan.sass_metrics_only = plan.is_sass_profiler && env.sass_metrics_only;

        if (plan.is_sass_profiler) {
            if (plan.sass_metrics_only) {
                plan.safe_sass_activity_defaults = true;
            } else if (env.sass_force_safe_activity) {
                plan.safe_sass_activity_defaults = true;
            } else if (env.sass_allow_full_activity) {
                plan.safe_sass_activity_defaults = false;
            } else {
                plan.safe_sass_activity_defaults = true;
            }
        }

        plan.allow_sass_kernel_activity =
            !plan.sass_metrics_only &&
            (!plan.safe_sass_activity_defaults || env.sass_allow_kernel_activity);
        plan.allow_sass_marker_activity =
            !plan.sass_metrics_only &&
            (!plan.safe_sass_activity_defaults || env.sass_allow_marker_activity);
        plan.allow_sass_mem_transfer_activity =
            !plan.sass_metrics_only &&
            (!plan.safe_sass_activity_defaults || env.sass_allow_mem_transfer_activity);

        const bool memory2Requested =
            env.sass_allow_memory2_activity || env.sass_allow_memory_activity;
        plan.allow_sass_memory2_activity =
            !plan.sass_metrics_only &&
            (!plan.safe_sass_activity_defaults || memory2Requested ||
             !env.sass_allow_mem_transfer_activity);

        plan.allow_sass_sync_activity =
            !plan.sass_metrics_only &&
            (!plan.safe_sass_activity_defaults || env.sass_allow_sync_activity);
        plan.allow_sass_graph_activity =
            !plan.sass_metrics_only &&
            (!plan.safe_sass_activity_defaults || env.sass_allow_graph_activity);
        plan.allow_sass_external_correlation =
            !plan.sass_metrics_only &&
            (!plan.safe_sass_activity_defaults ||
             env.sass_allow_external_correlation);

        plan.needs_cubin_capture = NeedsCubinCapture_(request.engine, env, plan);
        return plan;
    }

   private:
    static bool NeedsCubinCapture_(ProfilingEngine engine, const EnvOverrides& env,
                                   const ResolvedProfilingPlan& plan) {
        if (env.disable_cubin_capture) return false;
        if (plan.is_sass_profiler && env.sass_disable_cubin_capture) return false;
        return engine == ProfilingEngine::PcSampling ||
               engine == ProfilingEngine::SassMetrics ||
               engine == ProfilingEngine::Deep;
    }
};

}  // namespace gpufl
