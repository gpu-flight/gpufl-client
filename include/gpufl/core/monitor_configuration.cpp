#include "gpufl/core/monitor_configuration.hpp"

#include <cstdlib>
#include <string>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/env_vars.hpp"

namespace gpufl::detail {
namespace {

MonitorBackendKind toMonitorBackendKind(const BackendKind backend) {
    switch (backend) {
        case BackendKind::Nvidia:
            return MonitorBackendKind::Nvidia;
        case BackendKind::Amd:
            return MonitorBackendKind::Amd;
        case BackendKind::None:
            return MonitorBackendKind::None;
        case BackendKind::Auto:
        default:
            return MonitorBackendKind::Auto;
    }
}

bool applyProfilingEngineOverride(MonitorOptions& monitor_options) {
    const char* raw = std::getenv(env::kProfilingEngine);
    if (!raw) return true;

    const std::string value(raw);
    bool matched = true;
    if (value == "Monitor") {
        monitor_options.profiling_engine = ProfilingEngine::Monitor;
    } else if (value == "Trace") {
        monitor_options.profiling_engine = ProfilingEngine::Trace;
    } else if (value == "PcSampling") {
        monitor_options.profiling_engine = ProfilingEngine::PcSampling;
    } else if (value == "SassMetrics") {
        monitor_options.profiling_engine = ProfilingEngine::SassMetrics;
    } else if (value == "PmSampling") {
        monitor_options.profiling_engine = ProfilingEngine::PmSampling;
    } else if (value == "RangeProfiler") {
        monitor_options.profiling_engine = ProfilingEngine::RangeProfiler;
    } else if (value == "RangeProfilerKernelReplay") {
        monitor_options.profiling_engine =
            ProfilingEngine::RangeProfilerKernelReplay;
    } else if (value == "Deep") {
        monitor_options.profiling_engine = ProfilingEngine::Deep;
    } else {
        matched = false;
    }

    if (matched) {
        GFL_LOG_DEBUG("GPUFL_PROFILING_ENGINE override: ", value);
    } else {
        // Preserve the existing user-visible text in this behavior-preserving
        // refactor. The omission of RangeProfilerKernelReplay from this list
        // is a separate copy-only defect, not part of this module move.
        GFL_LOG_ERROR(
            "GPUFL_PROFILING_ENGINE='", value, "' is not a recognized "
            "engine name. Valid values: Monitor, Trace, PcSampling, "
            "SassMetrics, PmSampling, RangeProfiler, Deep. Keeping current engine "
            "selection.");
    }
    return matched;
}

void applyPcSamplingPeriodOverride(MonitorOptions& monitor_options) {
    const char* raw = std::getenv(env::kPcSamplingPeriod);
    if (!raw) return;

    char* end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (end != raw && *end == '\0' && value >= 5 && value <= 31) {
        monitor_options.pc_sampling_period = static_cast<uint32_t>(value);
        GFL_LOG_DEBUG("GPUFL_PC_SAMPLING_PERIOD override: 2^", value, " = ",
                      (1ul << value), " cycles/sample");
        return;
    }
    GFL_LOG_ERROR("GPUFL_PC_SAMPLING_PERIOD='", raw, "' is invalid "
                  "(expected an integer 5..31). Keeping ",
                  monitor_options.pc_sampling_period, ".");
}

void applyDeepArmOverride(MonitorOptions& monitor_options) {
    const char* raw = std::getenv(env::kDeepArm);
    if (!raw) return;

    const std::string value(raw);
    if (value == "window") {
        monitor_options.deep_arm_mode = DeepArmMode::WindowOnly;
    } else if (value == "always") {
        monitor_options.deep_arm_mode = DeepArmMode::Always;
    } else {
        GFL_LOG_ERROR("GPUFL_DEEP_ARM='", value,
                      "' is not recognized. Valid values: always, window. "
                      "Keeping current deep arm mode.");
    }
}

}  // namespace

MonitorOptions buildMonitorOptions(const InitOptions& options) {
    MonitorOptions monitor_options;
    monitor_options.enable_debug_output = options.enable_debug_output;
    monitor_options.profiling_engine = options.profiling_engine;
    applyProfilingEngineOverride(monitor_options);
    applyPcSamplingPeriodOverride(monitor_options);

    monitor_options.kernel_sample_rate_ms = options.kernel_sample_rate_ms;
    monitor_options.enable_stack_trace = options.enable_stack_trace;
    monitor_options.enable_source_collection = options.enable_source_collection;
    monitor_options.source_capture = options.source_capture;
    monitor_options.enable_external_correlation =
        options.enable_external_correlation;
    monitor_options.enable_synchronization = options.enable_synchronization;
    monitor_options.enable_memory_tracking = options.enable_memory_tracking;
    monitor_options.enable_cuda_graphs_tracking =
        options.enable_cuda_graphs_tracking;
    monitor_options.pm_sampling_interval_us = options.pm_sampling_interval_us;
    monitor_options.pm_sampling_max_samples = options.pm_sampling_max_samples;
    monitor_options.pm_sampling_preset = options.pm_sampling_preset;
    monitor_options.pm_sampling_metrics = options.pm_sampling_metrics;
    monitor_options.pm_sampling_scope_only = options.pm_sampling_scope_only;
    monitor_options.deep_arm_mode = options.deep_window_only
        ? DeepArmMode::WindowOnly
        : DeepArmMode::Always;
    applyDeepArmOverride(monitor_options);
    if (monitor_options.deep_arm_mode == DeepArmMode::WindowOnly) {
        monitor_options.pm_sampling_scope_only = true;
    }
    monitor_options.backend_kind = toMonitorBackendKind(options.backend);
    return monitor_options;
}

}  // namespace gpufl::detail
