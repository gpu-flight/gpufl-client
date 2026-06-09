#include "gpufl/core/config_file_loader.hpp"

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/json/json.hpp"

namespace gpufl {

void ConfigFileLoader::apply(InitOptions& opts, const std::string& path) {
    auto cfg = json::loadFile(path);
    if (cfg.is_null()) {
        GFL_LOG_ERROR("Config file not found or invalid JSON: ", path);
        return;
    }
    if (!cfg.is_object() || cfg.empty()) return;

    GFL_LOG_DEBUG("Config file loaded: ", path, " (", cfg.size(), " keys)");

    // Profiling engine (string → enum). Accepts the six canonical
    // engine names (see ProfilingEngine in monitor.hpp). Unrecognized
    // values leave the existing default untouched.
    if (cfg.contains("profiling_engine")) {
        const auto& v = cfg["profiling_engine"].get_string();
        if (v == "Monitor")            opts.profiling_engine = ProfilingEngine::Monitor;
        else if (v == "Trace")         opts.profiling_engine = ProfilingEngine::Trace;
        else if (v == "PcSampling")    opts.profiling_engine = ProfilingEngine::PcSampling;
        else if (v == "SassMetrics")   opts.profiling_engine = ProfilingEngine::SassMetrics;
        else if (v == "PmSampling")    opts.profiling_engine = ProfilingEngine::PmSampling;
        else if (v == "RangeProfiler") opts.profiling_engine = ProfilingEngine::RangeProfiler;
        else if (v == "RangeProfilerKernelReplay")
            opts.profiling_engine = ProfilingEngine::RangeProfilerKernelReplay;
        else if (v == "Deep")          opts.profiling_engine = ProfilingEngine::Deep;
    }

    // Integer fields
    if (cfg.contains("system_sample_rate_ms"))
        opts.system_sample_rate_ms = cfg.value<int>("system_sample_rate_ms", opts.system_sample_rate_ms);
    if (cfg.contains("kernel_sample_rate_ms"))
        opts.kernel_sample_rate_ms = cfg.value<int>("kernel_sample_rate_ms", opts.kernel_sample_rate_ms);
    if (cfg.contains("pm_sampling_interval_us"))
        opts.pm_sampling_interval_us = static_cast<uint32_t>(cfg.value<int>("pm_sampling_interval_us", static_cast<int>(opts.pm_sampling_interval_us)));
    if (cfg.contains("pm_sampling_max_samples"))
        opts.pm_sampling_max_samples = static_cast<uint32_t>(cfg.value<int>("pm_sampling_max_samples", static_cast<int>(opts.pm_sampling_max_samples)));

    // String fields
    if (cfg.contains("api_path") && cfg["api_path"].is_string())
        opts.api_path = cfg["api_path"].get_string();
    if (cfg.contains("pm_sampling_preset") && cfg["pm_sampling_preset"].is_string())
        opts.pm_sampling_preset = cfg["pm_sampling_preset"].get_string();

    // Boolean fields
    if (cfg.contains("enable_stack_trace"))
        opts.enable_stack_trace = cfg.value<bool>("enable_stack_trace", opts.enable_stack_trace);
    // Backward compat: silently ignore "enable_kernel_details" if a
    // legacy config file still sets it. Kernel details are now always
    // captured — see gpufl.hpp. Old configs continue to load cleanly
    // without erroring on the unknown key.
    if (cfg.contains("enable_source_collection"))
        opts.enable_source_collection = cfg.value<bool>("enable_source_collection", opts.enable_source_collection);
    if (cfg.contains("enable_debug_output"))
        opts.enable_debug_output = cfg.value<bool>("enable_debug_output", opts.enable_debug_output);
    if (cfg.contains("enable_external_correlation"))
        opts.enable_external_correlation = cfg.value<bool>("enable_external_correlation", opts.enable_external_correlation);
    if (cfg.contains("pm_sampling_scope_only"))
        opts.pm_sampling_scope_only = cfg.value<bool>("pm_sampling_scope_only", opts.pm_sampling_scope_only);
}

}  // namespace gpufl
