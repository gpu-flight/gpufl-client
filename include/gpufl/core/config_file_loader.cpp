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

    // Profiling engine (string → enum)
    if (cfg.contains("profiling_engine")) {
        const auto& v = cfg["profiling_engine"].get_string();
        if (v == "None")                    opts.profiling_engine = ProfilingEngine::None;
        else if (v == "PcSampling")         opts.profiling_engine = ProfilingEngine::PcSampling;
        else if (v == "SassMetrics")        opts.profiling_engine = ProfilingEngine::SassMetrics;
        else if (v == "RangeProfiler")      opts.profiling_engine = ProfilingEngine::RangeProfiler;
        else if (v == "PcSamplingWithSass") opts.profiling_engine = ProfilingEngine::PcSamplingWithSass;
    }

    // Integer fields
    if (cfg.contains("system_sample_rate_ms"))
        opts.system_sample_rate_ms = cfg.value<int>("system_sample_rate_ms", opts.system_sample_rate_ms);
    if (cfg.contains("kernel_sample_rate_ms"))
        opts.kernel_sample_rate_ms = cfg.value<int>("kernel_sample_rate_ms", opts.kernel_sample_rate_ms);

    // Boolean fields
    if (cfg.contains("enable_stack_trace"))
        opts.enable_stack_trace = cfg.value<bool>("enable_stack_trace", opts.enable_stack_trace);
    if (cfg.contains("enable_kernel_details"))
        opts.enable_kernel_details = cfg.value<bool>("enable_kernel_details", opts.enable_kernel_details);
    if (cfg.contains("enable_source_collection"))
        opts.enable_source_collection = cfg.value<bool>("enable_source_collection", opts.enable_source_collection);
    if (cfg.contains("enable_debug_output"))
        opts.enable_debug_output = cfg.value<bool>("enable_debug_output", opts.enable_debug_output);
    if (cfg.contains("enable_external_correlation"))
        opts.enable_external_correlation = cfg.value<bool>("enable_external_correlation", opts.enable_external_correlation);
}

}  // namespace gpufl
