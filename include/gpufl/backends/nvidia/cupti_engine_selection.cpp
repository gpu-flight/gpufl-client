#include "gpufl/backends/nvidia/cupti_engine_selection.hpp"

#include <cstdlib>
#include <string>
#include <utility>

#include "gpufl/backends/nvidia/engine/composite_engine.hpp"
#include "gpufl/backends/nvidia/engine/pc_sampling_engine.hpp"
#include "gpufl/backends/nvidia/engine/pc_sampling_with_sass_engine.hpp"
#include "gpufl/backends/nvidia/engine/pm_sampling_engine.hpp"
#include "gpufl/backends/nvidia/engine/range_profiler_engine.hpp"
#include "gpufl/backends/nvidia/engine/sass_metrics_engine.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/env_vars.hpp"

namespace gpufl {

// Parse GPUFL_ENGINE_COMBO (comma-separated canonical engine names) into an
// engine list. Unknown tokens are logged and skipped; empty/unset => {}.
std::vector<ProfilingEngine> ParseEngineComboEnv() {
    std::vector<ProfilingEngine> out;
    const char* raw = std::getenv(gpufl::env::kEngineCombo);
    if (!raw || raw[0] == '\0') return out;
    std::string s(raw);
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const size_t end = (comma == std::string::npos) ? s.size() : comma;
        std::string tok = s.substr(pos, end - pos);
        const auto b = tok.find_first_not_of(" \t\r\n");
        const auto e = tok.find_last_not_of(" \t\r\n");
        if (b != std::string::npos) {
            tok = tok.substr(b, e - b + 1);
            if (tok == "Monitor")            out.push_back(ProfilingEngine::Monitor);
            else if (tok == "Trace")         out.push_back(ProfilingEngine::Trace);
            else if (tok == "PcSampling")    out.push_back(ProfilingEngine::PcSampling);
            else if (tok == "SassMetrics")   out.push_back(ProfilingEngine::SassMetrics);
            else if (tok == "PmSampling")    out.push_back(ProfilingEngine::PmSampling);
            else if (tok == "RangeProfiler") out.push_back(ProfilingEngine::RangeProfiler);
            else if (tok == "RangeProfilerKernelReplay")
                out.push_back(ProfilingEngine::RangeProfilerKernelReplay);
            else if (tok == "Deep")          out.push_back(ProfilingEngine::Deep);
            else
                GFL_LOG_ERROR("[CuptiBackend] GPUFL_ENGINE_COMBO: unknown engine '",
                              tok, "' (ignored)");
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

EngineRequestSet BuildEngineRequestSet(
    ProfilingEngine selected,
    const std::vector<ProfilingEngine>& combo) {
    EngineRequestSet out;
    if (!combo.empty()) {
        for (const ProfilingEngine engine : combo) {
            out.trace = out.trace || engine == ProfilingEngine::Trace;
            out.pc = out.pc || engine == ProfilingEngine::PcSampling;
            out.sass = out.sass || engine == ProfilingEngine::SassMetrics;
            out.pm = out.pm || engine == ProfilingEngine::PmSampling;
            out.range = out.range || engine == ProfilingEngine::RangeProfiler;
            out.range_kernel = out.range_kernel ||
                engine == ProfilingEngine::RangeProfilerKernelReplay;
        }
        return out;
    }

    out.trace = selected == ProfilingEngine::Trace;
    out.pc = selected == ProfilingEngine::PcSampling ||
             selected == ProfilingEngine::Deep;
    out.sass = selected == ProfilingEngine::SassMetrics ||
               selected == ProfilingEngine::Deep;
    out.pm = selected == ProfilingEngine::PmSampling ||
             selected == ProfilingEngine::Deep;
    out.range = selected == ProfilingEngine::RangeProfiler;
    out.range_kernel = selected == ProfilingEngine::RangeProfilerKernelReplay;
    return out;
}

void ApplyComboPlanOverrides(ResolvedProfilingPlan& plan,
                             const std::vector<ProfilingEngine>& combo) {
    if (combo.empty()) return;
    const EngineRequestSet requests =
        BuildEngineRequestSet(ProfilingEngine::Monitor, combo);

    // PC / SASS read cubin binaries for source correlation and disassembly.
    // The policy resolver sees only the base single engine, so re-apply this
    // after every resolve while a combo is active.
    if (requests.needsCubin()) {
        plan.needs_cubin_capture = true;
    }

    // A combo dictates its own activity set. Single-engine SASS safe-mode
    // gating would otherwise suppress Trace's timeline activity when the base
    // engine happens to be SASS/Deep.
    plan.sass_metrics_only = false;
    plan.safe_sass_activity_defaults = false;
    plan.allow_sass_kernel_activity = true;
    plan.allow_sass_marker_activity = true;
    plan.allow_sass_mem_transfer_activity = true;
    plan.allow_sass_memory2_activity = true;
    plan.allow_sass_sync_activity = true;
    plan.allow_sass_graph_activity = true;
    plan.allow_sass_external_correlation = true;
}

std::unique_ptr<IProfilingEngine> CreateProfilingEngine(
    ProfilingEngine selected,
    const std::vector<ProfilingEngine>& combo) {
    if (!combo.empty()) {
        const EngineRequestSet requests =
            BuildEngineRequestSet(selected, combo);
        std::vector<std::unique_ptr<IProfilingEngine>> subs;
#if GPUFL_HAS_PERFWORKS
        if (requests.pm)
            subs.push_back(std::make_unique<PmSamplingEngine>());
#endif
        if (requests.sass)
            subs.push_back(std::make_unique<SassMetricsEngine>());
#if GPUFL_HAS_PERFWORKS
        if (requests.range)
            subs.push_back(std::make_unique<RangeProfilerEngine>());
        if (requests.range_kernel)
            subs.push_back(std::make_unique<RangeProfilerEngine>(
                RangeProfilerEngine::Mode::KernelReplay));
#endif
        if (requests.pc)
            subs.push_back(std::make_unique<PcSamplingEngine>());

        std::unique_ptr<IProfilingEngine> engine;
        if (!subs.empty()) {
            engine = std::make_unique<CompositeEngine>(std::move(subs));
        }
        GFL_LOG_DEBUG("[CuptiBackend] Engine: Composite combo (", combo.size(),
                      " entries; ", (engine ? "API engines armed" : "activity-only"),
                      ")");
        return engine;
    }

    // Create the engine (no CUDA context needed yet).
    switch (selected) {
        case ProfilingEngine::PcSampling: {
            GFL_LOG_DEBUG("[CuptiBackend] Engine: PcSampling");
            return std::make_unique<PcSamplingEngine>();
        }
        case ProfilingEngine::SassMetrics: {
            GFL_LOG_DEBUG("[CuptiBackend] Engine: SassMetrics");
            return std::make_unique<SassMetricsEngine>();
        }
        case ProfilingEngine::PmSampling:
#if GPUFL_HAS_PERFWORKS
            GFL_LOG_DEBUG("[CuptiBackend] Engine: PmSampling");
            return std::make_unique<PmSamplingEngine>();
#else
            GFL_LOG_ERROR(
                "[CuptiBackend] PmSampling engine requires "
                "GPUFL_HAS_PERFWORKS; falling back to kernel-trace only");
            return nullptr;
#endif
        case ProfilingEngine::RangeProfiler:
#if GPUFL_HAS_PERFWORKS
            GFL_LOG_DEBUG("[CuptiBackend] Engine: RangeProfiler");
            return std::make_unique<RangeProfilerEngine>();
#else
            GFL_LOG_ERROR(
                "[CuptiBackend] RangeProfiler engine requires "
                "GPUFL_HAS_PERFWORKS; falling back to kernel-trace only");
            return nullptr;
#endif
        case ProfilingEngine::RangeProfilerKernelReplay:
#if GPUFL_HAS_PERFWORKS
            GFL_LOG_DEBUG("[CuptiBackend] Engine: RangeProfilerKernelReplay");
            return std::make_unique<RangeProfilerEngine>(
                RangeProfilerEngine::Mode::KernelReplay);
#else
            GFL_LOG_ERROR(
                "[CuptiBackend] RangeProfilerKernelReplay engine requires "
                "GPUFL_HAS_PERFWORKS; falling back to kernel-trace only");
            return nullptr;
#endif
        case ProfilingEngine::Deep:
            // Deep = the deepest analysis the GPU supports. SASS (Profiler
            // API) and PC sampling are mutually exclusive on current drivers,
            // so the composite collects one or the other: SASS metrics on
            // Blackwell+ (where lazy patching is safe), PC sampling on older
            // GPUs (SASS lazy patching deadlocks against concurrent kernel
            // launches there - observed on Ampere sm_86). The engine logs
            // which path it selected in initialize().
            GFL_LOG_DEBUG("[CuptiBackend] Engine: Deep");
            return std::make_unique<PcSamplingWithSassEngine>();
        case ProfilingEngine::Trace:
        default:
            // No sampling engine - activity records only (kernels, memcpy,
            // sync). ProfilingEngine::Monitor never reaches here:
            // CreateMonitorAdapter returns nullptr for it, so no
            // CuptiBackend is created at all.
            GFL_LOG_DEBUG("[CuptiBackend] Engine: none (activity trace only)");
            return nullptr;
    }
}

EngineRuntimeState InspectEngineRuntimeState(const IProfilingEngine* engine,
                                             ProfilingEngine selected,
                                             bool comboActive) {
    EngineRuntimeState out;
    if (!engine) return out;

    auto observeConcrete = [&](const IProfilingEngine* sub) {
        if (!sub) return;
        const bool armed = sub->isOperational();
        const bool produced = sub->producedData();

        if (comboActive) {
            // Keep the matrix log close to the single place that inspects
            // runtime engine state.
            GFL_LOG_ERROR("[Composite][matrix] ", sub->name(), " armed=",
                          armed ? "yes" : "no", " produced=",
                          produced ? "yes" : "no");
        }

        if (dynamic_cast<const SassMetricsEngine*>(sub)) {
            out.sass.observe(armed, produced);
        } else if (dynamic_cast<const PcSamplingEngine*>(sub)) {
            out.pc.observe(armed, produced);
        } else if (dynamic_cast<const PmSamplingEngine*>(sub)) {
            out.pm.observe(armed, produced);
        } else if (const auto* range = dynamic_cast<const RangeProfilerEngine*>(sub)) {
            if (range->kernelReplayMode()) out.range_kernel.observe(armed, produced);
            else out.range.observe(armed, produced);
        }
    };

    if (comboActive) {
        if (const auto* comp = dynamic_cast<const CompositeEngine*>(engine)) {
            for (const auto& sub : comp->engines()) observeConcrete(sub.get());
        }
        return out;
    }

    if (const auto* deep = dynamic_cast<const PcSamplingWithSassEngine*>(engine)) {
        out.sass.observe(deep->sassActive(), deep->sassProducedData());
        out.pc.observe(deep->pcSamplingActive(), deep->pcProducedData());
        out.pm.observe(deep->pmSamplingActive(), deep->pmProducedData());
        return out;
    }

    const bool armed = engine->isOperational();
    const bool produced = engine->producedData();
    switch (selected) {
        case ProfilingEngine::SassMetrics:
            out.sass.observe(armed, produced);
            break;
        case ProfilingEngine::PcSampling:
            out.pc.observe(armed, produced);
            break;
        case ProfilingEngine::PmSampling:
            out.pm.observe(armed, produced);
            break;
        case ProfilingEngine::RangeProfiler:
            out.range.observe(armed, produced);
            break;
        case ProfilingEngine::RangeProfilerKernelReplay:
            out.range_kernel.observe(armed, produced);
            break;
        default:
            break;
    }
    return out;
}

}  // namespace gpufl
