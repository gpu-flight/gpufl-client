#pragma once

#include <memory>
#include <vector>

#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"
#include "gpufl/backends/nvidia/profiling_plan.hpp"
#include "gpufl/gpufl.hpp"

namespace gpufl {

struct EngineRequestSet {
    bool trace = false;
    bool pc = false;
    bool sass = false;
    bool pm = false;
    bool range = false;
    bool range_kernel = false;

    bool needsCubin() const { return pc || sass; }
    bool ownsTimelineActivity() const { return trace || pm || range; }
};

struct EnginePathState {
    bool active = false;
    bool has_data = false;

    void observe(bool armed, bool produced) {
        active = active || armed || produced;
        has_data = has_data || produced;
    }
};

struct EngineRuntimeState {
    EnginePathState sass;
    EnginePathState pc;
    EnginePathState pm;
    EnginePathState range;
    EnginePathState range_kernel;
};

std::vector<ProfilingEngine> ParseEngineComboEnv();
EngineRequestSet BuildEngineRequestSet(
    ProfilingEngine selected,
    const std::vector<ProfilingEngine>& combo);
void ApplyComboPlanOverrides(ResolvedProfilingPlan& plan,
                             const std::vector<ProfilingEngine>& combo);
std::unique_ptr<IProfilingEngine> CreateProfilingEngine(
    ProfilingEngine selected,
    const std::vector<ProfilingEngine>& combo);
EngineRuntimeState InspectEngineRuntimeState(const IProfilingEngine* engine,
                                             ProfilingEngine selected,
                                             bool comboActive);

}  // namespace gpufl
