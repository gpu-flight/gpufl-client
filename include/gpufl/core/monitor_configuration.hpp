#pragma once

#include "gpufl/core/monitor.hpp"
#include "gpufl/gpufl.hpp"

namespace gpufl::detail {

// Builds the deterministic MonitorOptions contract from InitOptions and the
// documented environment overrides. CUDA/process side effects deliberately do
// not belong here; eager module loading and device-specific tuning remain at
// the initialization boundary.
MonitorOptions buildMonitorOptions(const InitOptions& options);

}  // namespace gpufl::detail
