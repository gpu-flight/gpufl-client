#pragma once

// Umbrella header. The telemetry event structs are defined per family under
// events/, mirroring the per-family serializers in model/. Every existing
// translation unit includes "gpufl/core/events.hpp" and continues to see all
// of them, exactly as when this file held every struct in one place.
//
// New code may include a single family header (e.g. events/kernel_events.hpp)
// to keep its translation unit lean; the shared sample types live in
// events/sample_types.hpp.
#include "gpufl/core/events/deep_window_events.hpp"
#include "gpufl/core/events/graph_events.hpp"
#include "gpufl/core/events/kernel_events.hpp"
#include "gpufl/core/events/lifecycle_events.hpp"
#include "gpufl/core/events/memory_events.hpp"
#include "gpufl/core/events/nvtx_events.hpp"
#include "gpufl/core/events/perf_events.hpp"
#include "gpufl/core/events/sample_types.hpp"
#include "gpufl/core/events/scope_events.hpp"
#include "gpufl/core/events/sync_events.hpp"
#include "gpufl/core/events/system_events.hpp"
