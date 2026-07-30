#pragma once

namespace gpufl::segmentation {

// Both the launcher and injected runtime consult this single gate so direct
// environment injection cannot bypass the same execution-boundary decision.
// SegmentCoordinator now rotates the complete runtime context, including
// ownership, dictionary, scope-continuation, lifecycle, and terminal snapshots.
inline constexpr bool kRuntimeReady = true;

}  // namespace gpufl::segmentation
