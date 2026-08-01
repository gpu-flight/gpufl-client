#pragma once
#include <cstdint>
#include <string>

namespace gpufl {

/**
 * NVTX-style named range, captured via CUPTI_ACTIVITY_KIND_MARKER (or
 * gpufl's NVTX injection under `gpufl trace`). Sources include:
 *   - PyTorch's automatic NVTX annotations (via torch.cuda.nvtx or our
 *     gpufl.torch.TorchDispatchMode wrapping)
 *   - cuDNN / cuBLAS / NCCL / TensorRT which emit NVTX internally
 *   - User-emitted nvtxRangePush / nvtxMarkA calls
 *
 * gpufl's own scopes are NOT emitted here - they are recorded as
 * scope_event only. Paired START/END records from CUPTI are merged in the
 * client before emitting; one NvtxMarkerEvent per completed range.
 */
struct NvtxMarkerEvent {
    int pid = 0;
    std::string app;
    std::string session_id;
    std::string name;           // Range name (NVTX push argument)
    std::string domain;         // NVTX domain, "" for default
    int64_t start_ns = 0;
    int64_t end_ns = 0;
    int64_t duration_ns = 0;    // Redundant with end-start; kept for convenience
    uint32_t marker_id = 0;     // CUPTI marker ID (for debug / dedup)
};

}  // namespace gpufl
