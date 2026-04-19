"""
gpufl.torch — PyTorch integration for GPUFlight.

Captures PyTorch op dispatches and emits NVTX ranges that GPUFlight's
CUPTI marker path picks up. The result: every CUDA kernel in your
dashboard is annotated with the torch op that launched it (e.g.
`aten::matmul` at `my_model.py:42`), not just the raw kernel name.

Quick start:

    import gpufl
    import gpufl.torch
    import torch

    gpufl.init(app_name="my-training-run")
    gpufl.torch.attach()

    # Your existing training code — no other changes required.
    for batch in loader:
        out = model(batch)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    gpufl.torch.detach()
    gpufl.shutdown()

Alternatively use the context manager for a scoped region:

    with gpufl.torch.profile("one_step"):
        out = model(batch)
        loss = criterion(out, target)
        loss.backward()

How it works:
  - TorchDispatchMode catches every `aten::` op call at the dispatcher
    level (before decomposition into CUDA kernels)
  - Wraps the op in `torch.cuda.nvtx.range_push/pop` with a name that
    includes the op name and the user's Python call site
  - GPUFlight's C++ client captures those NVTX ranges via CUPTI
    (CUPTI_ACTIVITY_KIND_MARKER) — same pipeline that picks up GFL_SCOPE
  - Kernels launched inside the range are attributed to the op in the
    dashboard

Overhead: ~1-2 microseconds per op. For training loops doing 10k ops/sec
this is ~1-2% overhead — acceptable for nearly all workloads. If you
need zero overhead during hot paths, use `gpufl.torch.detach()` and
rely on your own GFL_SCOPE annotations.

See docs/python/pytorch.md for the full usage guide.
"""

from __future__ import annotations

from .dispatch import attach, detach
from .profile import profile, watch
from .trace_import import import_trace

__all__ = ["attach", "detach", "profile", "watch", "import_trace"]
