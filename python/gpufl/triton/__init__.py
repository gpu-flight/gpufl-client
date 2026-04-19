"""
gpufl.triton — planned Triton integration (stub during private beta).

Triton is a Python DSL for writing custom GPU kernels. When a Triton
kernel runs, GPUFlight's C++ client captures the CUDA launch via CUPTI
as usual — so you already see execution time, occupancy, and SASS for
Triton kernels. What's not yet available is mapping the generated kernel
back to the Triton source function and line that authored it.

Full integration will:
  - Hook Triton's JIT compiler to capture source metadata
  - Correlate generated kernel names (e.g. `triton__0d1d2d3d`) back to
    the `@triton.jit` function of origin
  - Surface the Triton source in the kernel-detail view

In the meantime, wrap your Triton kernels in a GPUFlight scope so they
appear by name in the dashboard:

    import gpufl
    import triton
    import triton.language as tl

    @triton.jit
    def my_kernel(...):
        ...

    with gpufl.Scope("my_triton_kernel"):
        my_kernel[grid](...)

Track full integration progress at:
  https://github.com/gpu-flight/gpufl-client/issues  (filter: "triton")
"""

from __future__ import annotations

_NOT_IMPLEMENTED_MESSAGE = (
    "Triton integration is planned for a future release.\n"
    "Track at https://github.com/gpu-flight/gpufl-client/issues (filter: triton).\n"
    "\n"
    "Workaround that works today: wrap Triton launches with gpufl.Scope(), e.g.\n"
    "    with gpufl.Scope('my_kernel'):\n"
    "        my_kernel[grid](...)\n"
    "Kernel-level profiling (execution time, occupancy, SASS) works today.\n"
    "Missing piece is source attribution to the @triton.jit function."
)


def attach(*args, **kwargs):
    raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)


def detach(*args, **kwargs):
    raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)


def profile(*args, **kwargs):
    raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)


def import_trace(*args, **kwargs):
    raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)


__all__ = ["attach", "detach", "profile", "import_trace"]
