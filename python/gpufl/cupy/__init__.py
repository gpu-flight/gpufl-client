"""
gpufl.cupy — planned CuPy integration (stub during private beta).

CuPy kernels launch via the same CUDA driver path CUPTI already hooks, so
the C++ client captures their execution transparently. What's not yet
available is Python-side op attribution (mapping a CUDA kernel back to
the CuPy call that launched it, like we do for PyTorch).

For now, the most useful thing you can do with CuPy + GPUFlight is
manually annotate regions of interest with CuPy's built-in NVTX:

    import gpufl
    import cupy
    from cupy.cuda import nvtx

    gpufl.init(app_name="my-cupy-app")

    nvtx.RangePush("matmul-batch")
    result = cupy.matmul(a, b)
    nvtx.RangePop()

    gpufl.shutdown()

GPUFlight captures those NVTX ranges via its CUPTI marker path, so the
`matmul-batch` region will appear in the dashboard alongside the CUDA
kernels it covers. No additional library needed — the stub here exists
so `import gpufl.cupy` gives a clear message rather than
`ModuleNotFoundError`.

Track full integration progress at:
  https://github.com/gpu-flight/gpufl-client/issues  (filter: "cupy")
"""

from __future__ import annotations

_NOT_IMPLEMENTED_MESSAGE = (
    "CuPy integration is planned for a future release.\n"
    "Track at https://github.com/gpu-flight/gpufl-client/issues (filter: cupy).\n"
    "\n"
    "Workaround that works today: manually annotate regions with CuPy's NVTX, e.g.\n"
    "    from cupy.cuda import nvtx\n"
    "    nvtx.RangePush('my-region')\n"
    "    # ... CuPy work ...\n"
    "    nvtx.RangePop()\n"
    "GPUFlight captures these ranges via CUPTI markers. See module docstring."
)


def attach(*args, **kwargs):
    """Placeholder. Will enable CuPy op-level capture once implemented."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)


def detach(*args, **kwargs):
    """Placeholder. Will stop CuPy op-level capture once implemented."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)


def profile(*args, **kwargs):
    """Placeholder. Will return a context manager wrapping attach()/detach()."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)


def import_trace(*args, **kwargs):
    """Placeholder. CuPy has no standard trace format equivalent to torch.profiler."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MESSAGE)


__all__ = ["attach", "detach", "profile", "import_trace"]
