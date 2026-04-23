"""
gpufl.numba — planned Numba CUDA integration (stub during private beta).

Numba's `@cuda.jit` compiles Python to PTX. The resulting kernels are
captured by GPUFlight's C++ client via CUPTI the same way any CUDA
kernel is. Full integration will add Python-function-name attribution so
the dashboard can show "this kernel came from `@cuda.jit def my_kernel`
at file.py:42".

Until full integration, wrap your Numba launches with GPUFlight scopes:

    import gpufl
    from numba import cuda

    @cuda.jit
    def my_kernel(a, b, c):
        i = cuda.grid(1)
        c[i] = a[i] + b[i]

    with gpufl.Scope("my_kernel_launch"):
        my_kernel[grid, block](a, b, c)
        cuda.synchronize()   # ensure scope covers GPU completion

Track full integration progress at:
  https://github.com/gpu-flight/gpufl-client/issues  (filter: "numba")
"""

from __future__ import annotations

_NOT_IMPLEMENTED_MESSAGE = (
    "Numba integration is planned for a future release.\n"
    "Track at https://github.com/gpu-flight/gpufl-client/issues (filter: numba).\n"
    "\n"
    "Workaround that works today: wrap Numba launches with gpufl.Scope(), e.g.\n"
    "    with gpufl.Scope('my_kernel_launch'):\n"
    "        my_kernel[grid, block](...)\n"
    "        cuda.synchronize()\n"
    "See module docstring for details."
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
