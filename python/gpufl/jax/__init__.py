"""
gpufl.jax — planned JAX integration (stub during private beta).

JAX compiles Python code to XLA kernels, which are then dispatched as
CUDA kernels. GPUFlight's C++ client captures those CUDA kernels via
CUPTI, but the JAX-level semantic (which `jit`-compiled function, which
primitive) is lost in the XLA compilation step.

Full JAX integration will hook into JAX's tracing machinery to restore
op-level attribution. This is more involved than the PyTorch integration
because XLA flattens the trace rather than dispatching op-by-op at
runtime.

For now, you can manually annotate regions with NVTX — JAX doesn't have
a built-in NVTX API, so use GPUFlight's GFL_SCOPE macro (via
`gpufl.Scope`) at the boundaries of the jitted functions you care about:

    import gpufl
    import jax
    import jax.numpy as jnp

    gpufl.init(app_name="my-jax-app")

    @jax.jit
    def my_model(x):
        ...

    with gpufl.Scope("forward_pass"):
        out = my_model(batch)
        out.block_until_ready()   # force sync so the scope covers GPU work

    gpufl.shutdown()

Track full integration progress at:
  https://github.com/gpu-flight/gpufl-client/issues  (filter: "jax")
"""

from __future__ import annotations

_NOT_IMPLEMENTED_MESSAGE = (
    "JAX integration is planned for a future release.\n"
    "Track at https://github.com/gpu-flight/gpufl-client/issues (filter: jax).\n"
    "\n"
    "Workaround that works today: wrap jitted calls with gpufl.Scope(), e.g.\n"
    "    with gpufl.Scope('forward_pass'):\n"
    "        out = my_jit_fn(batch)\n"
    "        out.block_until_ready()   # sync so scope covers GPU work\n"
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
