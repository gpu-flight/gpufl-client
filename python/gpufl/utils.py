import time
import gpufl as gfl
import sys

try:
    from numba import cuda
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

def _to_dim3_str(val):
    if isinstance(val, int):
        return f"({val},1,1)"
    if isinstance(val, (tuple, list)):
        x = val[0] if len(val) > 0 else 1
        y = val[1] if len(val) > 1 else 1
        z = val[2] if len(val) > 2 else 1
        return f"({x},{y},{z})"
    return "(1,1,1)"

def launch_kernel(kernel_func, grid, block, *args):
    """
    Executes a Numba CUDA kernel wrapped in a GPUFL KernelScope.
    """
    if not HAS_NUMBA:
        raise ImportError("Numba is required to use 'launch_kernel'.")

    if getattr(gfl, 'KernelScope', None) is None:
        kernel_func[grid, block](*args)
        cuda.synchronize()
        return

    grid_str = _to_dim3_str(grid)
    block_str = _to_dim3_str(block)

    # Since Numba introspection is failing/unsupported, we pass 0 defaults.
    # This prevents crashes and keeps logs clean.
    with gfl.KernelScope(
            kernel_func.__name__,
            tag="numba",
            grid=grid_str,
            block=block_str,
            occupancy=0.0,
            maxActiveBlocks=0
    ):
        kernel_func[grid, block](*args)
        cuda.synchronize()