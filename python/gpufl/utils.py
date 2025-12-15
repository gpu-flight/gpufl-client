import time
import gpufl as gfl
import sys

try:
    from numba import cuda
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
def launch_kernel(kernel_func, grid, block, *args):
    """
    Python equivalent of GFL_LAUNCH.
    Executes kernel, syncs, and logs the specific kernel duration.
    """
    if not HAS_NUMBA:
        raise ImportError("Numba is required to use 'launch_kernel'. Please run: pip install numba")

    # 1. Capture Start
    start_ns = time.monotonic_ns()

    # 2. Run Kernel
    kernel_func[grid, block](*args)

    # 3. Synchronize (Essential!)
    cuda.synchronize()

    # 4. Capture End
    end_ns = time.monotonic_ns()

    # 5. Log
    # Handle tuple/int formats for grid/block
    gx, gy, gz = (grid + (1, 1))[:3] if isinstance(grid, tuple) else (grid, 1, 1)
    bx, by, bz = (block + (1, 1))[:3] if isinstance(block, tuple) else (block, 1, 1)

    gfl.log_kernel(kernel_func.__name__, gx, gy, gz, bx, by, bz, start_ns, end_ns)