import gpufl
from gpufl.utils import launch_kernel
from numba import cuda
import numpy as np

@cuda.jit
def vector_add(a, b, c):
    idx = cuda.grid(1)
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

def run():
    gpufl.init("Kernel_Test", "./logs", 10)

    N = 1024 * 1024
    a = cuda.to_device(np.ones(N))
    b = cuda.to_device(np.ones(N))
    c = cuda.device_array(N)

    threads = 256
    blocks = (N + 255) // 256

    print("Launching Kernel with Monitoring...")

    # Use the wrapper instead of raw call
    # Raw: vector_add[blocks, threads](a, b, c)
    # Monitored:
    launch_kernel(vector_add, blocks, threads, a, b, c)

    gpufl.shutdown()
    print("Done. Check logs/gpufl.kernel.log")

if __name__ == "__main__":
    run()