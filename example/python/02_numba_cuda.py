import gpufl as gfl
import numpy as np
from numba import cuda
import math
import time

# --- 1. Define a Real CUDA Kernel (Matrix Mul) ---
@cuda.jit
def matmul_kernel(A, B, C):
    """
    Standard CUDA Matrix Multiplication (Naive implementation for stress testing)
    C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def run_benchmark():
    # --- 2. Initialize GPUFL ---
    # We enable the background sampler (16ms) to catch VRAM/Power usage during the heavy compute
    print("[GPUFL] Initializing...")
    gfl.init("Numba_App", "./gfl_logs", 100)

    try:
        # --- 3. Setup Data (Heavy Load) ---
        N = 2048 # 2048x2048 matrix = decent workload for testing
        print(f"[Setup] Generating {N}x{N} matrices...")

        # Host memory
        A_h = np.random.rand(N, N).astype(np.float32)
        B_h = np.random.rand(N, N).astype(np.float32)
        C_h = np.zeros((N, N), dtype=np.float32)

        # Device memory (VRAM allocation)
        # We wrap this in a scope to see memory usage spike!
        with gfl.Scope("allocation_phase", "setup"):
            d_A = cuda.to_device(A_h)
            d_B = cuda.to_device(B_h)
            d_C = cuda.to_device(C_h)

        # Configure Grid/Block
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(N / threadsperblock[0])
        blockspergrid_y = math.ceil(N / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        print("[Compute] Launching CUDA Kernels...")

        # --- 4. Profile the Compute Phase ---
        # This Scope will measure exactly how long the GPU was busy
        with gfl.Scope("matrix_mul_compute", "math"):

            # Launch kernel 10 times to simulate a "Training Step"
            for i in range(10):
                matmul_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)

            # CRITICAL: Numba calls are async.
            # gfl.Scope automatically calls cudaDeviceSynchronize() on exit,
            # ensuring we capture the TRUE execution time, not just the launch time.

        # Retrieve result
        C_h = d_C.copy_to_host()
        print("[Success] Compute finished.")

    finally:
        # --- 5. Cleanup ---
        print("[GPUFL] Shutting down...")
        gfl.shutdown()

if __name__ == "__main__":
    if cuda.is_available():
        run_benchmark()
    else:
        print("Skipping: No CUDA device found (Running in CI?)")