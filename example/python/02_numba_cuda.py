import gpufl as gfl
import math
import numpy as np
import os
import time
from gpufl.report import generate_report
from numba import cuda


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
    # LOG_PATH is the session output root. FileLogSink writes
    # <LOG_PATH>/<session_id>/<channel>.log[.gz], and generate_report()
    # can read the same directory later.
    LOG_PATH = "./gfl_logs"

    BACKEND_URL = os.environ.get("GPUFL_BACKEND_URL", "https://api.gpuflight.com")
    API_KEY     = os.environ.get("GPUFL_API_KEY", "")
    UPLOAD_ENABLED = bool(API_KEY)

    print("[GPUFL] Initializing...")
    if UPLOAD_ENABLED:
        print(f"[GPUFL] Deferred upload ON — will ship to {BACKEND_URL} "
              f"after shutdown.")
    else:
        print("[GPUFL] Upload OFF (set GPUFL_API_KEY to enable). Local "
              "files only — run `python -m gpufl.cli upload <log_path>` "
              "(or the `gpufl upload` binary on Linux) later to ship them.")

    gfl.init(
        app_name="Numba_App",
        log_path=LOG_PATH,
        continuous_system_sampling=True,
        system_sample_rate_ms=100,
        enable_debug_output=True,
        profiling_engine=gfl.ProfilingEngine.Deep,
        backend_url=BACKEND_URL,
        api_key=API_KEY,
    )

    try:
        # --- 3. Setup Data (Heavy Load) ---
        N = 2048  # 2048x2048 matrix = decent workload for testing
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

        # --- 5b. Deferred upload (replaces the old remote_upload=True
        # live-streaming flag). Upload runs AFTER shutdown, never during
        # the workload — so network failures here can't affect the GPU
        # job's exit code or performance. Skipped if no API key is set;
        # the local NDJSON files remain on disk and can be uploaded
        # later via `gpufl upload <LOG_PATH>`.
        if UPLOAD_ENABLED:
            print("[GPUFL] Uploading session to backend...")
            r = gfl.upload_logs(
                log_path=LOG_PATH,
                backend_url=BACKEND_URL,
                api_key=API_KEY,
            )
            if r.success:
                print(f"[GPUFL] Upload OK — {r.events_uploaded} events, "
                      f"{r.elapsed_ms / 1000:.1f}s")
            else:
                print(f"[GPUFL] Upload FAILED — {len(r.warnings)} warning(s):")
                for w in r.warnings:
                    print(f"  - {w}")

        # --- 6. Generate a text report from the logs we just wrote ---
        # shutdown() above flushes and closes the NDJSON channels, so the
        # report reflects the full session. generate_report reads the same
        # logs the analyzer uses — no GPU required for this step. We split
        # LOG_PATH into (dir, prefix) the way GpuFlightSession expects:
        #   "./gfl_logs" -> dir=".", prefix="gfl_logs"
        #                -> reads ./gfl_logs.{device,scope,system}.log
        # Wrap in print() so the report renders with real newlines (and,
        # in a Jupyter notebook, in the monospace stdout area so the
        # kernel tables stay aligned).
        log_dir = os.path.dirname(LOG_PATH) or "."
        log_prefix = os.path.basename(LOG_PATH)
        print("\n[GPUFL] Session report:\n")
        print(generate_report(log_dir, log_prefix=log_prefix, top_n=10))


if __name__ == "__main__":
    if cuda.is_available():
        run_benchmark()
    else:
        print("Skipping: No CUDA device found (Running in CI?)")
