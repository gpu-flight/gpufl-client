"""
CUDA GEMM benchmark — measures raw GPU compute overhead.
Uses torch.mm() (cuBLAS SGEMM) for a pure compute workload.
Returns (wall_ms, gpu_ms) for the timed region.
"""

import torch
import time


def run_gemm(n: int = 2048, iterations: int = 100, warmup: int = 10) -> tuple[float, float]:
    """Run N×N matrix multiply for `iterations`, return (wall_ms, gpu_ms)."""
    device = torch.device('cuda')
    a = torch.randn(n, n, device=device, dtype=torch.float32)
    b = torch.randn(n, n, device=device, dtype=torch.float32)

    # Warm-up
    for _ in range(warmup):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    # Timed region
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_event.record()

    for _ in range(iterations):
        _ = torch.mm(a, b)

    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    gpu_ms = start_event.elapsed_time(end_event)
    wall_ms = (wall_end - wall_start) * 1000.0

    return wall_ms, gpu_ms


if __name__ == '__main__':
    wall, gpu = run_gemm()
    print(f"GEMM N=2048, 100 iters: wall={wall:.1f}ms, gpu={gpu:.1f}ms")
