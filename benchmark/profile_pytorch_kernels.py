"""
PyTorch kernel-duration ground truth - confirms (or refutes) the
hypothesis that PyTorch MiniGPT's +657% PC Sampling overhead comes
from very short kernels combined with high actual launch count.

Hypothesis (from manykernel_benchmark.cu data):
  - CUPTI per-launch instrumentation cost on Blackwell WDDM ≈ 10-15us
  - Workloads with kernels >> 13us see relative overhead ≤ +50%
  - Workloads with kernels ~1us see relative overhead +150% (our C++ test)
  - PyTorch MiniGPT saw +657% - which implies (a) kernels are very short
    AND (b) actual kernel count is several times the visible op count
    (cuDNN/cuBLAS internal launches under each torch op)

This script measures (a) and (b) directly using torch.profiler - no
gpufl, no CUPTI through us, just torch's native kineto-backed kernel
recording. We then compare the numbers against our hypothesis.

Usage:
    python profile_pytorch_kernels.py
    python profile_pytorch_kernels.py --steps 20 --top 30
    python profile_pytorch_kernels.py --trace out.json   # also export chrome trace

The chrome trace is openable in chrome://tracing or ui.perfetto.dev
for visual inspection if the table summary leaves questions.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.profiler
import torch.nn.functional as F

# Reuse the exact MiniGPT model from the benchmark - we want kernel
# stats from the *same* workload that produced +657% overhead, not a
# subtly different one.
sys.path.insert(0, str(Path(__file__).parent))
from pytorch_train import MiniGPT  # noqa: E402


def run_and_profile(steps: int, warmup_steps: int, batch_size: int,
                    seq_len: int, trace_path: str | None = None):
    device = torch.device('cuda')
    model = MiniGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler()

    # Same warmup as pytorch_train.py - first-launch costs (CUDA context,
    # cuDNN/cuBLAS algo selection, kernel JIT) aren't part of steady-
    # state per-kernel overhead. Drain them outside the profile.
    for _ in range(warmup_steps):
        x = torch.randint(0, 512, (batch_size, seq_len), device=device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Profile the steady-state region. record_shapes=False keeps the
    # profiler's own overhead minimal; we're after kernel counts and
    # durations, not call-site provenance.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            x = torch.randint(0, 512, (batch_size, seq_len), device=device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"\n[chrome trace] written to: {trace_path}\n")

    return prof


def summarize(prof, steps: int, top: int):
    # Walk key_averages and split into CPU ops vs CUDA kernel records.
    # The CUDA kernel records come from kineto's CUPTI activity stream -
    # they're true kernel launches (one row per unique kernel name, with
    # `count` aggregating all launches of that kernel). That's exactly
    # the data we need to compute per-kernel average duration.
    key_avgs = prof.key_averages()

    cuda_kernels = []
    cpu_ops_with_cuda_time = []
    for k in key_avgs:
        # Newer PyTorch exposes device_type; check defensively.
        is_cuda_kernel = (
            getattr(k, 'device_type', None) == torch.autograd.DeviceType.CUDA
        )
        cuda_time_us = getattr(k, 'cuda_time_total', 0)
        count = max(1, k.count)
        if is_cuda_kernel and cuda_time_us > 0:
            cuda_kernels.append((k.key, count, cuda_time_us))
        elif cuda_time_us > 0:
            # CPU op that triggered CUDA work - count of underlying
            # kernel launches is hidden inside the op. Useful for the
            # "how many cuBLAS/cuDNN kernels does each torch op spawn"
            # multiplier story but doesn't give per-kernel duration.
            cpu_ops_with_cuda_time.append((k.key, count, cuda_time_us))

    if not cuda_kernels:
        print("[!] No CUDA kernel records found in profile. Falling back to CPU-op view.")
        print("    PyTorch version may be too old for kineto kernel records, or")
        print("    CUDA activities weren't enabled.")
        return

    total_launches = sum(c for _, c, _ in cuda_kernels)
    total_cuda_us = sum(t for _, _, t in cuda_kernels)
    avg_us = total_cuda_us / total_launches if total_launches else 0
    distinct = len(cuda_kernels)

    print("=" * 70)
    print("PyTorch MiniGPT - kernel-level ground truth")
    print("=" * 70)
    print(f"  Steps profiled:                {steps}")
    print(f"  Distinct CUDA kernel symbols:  {distinct}")
    print(f"  Total kernel LAUNCHES:         {total_launches}")
    print(f"  Launches per step:             {total_launches / steps:.1f}")
    print(f"  Total CUDA time:               {total_cuda_us / 1000:.1f} ms")
    print(f"  Average per-kernel duration:   {avg_us:.2f} us")
    print()
    print("Hypothesis check:")
    if avg_us < 3.0:
        verdict_dur = f"✓ confirms 'short kernel' hypothesis (avg < 3us)"
    elif avg_us < 8.0:
        verdict_dur = "partial - kernels are short-ish (avg 3-8us) but not as tiny as predicted"
    else:
        verdict_dur = f"✗ refutes short-kernel hypothesis (avg = {avg_us:.1f}us)"
    print(f"  Per-kernel duration:           {verdict_dur}")
    multiplier = total_launches / steps
    if multiplier > 100:
        verdict_mult = f"✓ confirms 'high actual launch count' (>{multiplier:.0f}/step vs ~20 visible ops)"
    elif multiplier > 50:
        verdict_mult = f"partial - {multiplier:.0f} launches/step"
    else:
        verdict_mult = f"unexpected - only {multiplier:.0f} launches/step"
    print(f"  Launches per step:             {verdict_mult}")
    print()
    print(f"Top {top} kernel symbols by total CUDA time:")
    print(f"{'Kernel':<55} {'Count':>8} {'Total (us)':>12} {'Avg (us)':>10}")
    print("-" * 88)
    for name, count, total in sorted(cuda_kernels, key=lambda x: -x[2])[:top]:
        avg = total / count
        # Truncate long mangled cuBLAS/cuDNN names
        short = name if len(name) <= 53 else (name[:50] + "...")
        print(f"{short:<55} {count:>8} {total:>12.1f} {avg:>10.2f}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20,
                   help='measured steps (matches pytorch_train.py default)')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seq', type=int, default=128)
    p.add_argument('--top', type=int, default=20,
                   help='show top-N kernels in the table')
    p.add_argument('--trace', type=str, default=None,
                   help='also export chrome trace JSON to this path')
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available - this script needs a GPU.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Profiling MiniGPT for {args.steps} steps "
          f"(batch={args.batch}, seq={args.seq}, warmup={args.warmup})...")

    prof = run_and_profile(args.steps, args.warmup, args.batch, args.seq, args.trace)
    summarize(prof, args.steps, args.top)


if __name__ == '__main__':
    main()
