"""
PyTorch kernel-duration ground truth via gpufl logs.

Background: torch.profiler/kineto fails on Blackwell laptop
(CUPTI_ERROR_INVALID_DEVICE - PyTorch's bundled kineto/CUPTI version
isn't sm_120-compatible). Our own CUPTI integration works on this
GPU, so gpufl-client's monitoring mode produces the same kernel-
duration data we wanted from torch.profiler.

This script:
  1. gpufl.init(profiling_engine=None, monitoring only)
  2. runs MiniGPT 20 steps (the exact workload from pytorch_train.py)
  3. gpufl.shutdown() - flushes/gzips device.log
  4. parses the resulting device.log.gz for kernel_event_batch records
  5. computes kernel count, per-kernel duration distribution
  6. confirms or refutes the short-kernel hypothesis

Output mirrors what profile_pytorch_kernels.py would have printed:
  - Total kernel launches
  - Launches per step
  - Average per-kernel duration
  - Top-N kernels by total time

Usage:
    python profile_pytorch_via_gpufl.py
    python profile_pytorch_via_gpufl.py --steps 30 --top 30
    python profile_pytorch_via_gpufl.py --log-dir D:\gpufl_minigpt_logs
"""

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_train import MiniGPT  # noqa: E402


def run_minigpt_under_gpufl(steps: int, warmup_steps: int, batch_size: int,
                             seq_len: int, log_dir: Path):
    """Run MiniGPT with gpufl monitoring; return nothing - data ends up
    in <log_dir>/<session_id>/device.log.gz once gpufl.shutdown() runs."""
    import gpufl  # imported here so the script can still print a clear
                  # error if gpufl isn't installed
    log_dir.mkdir(parents=True, exist_ok=True)

    # Trace mode - kernel activity records are on (that's what gives us
    # duration_ns) but PC sampling is OFF, so we're not measuring the
    # +657% overhead case, just the kernel-timing ground truth. (Note:
    # NOT Monitor - that disables CUPTI entirely and would yield zero
    # kernel records. We need the activity trace.) PC sampling on or off,
    # the duration_ns values are identical - they come from CUPTI's
    # KIND_KERNEL records either way.
    ok = gpufl.init(
        "minigpt_kernel_profile",
        str(log_dir),
        True,  # continuous_system_sampling - unused but historically positional
        0,     # system_sample_rate_ms - off
        profiling_engine=gpufl.ProfilingEngine.Trace,
        enable_debug_output=False,  # debug stderr spam dominated overhead
                                    # in earlier runs - keep off for clean
                                    # kernel timing
    )
    if not ok:
        print("[!] gpufl.init returned False (stub mode?). "
              "Cannot collect kernel data without CUPTI active.")
        return False

    device = torch.device('cuda')
    model = MiniGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler()

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

    # The measured region. Wrap in a scope so the kernels are
    # attributed (matches the recommended usage pattern). The scope
    # doesn't change kernel timings; it just buckets the data.
    with gpufl.Scope("minigpt_train"):
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

    gpufl.shutdown()
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return True


def find_device_log(log_dir: Path) -> Path | None:
    """Pick the most recent session subdir under log_dir and return its
    device.log[.gz]. If gpufl.init ran more than once into the same
    directory (e.g. previous incomplete runs), the latest mtime wins."""
    if not log_dir.is_dir():
        return None
    sessions = [p for p in log_dir.iterdir() if p.is_dir()]
    if not sessions:
        return None
    latest = max(sessions, key=lambda p: p.stat().st_mtime)
    for candidate in (latest / 'device.log.gz', latest / 'device.log'):
        if candidate.exists():
            return candidate
    return None


def parse_kernel_durations(device_log: Path):
    """Walk device.log[.gz] for kernel_event_batch records and yield
    (name_id, duration_ns) for every row. duration_ns is what CUPTI
    reported for that specific launch."""
    opener = gzip.open if device_log.suffix == '.gz' else open
    with opener(device_log, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # The gpufl wire format for kernel batches uses a
            # column-oriented "columns" + "rows" structure with
            # duration_ns somewhere in the column list. Look it up by
            # name so we don't depend on positional stability.
            if rec.get('type') != 'kernel_event_batch':
                continue
            cols = rec.get('columns') or []
            try:
                dur_idx = cols.index('duration_ns')
            except ValueError:
                continue
            name_idx = cols.index('kernel_id') if 'kernel_id' in cols else None
            for row in rec.get('rows') or []:
                if dur_idx >= len(row):
                    continue
                dur_ns = row[dur_idx]
                name_id = row[name_idx] if name_idx is not None and name_idx < len(row) else None
                yield name_id, dur_ns


def lookup_kernel_dict(device_log: Path) -> dict[int, str]:
    """Walk the same log for dictionary_update records and build the
    kernel_id → kernel_name mapping. The batch records carry numeric
    kernel_ids; the names come in separate dictionary_update events
    shipped ahead of the batches that reference them."""
    name_by_id: dict[int, str] = {}
    opener = gzip.open if device_log.suffix == '.gz' else open
    with opener(device_log, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('type') != 'dictionary_update':
                continue
            kdict = rec.get('kernel_dict') or {}
            for k, v in kdict.items():
                try:
                    name_by_id[int(k)] = v
                except (ValueError, TypeError):
                    continue
    return name_by_id


def summarize(device_log: Path, steps: int, top: int):
    name_by_id = lookup_kernel_dict(device_log)
    durations_by_name: dict[str, list[int]] = defaultdict(list)
    total_dur_ns = 0
    total_launches = 0

    for name_id, dur_ns in parse_kernel_durations(device_log):
        if dur_ns is None:
            continue
        name = name_by_id.get(name_id, f"kernel#{name_id}") if name_id is not None else "unknown"
        durations_by_name[name].append(int(dur_ns))
        total_dur_ns += int(dur_ns)
        total_launches += 1

    if total_launches == 0:
        print("[!] No kernel_event_batch rows found in", device_log)
        return

    distinct = len(durations_by_name)
    avg_us = (total_dur_ns / total_launches) / 1000.0
    per_step = total_launches / steps if steps > 0 else 0
    total_ms = total_dur_ns / 1e6

    print("=" * 70)
    print("PyTorch MiniGPT - kernel-level ground truth via gpufl logs")
    print("=" * 70)
    print(f"  Source: {device_log}")
    print(f"  Steps measured:                {steps}")
    print(f"  Distinct CUDA kernel symbols:  {distinct}")
    print(f"  Total kernel LAUNCHES:         {total_launches}")
    print(f"  Launches per step:             {per_step:.1f}")
    print(f"  Total CUDA time:               {total_ms:.1f} ms")
    print(f"  Average per-kernel duration:   {avg_us:.2f} us")
    print()
    print("Hypothesis check:")
    if avg_us < 3.0:
        verdict_dur = "confirms 'short kernel' hypothesis (avg < 3us)"
    elif avg_us < 8.0:
        verdict_dur = "partial - kernels are short-ish (avg 3-8us) but not as tiny as predicted"
    else:
        verdict_dur = f"refutes short-kernel hypothesis (avg = {avg_us:.1f}us)"
    print(f"  Per-kernel duration:           {verdict_dur}")
    if per_step > 100:
        verdict_mult = (f"confirms 'high actual launch count' "
                        f"({per_step:.0f}/step vs ~20 visible ops)")
    elif per_step > 50:
        verdict_mult = f"partial - {per_step:.0f} launches/step"
    else:
        verdict_mult = f"unexpected - only {per_step:.0f} launches/step"
    print(f"  Launches per step:             {verdict_mult}")
    print()

    # Top-N kernels by total time
    rows = []
    for name, durs in durations_by_name.items():
        total = sum(durs)
        count = len(durs)
        avg = total / count
        rows.append((name, count, total, avg))
    rows.sort(key=lambda r: -r[2])

    print(f"Top {top} kernel symbols by total CUDA time:")
    print(f"{'Kernel':<55} {'Count':>8} {'Total (us)':>12} {'Avg (us)':>10}")
    print("-" * 88)
    for name, count, total_ns, avg_ns in rows[:top]:
        short = name if len(name) <= 53 else (name[:50] + "...")
        print(f"{short:<55} {count:>8} {total_ns / 1000.0:>12.1f} {avg_ns / 1000.0:>10.2f}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20,
                   help='measured steps (matches pytorch_train.py default)')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seq', type=int, default=128)
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--log-dir', type=str, default=None,
                   help='where gpufl writes its NDJSON logs. Default: a '
                        'temp dir under the system temp, kept after run for inspection.')
    p.add_argument('--keep-logs', action='store_true', default=True,
                   help='keep the gpufl log dir after analysis (default: kept).')
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available - this script needs a GPU.")
        sys.exit(1)

    log_dir = Path(args.log_dir) if args.log_dir else Path(tempfile.mkdtemp(prefix='gpufl_minigpt_'))

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Running MiniGPT {args.steps} steps under gpufl monitoring "
          f"(batch={args.batch}, seq={args.seq}, warmup={args.warmup})...")
    print(f"Logs: {log_dir}")
    print()

    if not run_minigpt_under_gpufl(args.steps, args.warmup, args.batch, args.seq, log_dir):
        sys.exit(1)

    device_log = find_device_log(log_dir)
    if device_log is None:
        print(f"[!] No device.log found under {log_dir}. gpufl may have failed to write.")
        sys.exit(1)

    summarize(device_log, args.steps, args.top)

    if not args.keep_logs and args.log_dir is None:
        shutil.rmtree(log_dir, ignore_errors=True)
    else:
        print(f"\n(log dir retained at {log_dir} - pass --no-keep-logs to clean up)")


if __name__ == '__main__':
    main()
