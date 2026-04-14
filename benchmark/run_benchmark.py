#!/usr/bin/env python3
"""
GPUFlight Overhead Benchmark
=============================
Measures profiling overhead across all configurations vs. baseline.

Usage:
  python run_benchmark.py                    # Run all benchmarks
  python run_benchmark.py --gemm-only        # GEMM only
  python run_benchmark.py --pytorch-only     # PyTorch only
  python run_benchmark.py --runs 5           # 5 runs per config (default 3)
  python run_benchmark.py --csv results.csv  # Save CSV output
"""

import argparse
import os
import sys
import shutil
import tempfile
import statistics
import time
import gc

# ── Configurations ────────────────────────────────────────────────────────────

CONFIGS = [
    {
        'name': 'Baseline (no gpufl)',
        'use_gpufl': False,
        'engine': None,
    },
    {
        'name': 'Monitoring only',
        'use_gpufl': True,
        'engine': 'None_',
        'desc': 'System metrics sampling, no CUPTI',
    },
    {
        'name': 'PC Sampling',
        'use_gpufl': True,
        'engine': 'PcSampling',
        'desc': 'Stall reason sampling',
    },
    {
        'name': 'SASS Metrics',
        'use_gpufl': True,
        'engine': 'SassMetrics',
        'desc': 'Instruction-level counters',
    },
    {
        'name': 'PcSampling + SASS',
        'use_gpufl': True,
        'engine': 'PcSamplingWithSass',
        'desc': 'Full profiling (default)',
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_gpufl(config: dict, log_dir: str) -> bool:
    """Initialize GPUFlight with the given config. Returns True if successful."""
    if not config['use_gpufl']:
        return True

    try:
        import gpufl
        engine_name = config['engine']
        # C++ binding uses 'None', Python stub uses 'None_'
        engine = getattr(gpufl.ProfilingEngine, engine_name, None)
        if engine is None and engine_name == 'None_':
            engine = getattr(gpufl.ProfilingEngine, 'None')
        log_path = os.path.join(log_dir, 'bench')
        result = gpufl.init(
            app_name='benchmark',
            log_path=log_path,
            sampling_auto_start=True,
            system_sample_rate_ms=100,
            profiling_engine=engine,
        )
        return result is not False
    except Exception as e:
        print(f"  [WARN] Failed to init gpufl: {e}", file=sys.stderr)
        return False


def teardown_gpufl(config: dict):
    """Shutdown GPUFlight and clean up."""
    if not config['use_gpufl']:
        return
    try:
        import gpufl
        gpufl.shutdown()
    except Exception:
        pass


def run_single(workload_fn, config: dict, **kwargs) -> tuple[float, float]:
    """Run a single benchmark iteration with the given config."""
    log_dir = tempfile.mkdtemp(prefix='gpufl_bench_')
    try:
        setup_gpufl(config, log_dir)
        wall_ms, gpu_ms = workload_fn(**kwargs)
        teardown_gpufl(config)
        return wall_ms, gpu_ms
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_config(workload_fn, config: dict, runs: int, **kwargs) -> dict:
    """Run a config multiple times and return median results."""
    walls = []
    gpus = []
    for r in range(runs):
        wall_ms, gpu_ms = run_single(workload_fn, config, **kwargs)
        walls.append(wall_ms)
        gpus.append(gpu_ms)
        # Brief pause between runs to let GPU cool
        time.sleep(0.5)

    return {
        'name': config['name'],
        'wall_ms': statistics.median(walls),
        'gpu_ms': statistics.median(gpus),
        'walls': walls,
        'gpus': gpus,
    }


def print_table(title: str, results: list[dict]):
    """Print a formatted comparison table."""
    baseline_wall = results[0]['wall_ms'] if results else 1

    print(f"\n{title}")
    print(f"  {'Mode':<25} {'Wall (ms)':>10} {'GPU (ms)':>10} {'Overhead':>10}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 10} {'─' * 10}")

    for r in results:
        overhead = ((r['wall_ms'] - baseline_wall) / baseline_wall) * 100 if baseline_wall > 0 else 0
        overhead_str = '—' if r['name'].startswith('Baseline') else f"+{overhead:.1f}%"
        print(f"  {r['name']:<25} {r['wall_ms']:>10.1f} {r['gpu_ms']:>10.1f} {overhead_str:>10}")

    # Also print individual run details
    print(f"\n  Individual runs (wall ms):")
    for r in results:
        runs_str = ', '.join(f'{w:.1f}' for w in r['walls'])
        print(f"    {r['name']:<25} [{runs_str}]")


def save_csv(filepath: str, gemm_results: list[dict], pytorch_results: list[dict]):
    """Save results to CSV."""
    with open(filepath, 'w') as f:
        f.write('workload,mode,wall_ms,gpu_ms,overhead_pct\n')
        for label, results in [('gemm', gemm_results), ('pytorch', pytorch_results)]:
            if not results:
                continue
            baseline = results[0]['wall_ms']
            for r in results:
                overhead = ((r['wall_ms'] - baseline) / baseline) * 100 if baseline > 0 else 0
                f.write(f"{label},{r['name']},{r['wall_ms']:.1f},{r['gpu_ms']:.1f},{overhead:.2f}\n")
    print(f"\nResults saved to: {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GPUFlight Overhead Benchmark')
    parser.add_argument('--gemm-only', action='store_true', help='Run only GEMM benchmark')
    parser.add_argument('--pytorch-only', action='store_true', help='Run only PyTorch benchmark')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per config (default: 3)')
    parser.add_argument('--gemm-n', type=int, default=2048, help='Matrix size for GEMM (default: 2048)')
    parser.add_argument('--gemm-iters', type=int, default=100, help='GEMM iterations (default: 100)')
    parser.add_argument('--pytorch-steps', type=int, default=20, help='Training steps (default: 20)')
    parser.add_argument('--csv', type=str, default=None, help='Save results to CSV file')
    args = parser.parse_args()

    run_gemm = not args.pytorch_only
    run_pytorch = not args.gemm_only

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: No CUDA GPU available.", file=sys.stderr)
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GiB)")
    except ImportError:
        print("ERROR: PyTorch not installed.", file=sys.stderr)
        sys.exit(1)

    # Check gpufl availability
    gpufl_available = True
    try:
        import gpufl
        print(f"GPUFlight: available (v{getattr(gpufl, '__version__', '?')})")
    except ImportError:
        gpufl_available = False
        print("GPUFlight: NOT available — will only run baseline")

    print(f"Runs per config: {args.runs}")
    print(f"{'=' * 60}")

    gemm_results = []
    pytorch_results = []

    # Filter configs if gpufl not available
    configs = CONFIGS if gpufl_available else [CONFIGS[0]]

    # ── GEMM Benchmark ────────────────────────────────────────────────────
    if run_gemm:
        from cuda_gemm import run_gemm as gemm_fn

        print(f"\n--- CUDA GEMM Benchmark (N={args.gemm_n}, {args.gemm_iters} iters) ---")
        for i, config in enumerate(configs):
            print(f"  [{i+1}/{len(configs)}] {config['name']}...", end=' ', flush=True)
            result = run_config(gemm_fn, config, args.runs,
                                n=args.gemm_n, iterations=args.gemm_iters)
            gemm_results.append(result)
            print(f"wall={result['wall_ms']:.1f}ms, gpu={result['gpu_ms']:.1f}ms")

        print_table(f"CUDA GEMM (N={args.gemm_n}, {args.gemm_iters} iters):", gemm_results)

    # ── PyTorch Benchmark ─────────────────────────────────────────────────
    if run_pytorch:
        from pytorch_train import run_training

        print(f"\n--- PyTorch MiniGPT Benchmark ({args.pytorch_steps} steps) ---")
        for i, config in enumerate(configs):
            print(f"  [{i+1}/{len(configs)}] {config['name']}...", end=' ', flush=True)
            use_scope = config['use_gpufl']
            result = run_config(run_training, config, args.runs,
                                steps=args.pytorch_steps, use_scope=use_scope)
            pytorch_results.append(result)
            print(f"wall={result['wall_ms']:.1f}ms, gpu={result['gpu_ms']:.1f}ms")

        print_table(f"PyTorch MiniGPT ({args.pytorch_steps} steps):", pytorch_results)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    if gemm_results and len(gemm_results) > 1:
        worst = max(gemm_results[1:], key=lambda r: r['wall_ms'])
        overhead = ((worst['wall_ms'] - gemm_results[0]['wall_ms']) / gemm_results[0]['wall_ms']) * 100
        print(f"  GEMM worst-case overhead: {overhead:.1f}% ({worst['name']})")
    if pytorch_results and len(pytorch_results) > 1:
        worst = max(pytorch_results[1:], key=lambda r: r['wall_ms'])
        overhead = ((worst['wall_ms'] - pytorch_results[0]['wall_ms']) / pytorch_results[0]['wall_ms']) * 100
        print(f"  PyTorch worst-case overhead: {overhead:.1f}% ({worst['name']})")

    # Save CSV
    if args.csv:
        save_csv(args.csv, gemm_results, pytorch_results)


if __name__ == '__main__':
    main()
