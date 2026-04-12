# GPUFlight Overhead Benchmark

Measures profiling overhead across all GPUFlight configurations vs. baseline (no profiling).

## Quick Start

```bash
cd benchmark
python run_benchmark.py
```

## Options

```
--gemm-only        Run only CUDA GEMM benchmark
--pytorch-only     Run only PyTorch training benchmark
--runs N           Number of runs per config (default: 3)
--gemm-n N         Matrix size for GEMM (default: 2048)
--gemm-iters N     GEMM iterations (default: 100)
--pytorch-steps N  Training steps (default: 20)
--csv FILE         Save results to CSV
```

## Configurations Tested

| Mode | Engine | What it measures |
|------|--------|-----------------|
| Baseline | — | Pure CUDA/PyTorch, no GPUFlight |
| Monitoring only | `None_` | System metrics (GPU util, temp, mem) only |
| PC Sampling | `PcSampling` | Stall reason sampling |
| SASS Metrics | `SassMetrics` | Instruction-level counters |
| PcSampling + SASS | `PcSamplingWithSass` | Full profiling (default) |

## Workloads

- **GEMM**: 2048×2048 matrix multiply via cuBLAS (100 iterations). Raw compute, most sensitive to CUPTI overhead.
- **PyTorch MiniGPT**: 6-layer transformer training (20 steps, FP16). Real-world ML workload with diverse kernels.

## Benchmark Results

Tested on NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GiB), GPUFlight v0.1.0.dev.

### CUDA GEMM (N=2048, 100 iterations)

| Mode | Wall (ms) | GPU (ms) | Overhead |
|------|-----------|----------|----------|
| Baseline (no gpufl) | 184.3 | 184.2 | — |
| Monitoring only | 179.8 | 179.7 | ~0% |
| PC Sampling | 230.2 | 230.0 | +24.9% |
| SASS Metrics | 225.5 | 225.4 | +22.3% |
| PcSampling + SASS | 284.1 | 284.0 | +54.1% |

### PyTorch MiniGPT Training (20 steps, FP16)

| Mode | Wall (ms) | GPU (ms) | Overhead |
|------|-----------|----------|----------|
| Baseline (no gpufl) | 795.8 | 795.7 | — |
| Monitoring only | 852.4 | 852.3 | +7.1% |
| PC Sampling | 6085.6 | 6085.5 | +664.7% |
| SASS Metrics | 848.3 | 848.3 | +6.6% |
| PcSampling + SASS | 6011.2 | 6011.1 | +655.4% |

### Key Takeaways

- **Monitoring only / SASS Metrics: ~7% overhead** — acceptable for always-on profiling during development
- **PC Sampling: high overhead on many-small-kernel workloads** (PyTorch launches thousands of micro-kernels). Use for targeted profiling sessions, not always-on. Set via environment variable:
  ```bash
  GPUFL_PROFILING_ENGINE=SassMetrics python train.py  # lightweight
  GPUFL_PROFILING_ENGINE=PcSamplingWithSass python train.py  # full (default)
  ```
- **GEMM (large kernels):** overhead is moderate even with full profiling, because per-kernel CUPTI cost is amortized over longer kernel execution
