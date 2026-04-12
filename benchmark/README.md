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

## Example Output

```
GPU: NVIDIA GeForce RTX 4090 (24.0 GiB)
GPUFlight: available (v0.1.0.dev)

CUDA GEMM (N=2048, 100 iters):
  Mode                       Wall (ms)   GPU (ms)   Overhead
  Baseline (no gpufl)           1234.5     1200.0          —
  Monitoring only               1236.1     1200.2      +0.1%
  PC Sampling                   1245.8     1201.5      +0.9%
  SASS Metrics                  1260.2     1205.3      +2.1%
  PcSampling + SASS             1268.4     1208.1      +2.7%
```
