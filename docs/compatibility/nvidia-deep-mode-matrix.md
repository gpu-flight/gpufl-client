# NVIDIA Deep Mode Compatibility Matrix

This document records NVIDIA Deep mode test results, the hardware/software
environment used for each run, and the feature set each Deep mode combination is
expected to collect.

Deep mode is a decision pipeline, not a fixed "enable every CUPTI feature"
switch:

1. Deep decides whether to attempt SASS metrics.
2. If SASS does not arm, Deep falls back to PC sampling.
3. If SASS arms, Deep applies the SASS activity policy.
4. The product default is SASS-safe activity, not the legacy full CUPTI activity
   bundle.

The purpose of this document is to keep the runtime policy tied to observed
device behavior.

## Current Policy

| Policy | Default | Reason |
| --- | --- | --- |
| Deep attempts SASS first | Yes | Deep's most differentiated value is per-instruction SASS metrics and disassembly. |
| Deep falls back to PC sampling | Yes | If SASS is disabled, excluded, unavailable, or fails to arm, Deep should still produce stall/hot-PC data. |
| SASS-safe activity policy | Yes | SASS plus the full CUPTI activity bundle reproduced a scoped PyTorch hang on Windows/CUDA 13.2. |
| Full activity bundle | Opt-in only | Available for diagnostics through `GPUFL_SASS_ALLOW_FULL_ACTIVITY=1`, but not safe as a default. |
| PM Sampling in Deep | Yes when built with PerfWorks | Deep starts PM Sampling as a side-channel; successful rows still require NVIDIA performance-counter access. |
| SASS + PC sampling together | Experimental only | Controlled by `GPUFL_DEEP_TRY_BOTH=1`; current assumption is mutual exclusion. |

## Test Environment: 2026-06-02 Windows PyTorch Run

| Field | Value |
| --- | --- |
| Date | 2026-06-02 |
| OS | Microsoft Windows 11, NT version 10.0.26200.8457 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Compute capability | 12.0 |
| NVIDIA driver | 592.01 |
| CUDA Toolkit | 13.2, `nvcc` 13.2.51 |
| Toolkit CUPTI DLL | `cupti64_2026.1.0.dll` |
| PyTorch bundled CUPTI DLL | `cupti64_2026.1.1.dll` |
| Python | 3.13.5, 64-bit |
| PyTorch | 2.12.0+cu132 |
| PyTorch CUDA | 13.2 |
| Client package | `gpufl-1.1.0rc2-cp313-cp313-win_amd64.whl` |
| Workload | `C1M2_Assignment.py`, EMNIST one-epoch training workload |
| Init options | `ProfilingEngine.Deep`, `enable_stack_trace=True`, `enable_memory_tracking=True`, `enable_cuda_graphs_tracking=False`, `enable_debug_output=False` |
| Scope setup | One outer `gpufl.Scope("train_epoch")` unless otherwise noted |
| Timeout threshold | 240 seconds |

## Observed Results: Windows RTX 5060 PyTorch

| Run | Scope | Flags / policy | Result | Marker progress | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `baseline_outer_scope_20260602_145825` | Outer `train_epoch` | Pre-patch default full SASS activity | Timeout | Not instrumented | Reproduced hang with Deep + SASS + full activity bundle. |
| `defer_scope_flush_20260602_150305` | Outer `train_epoch` | `GPUFL_SASS_DEFER_SCOPE_FLUSH=1` | Timeout | Not instrumented | Deferring scope-stop SASS flush did not fix the hang. |
| `defer_no_cubin_capture_20260602_150815` | Outer `train_epoch` | `GPUFL_SASS_DEFER_SCOPE_FLUSH=1`, `GPUFL_SASS_DISABLE_CUBIN_CAPTURE=1` | Timeout | Not instrumented | CUBIN capture alone was not the root cause. |
| `instrumented_outer_scope_20260602_151945` | Outer `train_epoch` | Pre-patch default full SASS activity | Timeout | Reached `entered train_epoch scope`; did not return from `train_epoch` | Hang occurs inside scoped training after SASS is armed, not only at shutdown or scope exit. |
| `instrumented_no_scope_20260602_152434` | No scope | Pre-patch default full SASS activity | Exit | Reached `after gpufl.shutdown` | Removing the scope avoided SASS arming during the training loop, so the workload completed. |
| `instrumented_outer_sass_metrics_only_20260602_153557` | Outer `train_epoch` | `GPUFL_SASS_METRICS_ONLY=1` | Exit | Reached `after gpufl.shutdown` | SASS itself can run. The failing condition is SASS combined with other CUPTI activity/callback layers. |
| `instrumented_outer_sass_safe_activity_20260602_153829` | Outer `train_epoch` | `GPUFL_SASS_FORCE_SAFE_ACTIVITY=1` | Exit | Reached `after gpufl.shutdown` | Safe SASS activity policy avoids the hang. |
| `instrumented_outer_default_after_safe_patch_20260602_154729` | Outer `train_epoch` | Post-patch default safe SASS activity | Exit | Reached `after gpufl.shutdown` | The new default policy preserved Deep/SASS while avoiding the reproduced hang. |

`GPUFL_SASS_ALLOW_KERNEL_ACTIVITY=1` was not tested in this run because the
approval prompt for that diagnostic run was declined. The current evidence
therefore proves the unsafe bundle-level condition, not the exact individual
CUPTI activity kind that triggers the hang.

## PM Sampling Smoke Test: 2026-06-02 Windows RTX 5060 PyTorch

| Field | Value |
| --- | --- |
| Date | 2026-06-02 |
| OS | Microsoft Windows 11, NT version 10.0.26200.8457 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Compute capability | 12.0 |
| NVIDIA driver | 592.01 |
| CUDA Toolkit | 13.2, `nvcc` 13.2.51 |
| Toolkit CUPTI DLL | `cupti64_2026.1.0.dll` |
| PyTorch bundled CUPTI DLL | `cupti64_2026.1.1.dll` |
| Python | 3.13.5, 64-bit |
| PyTorch | 2.12.0+cu132 |
| PyTorch CUDA | 13.2 |
| Client build | `main` at `708ceab`; Windows wheel built with `GPUFL_HAS_PERFWORKS=1` and linked against `cupti.lib`, `nvperf_host.lib`, and `nvperf_target.lib` from CUDA 13.2 |
| Workload | PyTorch FP32 `4096 x 4096` matmul plus ReLU, 16 iterations inside `gpufl.Scope("pm_sampling_matmul")` |
| Init options | `ProfilingEngine.PmSampling`, `pm_sampling_interval_us=1000`, `pm_sampling_max_samples=4096`, `pm_sampling_metrics=["sm__warps_launched.sum"]`, `pm_sampling_scope_only=True`, `enable_debug_output=True`, `enable_stack_trace=False`, `enable_memory_tracking=False`, `continuous_system_sampling=False` |
| Log path | `C:\Temp\gpufl-pm-smoke-20260602_212250` |
| Generated report | `C:\Temp\gpufl-pm-smoke-20260602_212250-report.txt` |
| Timeout threshold | 180 seconds |

## Observed Results: Windows RTX 5060 PM Sampling Smoke

| Run | Result | PM config rows | PM sample rows | Report status | Diagnostic | Interpretation |
| --- | --- | ---: | ---: | --- | --- | --- |
| `pm_sampling_smoke_20260602_212250` | Exit | 1 | 0 | `Pm Sampling: on, no data`; `enabled_but_no_samples` | `cuptiProfilerGetCounterAvailability(data)` failed with `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES (35)` | The PM Sampling build/API/report path works, but this non-elevated Windows environment cannot read NVIDIA performance counters yet. Enable local NVIDIA performance-counter access or rerun elevated, then retest sample collection. |

Collected data from `pm_sampling_smoke_20260602_212250`:

| Feature | Result |
| --- | --- |
| Build | Succeeded; the wheel was built with `GPUFL_HAS_PERFWORKS=1`. |
| Python API | `gpufl.ProfilingEngine.PmSampling` was available and `gpufl.init(...)` returned `True`. |
| Kernel events | Collected; 34 kernel rows across 3 unique kernels. |
| Scope rows | Collected for `pm_sampling_matmul`. |
| PM Sampling config | Collected; one `pm_sampling_config` row was emitted for `sm__warps_launched.sum`. |
| PM Sampling samples | Not collected in this run because CUPTI performance-counter access was denied by the driver (`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`). |
| Text report | Generated successfully; the Capture Capabilities section reported `Requested Engine: nvidia.pm_sampling`, `Selected Engine: nvidia.pm_sampling`, and `Pm Sampling: on, no data`. |

This run validates PM Sampling packaging and report plumbing on Windows/CUDA 13.2.
It does not validate successful hardware-counter sample collection on this
machine account because NVIDIA performance-counter privileges blocked the CUPTI
counter-availability query before `cuptiPmSamplingStart` could arm.

## Test Environment: 2026-06-02 Linux RTX 3090 PyTorch Run

| Field | Value |
| --- | --- |
| Date | 2026-06-02 |
| OS | Ubuntu 24.04.1, Linux kernel 6.17.0-29-generic |
| GPU | NVIDIA GeForce RTX 3090 |
| Compute capability | 8.6 |
| NVIDIA driver | 590.48.01 |
| CUDA Toolkit | 13.1, `nvcc` 13.1.115 |
| Toolkit CUPTI library | `/usr/local/cuda-13.1/lib64/libcupti.so.13` |
| CUPTI version reported by gpufl | 13.1.1 (`130101`) |
| Python | 3.13.11 |
| PyTorch | 2.12.0+cu132 |
| PyTorch CUDA | 13.2 |
| Client build | Local `features/pm_sampling` branch rebased on `origin/main` at `f4462ad`, commit `6d55a21` |
| Workload | PyTorch EMNIST MLP derived from `/home/myounghoshin/PyCharmMiscProject/C1M2_Assignment.py` |
| Init options | `ProfilingEngine.Deep`, `enable_stack_trace=True`, `enable_memory_tracking=True`, `enable_cuda_graphs_tracking=False`, `enable_debug_output=False` |
| Scope setup | One outer `gpufl.Scope("train_epoch")` for full-epoch attempts; `warmup` plus `train_subset_2048` for completed smoke run |
| Timeout threshold | 240 seconds for full epoch; 180 seconds for subset smoke |

## Observed Results: Linux RTX 3090 PyTorch

| Run | Scope | Flags / policy | Result | Marker progress | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `pytorch_full_epoch_default_safe_20260602` | Outer `train_epoch` | Post-patch default safe SASS activity; PM Sampling enabled by Deep | Timeout | Scope log recorded `train_epoch` start; no scope stop, shutdown, kernel, SASS, or PM sample rows were flushed before timeout | The full EMNIST epoch did not complete within 240 seconds, so it is not a clean compatibility pass/fail. |
| `pytorch_full_epoch_pc_only_20260602` | Outer `train_epoch` | `GPUFL_DEEP_PC_ONLY=1` | Timeout | Reached `entered train_epoch scope`; no scope stop, shutdown, kernel, PC, or PM sample rows were flushed before timeout | PC-only also exceeded 240 seconds on the same full-epoch workload, which points to workload/runtime duration as a confounder rather than SASS alone. |
| `pytorch_subset_2048_default_safe_20260602` | `warmup`, `train_subset_2048` | Post-patch default safe SASS activity; no override flags | Exit | Reached `marker after shutdown` and emitted text report | Default safe Deep completed on sm_86 PyTorch. SASS won, PC sampling was skipped as mutually exclusive, PM Sampling also collected, and memory allocation tracking produced rows. |
| `pytorch_subset_2048_kernel_activity_20260602` | `warmup`, `train_subset_2048` | `GPUFL_SASS_ALLOW_KERNEL_ACTIVITY=1` | Timeout | Reached `marker before warmup`; `scope.log` remained empty and no shutdown/report was emitted | Enabling CUPTI kernel activity alongside SASS reproduced a hang on Linux sm_86 PyTorch subset. Kernel activity should remain opt-in and should not be selected automatically for this arch/OS/workload class. |

Collected data from `pytorch_subset_2048_default_safe_20260602`:

| Feature | Result |
| --- | --- |
| Selected engine | `nvidia.sass_metrics` |
| Kernel events | Fallback/partial; CUPTI kernel activity disabled by safe SASS policy, so the report's kernel execution tables had no kernel rows |
| Scope rows | 4 rows: start/stop for `warmup` and `train_subset_2048` |
| SASS metrics | Collected; 37,364 `profile_sample_batch` rows and per-function SASS efficiency in the report |
| PC sampling | Skipped as mutually exclusive with SASS metrics |
| PM Sampling | Collected; 1,570 `pm_sample_batch` rows for the overview preset |
| Memory activity | Collected; 1,023 memory allocation rows |
| CUBIN disassembly | Collected |
| External correlation | Skipped by safe SASS policy |
| Diagnostics | One second-scope SASS re-arm message was printed: `cuptiSassMetricsEnable failed: CUPTI_ERROR_INVALID_PARAMETER`; the run still completed and emitted SASS rows. PM decode also printed one `CUPTI_ERROR_UNKNOWN` message while still reporting PM Sampling as collected. |

`pytorch_subset_2048_kernel_activity_20260602` used the same subset workload and
only added `GPUFL_SASS_ALLOW_KERNEL_ACTIVITY=1`. It timed out after 180 seconds
with an empty `scope.log`, while the default safe run completed in 40.91 seconds.
That makes kernel activity unsafe as an automatic Deep choice for this Linux
sm_86 PyTorch case.

## Supplemental Test Environment: 2026-06-02 Linux RTX 3090 CUDA Demo

| Field | Value |
| --- | --- |
| Date | 2026-06-02 |
| OS | Ubuntu 24.04.1, Linux kernel 6.17.0-29-generic |
| GPU | NVIDIA GeForce RTX 3090 |
| Compute capability | 8.6 |
| NVIDIA driver | 590.48.01 |
| CUDA Toolkit | 13.1, `nvcc` 13.1.115 |
| Toolkit CUPTI library | `/usr/local/cuda-13.1/lib64/libcupti.so.13` |
| CUPTI version reported by gpufl | 13.1.1 (`130101`) |
| Client build | Local `features/pm_sampling` branch rebased on `origin/main` at `f4462ad`, commit `6d55a21` |
| Workload | `example/cuda/sass_divergence_demo`, CUDA divergence demo |
| Init options | `ProfilingEngine.Deep`, default demo options |
| Scope setup | `0_warmup`, `1_uniform_work_warmup`, `1_uniform_work`, `2_branch_by_lane`, `3_branch_by_quad`, `4_early_exit`, `5_indirect_branch` |
| Timeout threshold | 60 seconds |

## Observed Results: Linux RTX 3090 CUDA Demo

| Run | Scope | Flags / policy | Result | Marker progress | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `sass_divergence_default_safe_20260602` | Seven demo scopes | Post-patch default safe SASS activity; no override flags | Exit | Reached `Shutdown complete` and emitted text report | Default safe Deep completed on sm_86. SASS armed and was selected; PC sampling was skipped as mutually exclusive; PM Sampling also collected. |

Collected data from `sass_divergence_default_safe_20260602`:

| Feature | Result |
| --- | --- |
| Selected engine | `nvidia.sass_metrics` |
| Kernel events | Fallback synthetic rows from launch callbacks; CUPTI kernel activity disabled by safe SASS policy |
| Kernel rows | 22 |
| SASS metrics | Collected; per-function warp efficiency and memory efficiency reported |
| PC sampling | Skipped as mutually exclusive with SASS metrics |
| PM Sampling | Collected; 52 `sm__warps_launched.sum` samples |
| CUBIN disassembly | Collected; 5 functions parsed |
| Memory activity | Not requested in this demo |
| External correlation | Skipped by safe SASS policy |

## Deep Mode Feature Collection Semantics

| Combination | Selection | Expected collected data | Expected missing or degraded data | Stability status |
| --- | --- | --- | --- | --- |
| Default Deep, SASS wins | `ProfilingEngine.Deep`, no activity override flags | SASS metrics, CUBIN capture, SASS disassembly, callback-based synthetic kernel rows, launch API timestamps, scope path, launch grid/block parameters when available, host metrics, NVML device metrics, MEMORY2 rows if `enable_memory_tracking=True` and not blocked by policy | CUPTI kernel activity GPU start/end records are skipped; kernel duration and occupancy can be partial; PC sampling is skipped | Recommended default |
| Deep PC-only fallback | `GPUFL_DEEP_PC_ONLY=1`, SASS excluded, or SASS fails to arm | PC samples, stall reasons, hot PCs, source/function correlation when emitted, callback-based kernel rows, launch metadata, host metrics, NVML device metrics | SASS metrics and SASS replay counters are not collected | Recommended fallback |
| SASS metrics only | `GPUFL_SASS_METRICS_ONLY=1` | SASS metric rows and CUBIN/SASS disassembly unless CUBIN capture is disabled | Kernel rows, kernel names/details, memory activity, sync activity, marker activity, graph activity, external correlation, and PC sampling are skipped | Diagnostic safe mode |
| Safe SASS + kernel activity | `GPUFL_SASS_ALLOW_KERNEL_ACTIVITY=1` | Default SASS data plus CUPTI `KERNEL` and `CONCURRENT_KERNEL` activity records, GPU start/end timing, richer kernel names/details | PC sampling remains skipped when SASS wins | Experimental; needs per-workload validation |
| Safe SASS + memcpy activity | `GPUFL_SASS_ALLOW_MEM_TRANSFER_ACTIVITY=1` | Default SASS data plus CUPTI memcpy/memset timing, bytes, and direction | MEMORY2 allocation tracking is disabled by default when mem-transfer activity is explicitly enabled, unless `GPUFL_SASS_ALLOW_MEMORY2_ACTIVITY=1` is also set | Experimental |
| Safe SASS + memory allocation activity | `enable_memory_tracking=True`; optional `GPUFL_SASS_ALLOW_MEMORY2_ACTIVITY=1` | CUPTI `MEMORY2` allocation/free rows with address, size, and memory kind | Memcpy/memset activity remains off unless separately enabled | Default-safe when mem-transfer activity is not enabled |
| Safe SASS + synchronization activity | `GPUFL_SASS_ALLOW_SYNC_ACTIVITY=1`, `enable_synchronization=True` | CUPTI synchronization timing records | Synchronization callback stack attribution remains disabled in SASS profiler mode | Experimental |
| Safe SASS + marker activity | `GPUFL_SASS_ALLOW_MARKER_ACTIVITY=1` | CUPTI marker/NVTX records | Does not enable kernel/memory/sync activity by itself | Experimental |
| Safe SASS + graph activity | `GPUFL_SASS_ALLOW_GRAPH_ACTIVITY=1`, `enable_cuda_graphs_tracking=True` | CUPTI graph launch activity | Per-node graph timing is not implied; kernel activity still requires its own flag | Experimental |
| Safe SASS + external correlation | `GPUFL_SASS_ALLOW_EXTERNAL_CORRELATION=1`, `enable_external_correlation=True` | Framework external correlation IDs when emitted; runtime/driver anchors are enabled internally | No rows if the framework does not emit external IDs | Experimental, potentially high volume |
| Legacy SASS + full activity bundle | `GPUFL_SASS_ALLOW_FULL_ACTIVITY=1` | SASS metrics, CUBIN/SASS disassembly, and all requested CUPTI activity layers | PC sampling is still skipped unless `GPUFL_DEEP_TRY_BOTH=1` is also used | Diagnostic only; reproduced hang risk |
| SASS + PC sampling try-both | `GPUFL_DEEP_TRY_BOTH=1` | If both APIs arm, potentially SASS metrics and PC sampling in one session | Expected to fail or degrade on currently tested drivers because SASS and PC sampling are treated as mutually exclusive | Research-only |
| CUBIN capture disabled | `GPUFL_SASS_DISABLE_CUBIN_CAPTURE=1` or `GPUFL_DISABLE_CUBIN_CAPTURE=1` | Selected engine can still run | Source/SASS disassembly and CUBIN-backed instruction mapping are unavailable or degraded | Diagnostic only |

## Flag Reference

| Flag | Effect |
| --- | --- |
| `GPUFL_DEEP_PC_ONLY=1` | Forces Deep to skip SASS and run PC sampling only. |
| `GPUFL_SASS_EXCLUDE_ARCHS=86,120` | Skips SASS on listed compute capabilities and lets Deep fall back to PC sampling. |
| `GPUFL_SASS_METRICS_ONLY=1` | Runs SASS in isolation by disabling activity/callback tracing around SASS. |
| `GPUFL_SASS_FORCE_SAFE_ACTIVITY=1` | Forces safe SASS activity policy. Equivalent to the current default unless full activity is explicitly allowed. |
| `GPUFL_SASS_ALLOW_FULL_ACTIVITY=1` | Restores legacy full CUPTI activity behavior in SASS profiler mode. Diagnostic only. |
| `GPUFL_SASS_ALLOW_KERNEL_ACTIVITY=1` | Enables CUPTI kernel activity in safe SASS mode. |
| `GPUFL_SASS_ALLOW_MEM_TRANSFER_ACTIVITY=1` | Enables CUPTI memcpy/memset activity in safe SASS mode. |
| `GPUFL_SASS_ALLOW_MEMORY2_ACTIVITY=1` | Explicitly enables CUPTI `MEMORY2` allocation/free activity in safe SASS mode. |
| `GPUFL_SASS_ALLOW_MEMORY_ACTIVITY=1` | Alias accepted by the memory activity policy. |
| `GPUFL_SASS_ALLOW_SYNC_ACTIVITY=1` | Enables CUPTI synchronization activity in safe SASS mode. |
| `GPUFL_SASS_ALLOW_MARKER_ACTIVITY=1` | Enables CUPTI marker activity in safe SASS mode. |
| `GPUFL_SASS_ALLOW_GRAPH_ACTIVITY=1` | Enables CUPTI graph launch activity in safe SASS mode. |
| `GPUFL_SASS_ALLOW_EXTERNAL_CORRELATION=1` | Enables CUPTI external correlation records and runtime/driver anchors in safe SASS mode. |
| `GPUFL_DEEP_TRY_BOTH=1` | Attempts PC sampling after SASS. Research-only. |
| `GPUFL_SASS_DISABLE_CUBIN_CAPTURE=1` | Disables CUBIN capture only in SASS profiler modes. |
| `GPUFL_DISABLE_CUBIN_CAPTURE=1` | Disables CUBIN capture globally. |
| `GPUFL_SASS_DEFER_SCOPE_FLUSH=1` | Diagnostic mode that keeps SASS armed across scope stop and flushes later. |

## Decision Tree

```text
ProfilingEngine.Deep
  |
  +-- GPUFL_DEEP_PC_ONLY=1?
  |     |
  |     +-- yes -> PC sampling only
  |     |
  |     +-- no
  |          |
  |          +-- GPU architecture excluded by GPUFL_SASS_EXCLUDE_ARCHS?
  |          |     |
  |          |     +-- yes -> PC sampling only
  |          |     |
  |          |     +-- no
  |          |          |
  |          |          +-- Attempt SASS
  |          |                |
  |          |                +-- SASS failed to arm -> PC sampling only
  |          |                |
  |          |                +-- SASS armed
  |          |                      |
  |          |                      +-- GPUFL_SASS_METRICS_ONLY=1?
  |          |                      |     |
  |          |                      |     +-- yes -> SASS metrics only
  |          |                      |
  |          |                      +-- GPUFL_SASS_ALLOW_FULL_ACTIVITY=1?
  |          |                      |     |
  |          |                      |     +-- yes -> SASS + full CUPTI activity bundle
  |          |                      |              (diagnostic only; observed hang risk)
  |          |                      |
  |          |                      +-- default -> Safe SASS Deep
  |          |                              |
  |          |                              +-- optional per-kind flags add activity layers:
  |          |                                  GPUFL_SASS_ALLOW_KERNEL_ACTIVITY
  |          |                                  GPUFL_SASS_ALLOW_MEM_TRANSFER_ACTIVITY
  |          |                                  GPUFL_SASS_ALLOW_MEMORY2_ACTIVITY
  |          |                                  GPUFL_SASS_ALLOW_SYNC_ACTIVITY
  |          |                                  GPUFL_SASS_ALLOW_MARKER_ACTIVITY
  |          |                                  GPUFL_SASS_ALLOW_GRAPH_ACTIVITY
  |          |                                  GPUFL_SASS_ALLOW_EXTERNAL_CORRELATION
```

## How to Add a New Device Result

Add a new environment section and append rows to the observed results table.
Record at least:

| Required field | Example |
| --- | --- |
| Date | `2026-06-02` |
| GPU | `NVIDIA GeForce RTX 5060 Laptop GPU` |
| Compute capability | `12.0` |
| Driver | `592.01` |
| CUDA Toolkit | `13.2` |
| Toolkit CUPTI DLL | `cupti64_2026.1.0.dll` |
| Framework | `PyTorch 2.12.0+cu132` |
| Workload | `C1M2_Assignment.py`, one epoch |
| Scope setup | Outer `train_epoch`, nested scopes, or no scope |
| Flags | Exact `GPUFL_*` environment variables |
| Result | `Exit`, `Timeout`, crash, or degraded output |
| Collected data | Which capture capabilities produced rows |
| Decision impact | Whether the result changes the default policy |

## Code References

| File | Role |
| --- | --- |
| `include/gpufl/backends/nvidia/cupti_backend.hpp` | SASS safe/full activity policy and CUBIN capture gates. |
| `include/gpufl/backends/nvidia/cupti_backend.cpp` | CUPTI activity enable/disable logic and capture capability reporting. |
| `include/gpufl/backends/nvidia/kernel_launch_handler.cpp` | Launch callbacks, synthetic kernel rows, and CUPTI kernel activity handling. |
| `include/gpufl/backends/nvidia/engine/pc_sampling_with_sass_engine.cpp` | Deep mode selection between SASS and PC sampling. |
| `include/gpufl/backends/nvidia/engine/sass_metrics_engine.cpp` | SASS arm/flush/disable behavior and diagnostic scope flush controls. |
