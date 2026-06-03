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

## Observed Results

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
