# AMD / ROCm Examples

This folder mirrors the CUDA example area with runnable HIP examples for AMD GPUs.

## What Works Today

- ROCm / AMD system telemetry via `rocm_smi`
- AMD static device inventory via HIP
- AMD kernel dispatch tracing via `rocprofiler-sdk`
- AMD memcpy tracing via `rocprofiler-sdk`
- AMD memory-allocation tracing via `rocprofiler-sdk`
- AMD synchronization tracing via filtered ROCprofiler HIP runtime API records
- Per-dispatch AMD hardware counters via ROCprofiler dispatch counting
- Device-wide `PmSampling` timelines via ROCprofiler device counting
- `gpufl` initialization with `backend = gpufl::BackendKind::Amd`
- User-defined scope logging via `GFL_SCOPE(...)`
- HIP example programs that run on ROCm hardware

## What Does Not Work Yet

- AMD PC sampling
- Instruction-level SASS metrics and NVIDIA-compatible Range Profiler metrics

Today, the AMD backend is useful for:

- system metric logging
- device inventory
- automatic HIP kernel, memcpy, memory-allocation, and synchronization tracing
- per-dispatch and device-wide hardware-counter profiling
- scope-level application instrumentation

It is not yet useful for:

- PC sampling or instruction-level profiling

## Targets

- `amd_check_device`
  - Basic HIP device detection smoke test
- `amd_vector_add_demo`
  - Profiled HIP vector addition with one scoped kernel, H2D/D2H transfers, and result verification
- `amd_gpufl_scope_demo`
  - Initializes `gpufl` with the AMD backend, runs HIP work inside scopes, and writes logs
- `amd_memory_allocation_rows`
  - Exits successfully only when two HIP allocation/free phases emit memory-allocation rows
- `amd_pm_sampling_sample_rows`
  - Selects AMD device counting and exits successfully only when each of two named scopes emits PM sample rows
- `amd_synchronization_rows`
  - Exits successfully only when two HIP workloads emit the expected synchronization rows

## Build

From the repository root:

```bash
cmake -S . -B build-rocm-examples \
  -DGPUFL_ENABLE_AMD=ON \
  -DGPUFL_ENABLE_NVIDIA=OFF \
  -DBUILD_GPUFL_EXAMPLE=ON \
  -DBUILD_PYTHON=OFF \
  -DBUILD_TESTING=OFF

cmake --build build-rocm-examples --target amd_check_device
cmake --build build-rocm-examples --target amd_vector_add_demo
cmake --build build-rocm-examples --target amd_gpufl_scope_demo
cmake --build build-rocm-examples --target amd_memory_allocation_rows
cmake --build build-rocm-examples --target amd_pm_sampling_sample_rows
cmake --build build-rocm-examples --target amd_synchronization_rows
```

The AMD example targets are only added when CMake detects HIP successfully.
`GPUFL_ENABLE_AMD=ON` by itself is not enough.

If configure succeeds but `cmake --build ... --target amd_check_device` says the
target does not exist, inspect the configure output and make sure you see:

```text
-- Found HIP host runtime support
-- Found HIP: /opt/rocm ...
```

If `rocprofiler-sdk` is available, configure output should also include:

```text
-- Found ROCprofiler-SDK support
```

If HIP is installed in a non-default location, pass it explicitly:

```bash
cmake -S . -B build-rocm-examples \
  -DROCM_PATH=/path/to/rocm \
  -DHIP_PATH=/path/to/rocm \
  -DGPUFL_ENABLE_AMD=ON \
  -DGPUFL_ENABLE_NVIDIA=OFF \
  -DBUILD_GPUFL_EXAMPLE=ON \
  -DBUILD_PYTHON=OFF \
  -DBUILD_TESTING=OFF
```

If you want to open only `example/amd` in CLion, that folder now supports
top-level CMake configure as well. In that mode, CLion should point at:

```text
/path/to/repo/example/amd
```

and use cache variables such as:

```text
-DROCM_PATH=/opt/rocm
-DHIP_PATH=/opt/rocm
```

The standalone `example/amd` configure internally adds the repository root as a
subproject and disables the parent example/test targets to avoid recursion.

## Run

```bash
./build-rocm-examples/example/amd/amd_check_device
./build-rocm-examples/example/amd/amd_vector_add_demo
./build-rocm-examples/example/amd/amd_gpufl_scope_demo
./build-rocm-examples/example/amd/amd_memory_allocation_rows
./build-rocm-examples/example/amd/amd_pm_sampling_sample_rows
./build-rocm-examples/example/amd/amd_synchronization_rows
```

`amd_vector_add_demo` replaces the CPU-versus-GPU benchmark. It initializes
GPUFlight in Trace mode, copies two 4 MiB input vectors to the GPU, launches
`vectorAdd` inside `vector-addition-scope`, and copies the result back. It
checks every output element and prints a session report; there are no CPU
timings or speedup comparisons. Allocations and synchronization are also
traced. A short capture may not contain periodic system metric samples.

To include the demo source when running outside the repository (for example,
from an IDE build directory), set the approved source root explicitly:

```bash
GPUFL_SOURCE_ROOT="$PWD/example/amd" \
  ./build-rocm-examples/example/amd/amd_vector_add_demo
```

Run that command from the repository root. Source capture remains restricted
to the approved directory; unrelated files and system headers are not included.

`amd_memory_allocation_rows` enables `enable_memory_tracking`, runs two
allocation/free phases, and checks that every expected allocate and free
operation produces a `memory_alloc_event_batch` row. It returns exit code 2
when ROCprofiler allocation tracing is unavailable or rows are missing.

`amd_synchronization_rows` issues event synchronize, stream wait-event,
stream synchronize, and device synchronize calls in each phase, then checks
that the synchronization row count increases by the expected amount. It
returns exit code 2 when ROCprofiler HIP runtime tracing is unavailable or
rows are missing.

`amd_pm_sampling_sample_rows` requests the portable `GPUBusy` counter, runs
GPU work in `pm_rows_phase_a` and `pm_rows_phase_b`, and checks that the PM row
count increases after each scope. It returns exit code 2 when AMD device
counting is unavailable or either scope does not produce a row. Its generated
report shows the same rows grouped by scope for manual inspection.

The scope demo selects per-dispatch counters by default. To exercise the
device-wide PM timeline instead:

```bash
GPUFL_PROFILING_ENGINE=PmSampling \
  ./build-rocm-examples/example/amd/amd_gpufl_scope_demo
```

`PmSampling` uses the portable `GPUBusy` derived counter by default. Set
`pm_sampling_metrics` programmatically to request other native ROCprofiler
counter names.

On a working ROCm system, `amd_check_device` should print output similar to:

```text
Found 2 HIP devices.
Success! Device 0: AMD Radeon RX 9070 XT (arch gfx1201, capability 12.0)
```

## Logs

`amd_vector_add_demo` writes logs with prefix:

```text
gfl_amd_vector_add
```

`amd_gpufl_scope_demo` writes logs with prefix:

```bash
gfl_amd_scope
```

`amd_memory_allocation_rows` writes logs with prefix:

```bash
gfl_amd_memory_rows
```

`amd_pm_sampling_sample_rows` writes logs with prefix:

```bash
gfl_amd_pm_rows
```

`amd_synchronization_rows` writes logs with prefix:

```bash
gfl_amd_sync_rows
```

Check a completed example capture before uploading it:

```bash
python3 example/amd/verify_trace_timestamps.py gfl_amd_pm_rows/<session-id>
```

This checks that kernel, copy, allocation, synchronization, scope, and PM rows
use the session epoch clock. Use the corresponding log directory for other
examples. Static ISA mappings are checked for duplicate delivery, not timing.
Run `python3 example/amd/test_verify_trace_timestamps.py` for the validator tests. Old
captures recorded with profiler-relative timestamps must be regenerated;
re-uploading those same files will not repair their clock.

With `rocprofiler-sdk` available, expect:

- `job_start` inventory
- kernel dictionaries
- `kernel_event_batch`
- `kernel_detail`
- `memcpy_event_batch`
- `memory_alloc_event_batch` when `enable_memory_tracking` is enabled
- `synchronization_event_batch` when `enable_synchronization` is enabled
- `profile_sample_batch` for dispatch-counting requests
- `pm_sampling_config` and `pm_sample_batch` for `PmSampling`
- system metric samples
- scope events

Without `rocprofiler-sdk`, expect only telemetry, static inventory, and scope
events.

PC samples and instruction-level SASS samples are not available on AMD yet.
