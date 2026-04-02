# AMD / ROCm Examples

This folder mirrors the CUDA example area with runnable HIP examples for AMD GPUs.

## What Works Today

- ROCm / AMD system telemetry via `rocm_smi`
- AMD static device inventory via HIP
- AMD kernel dispatch tracing via `rocprofiler-sdk`
- AMD memcpy tracing via `rocprofiler-sdk`
- `gpufl` initialization with `backend = gpufl::BackendKind::Amd`
- User-defined scope logging via `GFL_SCOPE(...)`
- HIP example programs that run on ROCm hardware

## What Does Not Work Yet

- AMD profiling engines equivalent to CUPTI PC Sampling / SASS Metrics / Range Profiler

Today, the AMD backend is useful for:

- system metric logging
- device inventory
- automatic HIP kernel and memcpy tracing
- scope-level application instrumentation

It is not yet useful for:

- instruction-level or hardware-counter profiling

## Targets

- `amd_check_device`
  - Basic HIP device detection smoke test
- `amd_vector_add_benchmark`
  - HIP vector add benchmark with result verification
- `amd_gpufl_scope_demo`
  - Initializes `gpufl` with the AMD backend, runs HIP work inside scopes, and writes logs

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
cmake --build build-rocm-examples --target amd_vector_add_benchmark
cmake --build build-rocm-examples --target amd_gpufl_scope_demo
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
./build-rocm-examples/example/amd/amd_vector_add_benchmark
./build-rocm-examples/example/amd/amd_gpufl_scope_demo
```

On a working ROCm system, `amd_check_device` should print output similar to:

```text
Found 2 HIP devices.
Success! Device 0: AMD Radeon RX 9070 XT (arch gfx1201, capability 12.0)
```

## Logs

`amd_gpufl_scope_demo` writes logs with prefix:

```bash
gfl_amd_scope
```

With `rocprofiler-sdk` available, expect:

- `job_start` inventory
- kernel dictionaries
- `kernel_event_batch`
- `kernel_detail`
- `memcpy_event_batch`
- system metric samples
- scope events

Without `rocprofiler-sdk`, expect only telemetry, static inventory, and scope
events.

Do not expect AMD profiling samples such as PC sampling or hardware counters.
