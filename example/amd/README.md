# AMD / ROCm Examples

This folder mirrors the CUDA example area with runnable HIP examples for AMD GPUs.

## What Works Today

- ROCm / AMD system telemetry via `rocm_smi`
- AMD static device inventory via HIP
- `gpufl` initialization with `backend = gpufl::BackendKind::Amd`
- User-defined scope logging via `GFL_SCOPE(...)`
- HIP example programs that run on ROCm hardware

## What Does Not Work Yet

- AMD kernel activity tracing
- AMD memcpy tracing
- AMD profiling engines equivalent to CUPTI PC Sampling / SASS Metrics / Range Profiler

Today, the AMD backend is useful for:

- system metric logging
- device inventory
- scope-level application instrumentation

It is not yet useful for:

- automatic HIP kernel tracing
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

## Run

```bash
./build-rocm-examples/example/amd/amd_check_device
./build-rocm-examples/example/amd/amd_vector_add_benchmark
./build-rocm-examples/example/amd/amd_gpufl_scope_demo
```

## Logs

`amd_gpufl_scope_demo` writes logs with prefix:

```bash
gfl_amd_scope
```

Because AMD tracing is not implemented yet, expect:

- `job_start` inventory
- system metric samples
- scope events

Do not expect:

- automatic kernel events
- memcpy events
- AMD profiling samples
