# GPUFL Client Library (multi-header)

Header-only C++ instrumentation library that logs GPU activity and scoped phases in NDJSON, designed for multiple GPU backends. Currently ships with a CUDA backend; other backends can be added behind the same core API.

## Features

- Header-only, zero link-time cost
- NDJSON logging, one event per line
- Scoped monitoring with optional periodic GPU memory sampling
- CUDA kernel auto-timing macros (sync for now)
- Thread-safe logging
- Cross-platform (Windows/Linux) for CUDA; backend abstraction for future platforms

## Backends and auto-detection

- CUDA is picked automatically if compiling with NVCC (`__CUDACC__`).
- To force-select, define one of:
  - `GFL_BACKEND_CUDA`
  - `GFL_BACKEND_OPENCL` (not yet implemented; will error if selected)

## Quick Start

### 1) Include

```cpp
#include <gpufl/gpufl.hpp>
```

### 2) Initialize and Shutdown

```cpp
int main() {
    gpufl::InitOptions opts;
    opts.appName = "my_cuda_app";
    // Base path/prefix (no extension). Three files will be created:
    //   <base>.kernel.log, <base>.scope.log, <base>.system.log
    // If empty, default is "gpufl" in the current working directory.
    opts.logPath = "logs/my_app"; 
    // >0 enables periodic GPU device sampling during scopes
    opts.sampleIntervalMs = 0; 

    gpufl::init(opts);

    // ... your code ...

    gpufl::shutdown();
}
```


### 3) Monitor work

- Block-style scope with automatic begin/sample/end events:

```cpp
GFL_SCOPE("training-epoch") {
    // launch kernels, do work...
}
```

- RAII object (equivalent):

```cpp
{
    gpufl::ScopedMonitor m{"stage-1"};
    // work...
}
```

- Functional helper:

```cpp
gpufl::monitor("data-load", [&]{
    // work...
});
```

**Note:** CUDA kernel monitoring is now automatic via CUPTI event collection. No manual kernel launch macros are required.

## Public API (summary)

```cpp
namespace gpufl {
  struct InitOptions {
    std::string appName;
    // Base path/prefix for logs; three files will be produced:
    //   <base>.kernel.log, <base>.scope.log, <base>.system.log
    // If empty, default is "gpufl" in the current directory
    std::string logPath;   
    uint32_t    sampleIntervalMs = 0; // background sampling for scopes (0 = off)
  };

  bool init(const InitOptions&);
  void shutdown();

  class ScopedMonitor { /* RAII scope begin/sample/end */ };
  void monitor(const std::string& name, const std::function<void()>& fn, const std::string& tag = "");
}

// Macros (core):
//   GFL_SCOPE(name)
//   GFL_SCOPE_TAGGED(name, tag)

// Macros (CUDA backend):
//   GFL_LAUNCH(kernel, grid, block, sharedMem, stream, ...)
//   GFL_LAUNCH_TAGGED(tag, kernel, grid, block, sharedMem, stream, ...)
```

Notes:
- `GFL_SCOPE` optionally samples GPU/device metrics periodically if `sampleIntervalMs > 0`.
- Scope end will call the backend `synchronize()` to ensure timing covers in-flight GPU work.

## Log files and categories

Events are now written to three separate NDJSON files, one line per event:

- Kernel events: `<base>.kernel.log`
- Scope lifecycle/events: `<base>.scope.log`
- System-level periodic sampling: `<base>.system.log`

Meta events (`init`, `shutdown`) are written to all three logs so that each file is self-contained.

## NDJSON events

One JSON object per line (NDJSON). Key event types and shapes (fields may be extended in the future):

- Initialization

```json
{
  "type": "init",
  "pid": 1234,
  "app": "my_cuda_app",
  "ts_ns": 1731958400123456
}
```

- Scope lifecycle (begin, sample, end). Examples (two lines shown as text to illustrate NDJSON):

```text
{"type":"scope_begin","pid":1234,"app":"my_cuda_app","name":"epoch_1","ts_ns":1731958400123456,"devices":[{"id":0,"name":"NVIDIA RTX","uuid":"GPU-...","pci_bus":1,"used_mib":1024,"free_mib":8192,"total_mib":9216,"util_gpu":0,"util_mem":0,"temp_c":0,"power_mw":0,"clk_gfx":0,"clk_sm":0,"clk_mem":0}]}
{"type":"scope_sample","pid":1234,"app":"my_cuda_app","name":"epoch_1","ts_ns":1731958401123456,"devices":[{"id":0,"name":"NVIDIA RTX","uuid":"GPU-...","pci_bus":1,"used_mib":1100,"free_mib":8116,"total_mib":9216,"util_gpu":0,"util_mem":0,"temp_c":0,"power_mw":0,"clk_gfx":0,"clk_sm":0,"clk_mem":0}]}
```

- Scope end includes full timing and final memory snapshot:

```json
{
  "type": "scope_end",
  "pid": 1234,
  "app": "my_cuda_app",
  "name": "epoch_1",
  "ts_start_ns": 1731958400123456,
  "ts_end_ns":   1731958403123456,
  "duration_ns": 3000000,
  "devices": [{"id":0,"name":"NVIDIA RTX","uuid":"GPU-...","pci_bus":1,"used_mib":1110,"free_mib":8106,"total_mib":9216,"util_gpu":0,"util_mem":0,"temp_c":0,"power_mw":0,"clk_gfx":0,"clk_sm":0,"clk_mem":0}]
}
```

- CUDA kernel event (from `GFL_LAUNCH`):

```json
{
  "type": "kernel",
  "pid": 1234,
  "app": "my_cuda_app",
  "devices": [{"id":0,"name":"NVIDIA RTX","uuid":"GPU-...","pci_bus":1,"total_mib":9216}],
  "name": "vectorAdd",
  "ts_start_ns": 1731958400123456,
  "ts_end_ns":   1731958400126789,
  "duration_ns": 3333,
  "grid": "(128,1,1)",
  "block": "(256,1,1)",
  "dyn_shared_bytes": 0,
  "num_regs": 32,
  "static_shared_bytes": 0,
  "local_bytes": 0,
  "const_bytes": 0,
  "cuda_error": "cudaSuccess"
}
```

- System sample (periodic, independent of scopes):

```json
{
  "type": "system_sample",
  "app": "my_cuda_app",
  "name": "system",
  "ts_ns": 1731958402123456,
  "devices": [{"id":0,"name":"NVIDIA RTX","uuid":"GPU-...","pci_bus":1,"used_mib":1024,"free_mib":8192,"total_mib":9216,"util_gpu":0,"util_mem":0,"temp_c":0,"power_mw":0,"clk_gfx":0,"clk_sm":0,"clk_mem":0}]
}
```

- Shutdown

```json
{
  "type": "shutdown",
  "pid": 1234,
  "app": "my_cuda_app",
  "ts_ns": 1731958404123456
}
```

## Data flow and schema

GPUfl client does not send data to your backend directly. Its sole responsibility is to write NDJSON log lines to files.

- Client output: three NDJSON log files written by the target process, based on `InitOptions::logPath` base.
  - `<base>.kernel.log`, `<base>.scope.log`, `<base>.system.log`
- A downstream process (crawler/ingestor) can:
  - Discover and tail the three files
  - Validate each NDJSON line against the schema
  - Forward well-formed events to your backend/metrics pipeline
- Schema location: `gpufl_client/schema` contains human-readable docs and a JSON Schema.
  - `schema/README.md` — event shapes, examples, notes
  - `schema/ndjson.schema.json` — JSON Schema (draft‑07) keyed on the `type` field

## End-to-end example

```cpp
#include <gpufl/gpufl.hpp>
#include <cuda_runtime.h>

__global__
void vectorAdd(int* a, int* b, int* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  gpufl::InitOptions opts;
  opts.appName = "vector_add_demo";
  opts.logPath = "gpufl"; // produces gpufl.kernel.log, gpufl.scope.log, gpufl.system.log
  opts.sampleIntervalMs = 5; // try periodic sampling inside scopes

  gpufl::init(opts);

  dim3 grid(4), block(256);
  int *a, *b, *c; // assume allocated/initialized
  // ... allocate and copy ...

  // Monitor a phase
  GFL_SCOPE("phase-1") {
    vectorAdd<<<grid, block>>>(a, b, c, 1024);
    cudaDeviceSynchronize();
  }

  // Kernel timing (sync)
  GFL_LAUNCH(vectorAdd, grid, block, 0, 0, a, b, c, 1024);

  gpufl::shutdown();
}
```

## CMake integration

The library is header-only. Target name is `gpufl::gpufl`.


```cmake
add_subdirectory(path/to/gpufl/gpufl_client)

add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE gpufl::gpufl)
set_target_properties(my_app PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 17)
```


Install from this directory:

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix <dest>
```

## Building the bundled example

From `gpufl_client` directory:

```bash
cmake -S . -B build
cmake --build build
# Executable name may be gpufl_block_example
``;

On Windows with VS Generator:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Performance notes

- `GFL_LAUNCH` performs a `cudaDeviceSynchronize()` to measure kernel duration; this is simpler but adds overhead.
- Scoped monitoring calls backend `synchronize()` only when the scope ends to capture total work in the scope.

Roadmap items include async timing via CUDA events and non-blocking instrumentation.

## Troubleshooting

- No log files are produced
  - Ensure the directory for your base path exists and the process can write there
  - Pass an absolute `logPath` if running from an unexpected working directory

- CUDA compile errors about `dim3` or runtime headers
  - Make sure the translation unit is compiled with NVCC and includes `<cuda_runtime.h>`

- Link/undefined references
  - This is header-only; usually indicates the file isn’t compiled as CUDA (`.cu`) when needed. Enable `CUDA_SEPARABLE_COMPILATION` or place CUDA code in `.cu` sources

## Python bindings (optional)

Basic Python API is provided via a tiny extension:

```python
import gpufl

# Initialize
gpufl.init(app_name="my_py_app", log_path="logs/my_py", interval_ms=0)

# Scoped monitor (context manager)
with gpufl.scope("stage-1", tag="train"):
    # ... work ...

# Shutdown (optional; occurs at interpreter exit too)
gpufl.shutdown()
```

Arguments:
- app_name: string application name
- log_path: base path/prefix used to create `<base>.kernel.log`, `<base>.scope.log`, `<base>.system.log` (default: `"gpufl"`)
- interval_ms: background sampling interval for scopes (0 disables)
