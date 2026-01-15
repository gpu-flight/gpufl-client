# GPUFlight Client Library (gpufl)

**The Flight Recorder for GPU Production Workloads.**

`gpufl` is a lightweight, high-performance C++ observability library designed for always-on monitoring of GPU applications. 
Unlike traditional profilers (Nsight) that stop the world, GPUFlight is designed to run in production with minimal overhead, capturing kernel telemetry and logical scopes into structured logs.

## 🚀 Key Features

- **Kernel Monitoring**: Automatically intercepts all CUDA kernel launches via CUPTI.
- **Production Grade**: Uses a **Lock-Free Ring Buffer** and a **Background Collector Thread** to decouple logging from your hot path.
- **Logical Scoping**: Group thousands of micro-kernels into meaningful phases (e.g., "Inference", "PhysicsStep") using `GFL_SCOPE`.
- **Rich Metadata**: Captures Kernel Names, Grid/Block dimensions, Register counts, and Shared Memory usage.
- **Sidecar Ready**: Outputs structured NDJSON logs designed to be tailed by a separate Agent/Crawler (e.g., Kafka/Elastic).
- **Vendor Agnostic Design**: Architecture ready for AMD (ROCm) support.

---

## 📦 Integration

### C++ Integration
`gpufl` is designed to be pulled in via CMake `FetchContent`.

### Python Integration
Install the library with all analysis and visualization dependencies:
```bash
pip install ".[numba,viz,analyzer]"
```

---

## 📊 Python Analysis & Visualization

The `gpufl` Python library provides a powerful suite for analyzing logs produced by the C++ library.

### 1. Analyzer (CLI Dashboard)
Use the `analyzer` module to get an "Executive Summary" of your GPU performance directly in the terminal.

```python
from gpufl.analyzer import GpuFlightSession

# Load a session (automatically picks up .kernel, .scope, and .system logs)
session = GpuFlightSession("./logs", log_prefix="stress")

# 1. Executive Summary: Duration, Utilization, Peak VRAM
session.print_summary()

# 2. Hierarchical Scope Analysis: Time spent in GFL_SCOPE blocks
session.inspect_scopes()

# 3. Kernel Hotspots: Top expensive kernels with Stack Trace visualization
session.inspect_hotspots(top_n=5, max_stack_depth=5)
```

### 2. Visualization (Timeline)
The `viz` module provides interactive `matplotlib` plots to correlate kernel execution with system metrics.

```python
import gpufl.viz as viz

# Load all logs in a directory
viz.init("./logs/*.log")

# Show interactive timeline with:
# - GPU/Host utilization & VRAM
# - Kernel occupancy markers
# - Hover-able kernel names (to reduce clutter)
viz.show()
```

---

## 🛠️ Usage (C++)

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX CUDA)

include(FetchContent)
FetchContent_Declare(
    gpufl
    GIT_REPOSITORY [https://github.com/gpu-flight/gpufl-client.git](https://github.com/gpu-flight/gpufl-client.git)
    GIT_TAG        main 
)
FetchContent_MakeAvailable(gpufl)

add_executable(my_app main.cu)

# Link against gpufl and required CUDA libraries
target_link_libraries(my_app PRIVATE gpufl CUDA::cudart CUDA::cupti)

# Ensure DLLs are copied on Windows (Optional but recommended)
# See 'Troubleshooting' section for post-build commands.
```

---

## 🧪 Testing

The project includes a suite of unit tests using GoogleTest. These tests are hardware-aware and will automatically skip NVIDIA-specific tests if a compatible GPU or driver is not detected.

### Running Tests (C++)
The C++ tests use GoogleTest and are hardware-aware.

1.  **Build the tests**:
    ```bash
    cmake --build cmake-build-debug --target gpufl_tests
    ```

2.  **Run via CTest**:
    ```bash
    ctest --test-dir cmake-build-debug --output-on-failure
    ```

3.  **Run directly**:
    ```bash
    ./cmake-build-debug/tests/gpufl_tests.exe
    ```

### Running Tests (Python)
The Python tests use `pytest` and verify the analyzer and visualization logic using mocked data.

1.  **Install pytest**:
    ```bash
    pip install pytest
    ```

2.  **Run tests**:
    ```bash
    # Ensure python directory is in PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:$(pwd)/python
    pytest tests/python
    ```

### Running Tests (CLion)
- The `gpufl_tests` target will appear in your run configurations.
- You can run individual tests or the entire suite using the built-in GoogleTest runner.

## 🔧 Linux Configuration (Required for CUPTI)

To allow non-root users to profile GPU kernels (using CUPTI/PC Sampling) on Linux, you must relax the NVIDIA driver security restrictions. Without this, `gpufl` may fail to capture kernel activity.

1. **Create a configuration file:**
   ```bash
   sudo nano /etc/modprobe.d/nvidia-profiler.conf

2. **Add the following line:**
    ```bash
   options nvidia NVreg_RestrictProfilingToAdminUsers=0
   
3. **Apply changes and reboot:**
    ```bash
    sudo update-initramfs -u
    sudo reboot