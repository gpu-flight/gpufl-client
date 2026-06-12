# NVAPI (Windows-only, fetched via FetchContent)

`gpufl` uses NVIDIA's **NVAPI** to read GPU and memory-controller utilization on
Windows **WDDM** (consumer / GeForce) GPUs. There, NVML's
`nvmlDeviceGetUtilizationRates` returns 0% for CUDA workloads, and the Windows
PDH performance counters expose **no** memory-utilization counter at all. The
`FB` (frame-buffer) domain of `NvAPI_GPU_GetDynamicPstatesInfoEx` is the only
source for `mem_util` on those GPUs; its `GPU` domain also backstops `gpu_util`.

NVAPI is **open source under the MIT license** (<https://github.com/NVIDIA/nvapi>),
including the prebuilt `amd64/nvapi64.lib` import stub. We therefore **do not
commit it here** - the build pulls it via CMake `FetchContent` (same mechanism as
zlib), pinned to a commit.

## How it's sourced (default)

On a Windows build with NVIDIA enabled, CMake clones the repo into
`build/_deps/nvapi-src/` and links `amd64/nvapi64.lib` from there. Pinned to:

```
github.com/NVIDIA/nvapi @ 9b181ea572f680327fe01a14a0f1f41c78034104   # R595
```

The repo ships no tags, so the pin is a commit SHA. **To update**: bump
`GIT_TAG` in the NVAPI block of the top-level `CMakeLists.txt`. Linux / non-Windows
builds never fetch it (`GPUFL_HAS_NVAPI=0`; the NVAPI code in
`nvml_collector.{hpp,cpp}` is compiled out via `#if defined(_WIN32) && GPUFL_HAS_NVAPI`).

## Offline / local override (optional)

To build without the network, point CMake at a local copy instead - it takes
precedence over FetchContent:

- `cmake -DNVAPI_ROOT=C:/path/to/nvapi ...`, **or** set `NVAPI_ROOT` in the env,
  **or** drop the SDK into this folder so the layout is:
  ```
  third_party/nvapi/
    nvapi.h
    nvapi_lite_*.h        (and the other NVAPI headers)
    amd64/nvapi64.lib
  ```

On success you'll see: `-- Found NVAPI: .../nvapi64.lib (mem_util/gpu_util on WDDM enabled)`.

## License compliance

NVAPI is MIT (`SPDX-License-Identifier: MIT`, © NVIDIA). MIT permits commercial
use and redistribution; the only obligation is to **include the copyright +
permission notice** with the distributed product. Since the shipped `gpufl`
binary statically links the `nvapi64.lib` stub, add NVAPI's MIT notice (its
`License.txt`) to the product's third-party license/NOTICE file.
