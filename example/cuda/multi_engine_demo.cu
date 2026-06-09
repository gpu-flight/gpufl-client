// multi_engine_demo.cu
//
// Runs MULTIPLE profiling engines in ONE process via the generalized
// CompositeEngine — to measure the engine-compatibility matrix (which engines can
// coexist) and to exercise the redefined Deep (= the maximal coexisting set).
//
// ── HOW TO SELECT ENGINES ────────────────────────────────────────────────────
// Set GPUFL_ENGINE_COMBO (comma-separated canonical engine names) BEFORE running.
// This is the cross-cutting knob: it works here AND for the `gpufl` launcher
// injecting into PyTorch / any CUDA process (you cannot recompile a Python run,
// so the env var is the real mechanism). When set, it overrides the single
// `opts.profiling_engine` selection and builds a CompositeEngine.
//
//   PowerShell (this demo):
//     $env:GPUFL_ENGINE_COMBO = "Trace,PcSampling"
//     .\multi_engine_demo.exe
//
//   Launcher / PyTorch:
//     $env:GPUFL_ENGINE_COMBO = "Trace,PcSampling,PmSampling,RangeProfiler"
//     gpufl.exe trace --engine Trace -- python train.py
//
// Combos to try — Trace gives REAL kernel timings; the rest enrich the same run:
//     Trace,PcSampling             PC stall-reason sampling + real kernels
//     Trace,PmSampling             time-series SM / mem / tensor utilization
//     Trace,RangeProfiler          HW throughput counters (collected per scope)
//     Trace,PcSampling,PmSampling  a triple (if both pairs pass on your GPU)
//   (Trace,SassMetrics DEADLOCKS — SASS + kernel activity is an NVIDIA driver bug.)
//
// ── RUN AS ADMINISTRATOR ─────────────────────────────────────────────────────
// PerfWorks engines (PmSampling / RangeProfiler) call cuptiProfilerInitialize,
// which returns CUPTI_ERROR_UNKNOWN (999) WITHOUT GPU performance-counter access.
// Either run elevated, OR enable: NVIDIA Control Panel -> Developer -> "Manage GPU
// Performance Counters" -> "Allow access to the GPU performance counters to all
// users". (PcSampling + Trace do not need this.)
//
// ── WHY THE SCOPES MATTER ────────────────────────────────────────────────────
// PC / PM / Range collect PER gpufl scope (GFL_SCOPE / GFL_BENCH). Without a scope
// they arm but produce nothing — the final drain races CUDA context teardown.
// Every measured region below is wrapped in a scope so the samplers collect while
// the context is alive.
//
// ── READING THE RESULT ───────────────────────────────────────────────────────
// With enable_debug_output the run prints, after shutdown:
//     [Composite][matrix] PcSamplingEngine  armed=yes produced=yes
//     [Composite][matrix] PmSamplingEngine  armed=yes produced=no
//     ...
// "armed" = the engine started; "produced" = it emitted >=1 sample/record. Plus
// the uploaded session shows real kernels (Trace) alongside each engine's data.
//
// Build (example tree, BUILD_GPUFL_EXAMPLE=ON):
//   cmake --build <build-dir> --config Release --target multi_engine_demo
// NOTE: the gencode in CMakeLists.txt targets sm_86; adjust it to your GPU's
// compute capability for matching cubin (PTX JIT still runs on other archs).

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "gpufl/core/common.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/gpufl.hpp"

static bool CheckCuda(cudaError_t err, const char* call, const char* file,
                      int line) {
    if (err == cudaSuccess) return true;
    std::cerr << "[CUDA ERROR] " << file << ":" << line << " " << call
              << " failed: " << cudaGetErrorString(err) << " ("
              << static_cast<int>(err) << ")" << std::endl;
    return false;
}

#define CHECK_CUDA(call)                                             \
    do {                                                             \
        if (!CheckCuda((call), #call, __FILE__, __LINE__)) return 2; \
    } while (0)

// Compute-bound: a long FMA chain. Keeps the SMs busy long enough for PC sampling
// to land stall samples and for PM / Range to read steady-state counters.
__global__ void computeHeavy(float* out, const float* in, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        for (int i = 0; i < iters; ++i) {
            val = val * 1.0009f + 0.0001f;
            val = fmaf(val, 0.9991f, 0.0002f);
        }
        out[idx] = val;
    }
}

// Memory-bound: strided gather/scatter. A different stall profile (memory
// dependency / long-scoreboard) so PC sampling and the counters show contrast.
__global__ void memoryStride(float* out, const float* in, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float acc = 0.0f;
        for (int k = 0; k < 32; ++k) {
            int j = (idx * stride + k * 131) % n;
            acc += in[j];
        }
        out[idx] = acc;
    }
}

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "multi_engine_demo";
    opts.log_path = "multi_engine";
    opts.system_sample_rate_ms = 10;
    opts.continuous_system_sampling = true;
    opts.enable_debug_output = true;  // shows the [Composite][matrix] lines
    // Base engine when GPUFL_ENGINE_COMBO is NOT set: plain Trace (kernels only).
    // When the combo env IS set it overrides this and builds the CompositeEngine.
    opts.profiling_engine = gpufl::ProfilingEngine::Trace;

    if (const char* combo = std::getenv("GPUFL_ENGINE_COMBO")) {
        std::cout << "GPUFL_ENGINE_COMBO = " << combo << std::endl;
    } else {
        std::cout << "GPUFL_ENGINE_COMBO not set -> plain Trace. Set it to run a "
                     "composite, e.g. Trace,PcSampling,PmSampling" << std::endl;
    }

    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl" << std::endl;
        return 1;
    }

    std::cout << "=== Multi-engine composite demo ===" << std::endl;

    const int n = 1 << 20;  // 1M elements
    const size_t bytes = n * sizeof(float);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    float* h_in = new float[n];
    srand(7);
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // Warmup scope absorbs one-time cost (PTX->SASS JIT, cubin load, context
    // init, PC/PM/Range first-time configuration) so the measured scopes reflect
    // steady-state work.
    std::cout << "  [warmup] priming context + JIT + sampler config..." << std::endl;
    GFL_SCOPE("0_warmup") {
        computeHeavy<<<grid, block>>>(d_out, d_in, n, 256);
        memoryStride<<<grid, block>>>(d_out, d_in, n, 17);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Measured region 1: compute-bound, repeated so the samplers accumulate.
    std::cout << "  [1/3] compute_heavy — 3 warmup + 12 measured" << std::endl;
    GFL_BENCH("1_compute_heavy",
              gpufl::ScopeMeta{}.setRepeat(12).setWarmup(3)) {
        computeHeavy<<<grid, block>>>(d_out, d_in, n, 2048);
        cudaDeviceSynchronize();
    };
    CHECK_CUDA(cudaGetLastError());

    // Measured region 2: memory-bound.
    std::cout << "  [2/3] memory_stride — 3 warmup + 12 measured" << std::endl;
    GFL_BENCH("2_memory_stride",
              gpufl::ScopeMeta{}.setRepeat(12).setWarmup(3)) {
        memoryStride<<<grid, block>>>(d_out, d_in, n, 97);
        cudaDeviceSynchronize();
    };
    CHECK_CUDA(cudaGetLastError());

    // Measured region 3: mixed — both kernels in one scope.
    std::cout << "  [3/3] mixed — both kernels" << std::endl;
    GFL_SCOPE("3_mixed") {
        computeHeavy<<<grid, block>>>(d_out, d_in, n, 1024);
        memoryStride<<<grid, block>>>(d_out, d_in, n, 53);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    delete[] h_in;

    gpufl::shutdown();
    gpufl::generateReport();

    std::cout << "\n=== Done ===" << std::endl;
    std::cout << "Look for '[Composite][matrix] <engine> armed=.. produced=..' "
                 "above, and check the uploaded session for kernels + each "
                 "engine's data." << std::endl;
    return 0;
}
