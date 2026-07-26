// deep_window_demo.cu
//
// Exercises the bounded deep-profiling window end to end: a long-running
// kernel loop that arms deep capture for a few seconds when a metric drops,
// then keeps running once the window closes itself.
//
// It stands in for the real case - a training loop that notices its
// throughput fell and wants to know why, without paying replay cost for the
// whole run.
//
// ── WHAT TO CHECK ────────────────────────────────────────────────────────────
//   1. Before the trigger: the engine is armed but IDLE, so the session
//      carries no PC / PM / SASS samples from those iterations.
//   2. During the window: samples appear, bounded to the window.
//   3. After the window: the process keeps running to completion. A window
//      ending is not a session ending.
//   4. The trigger fires 50 times in a row and still opens ONE window - a
//      re-fire while a window is open is ignored, not an extension. (Note
//      that a trigger left firing across many iterations would correctly
//      open a NEW window each time the previous one expired; that is the
//      caller's cooldown to own, not the library's.)
//   5. The deep_window event records how many launches the window actually
//      covered and which bound closed it. Under a replay engine that count
//      is small; that is the engine, not a failure.
//
// ── RUN ──────────────────────────────────────────────────────────────────────
//   PowerShell:
//     $env:GPUFL_PROFILING_ENGINE = "PmSampling"   # or PcSampling / SassMetrics
//     .\deep_window_demo.exe
//
//   Bound the window by launches instead of time (better for replay engines):
//     $env:GPUFL_DEEP_WINDOW_MAX_LAUNCHES = "200"
//
// PerfWorks engines (PmSampling / RangeProfiler*) need GPU performance-counter
// access: run elevated, or NVIDIA Control Panel -> Developer -> "Manage GPU
// Performance Counters" -> allow all users.

#include <cuda_runtime.h>

#include <cstdlib>  // std::getenv
#include <iostream>
#include <string>

#include "gpufl/gpufl.hpp"

static bool CheckCuda(const cudaError_t err, const char* call, const char* file,
                      const int line) {
    if (err == cudaSuccess) return true;
    std::cerr << "[CUDA ERROR] " << file << ":" << line << " " << call
              << " failed: " << cudaGetErrorString(err) << " ("
              << err << ")" << std::endl;
    return false;
}

#define CHECK_CUDA(call)                                             \
    do {                                                             \
        if (!CheckCuda((call), #call, __FILE__, __LINE__)) return 2; \
    } while (0)

// Long FMA chain - keeps the SMs busy long enough for PC sampling to land
// stall samples and for PM to read steady-state counters.
__global__ void computeHeavy(float* out, const float* in, int n, int iters) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = in[idx];
    for (int i = 0; i < iters; ++i) {
        val = val * 1.0009f + 0.0001f;
        val = fmaf(val, 0.9991f, 0.0002f);
    }
    out[idx] = val;
}

namespace {

gpufl::ProfilingEngine EngineFromEnv() {
    const char* v = std::getenv("GPUFL_PROFILING_ENGINE");
    // gpufl::init() parses this env var itself; we mirror it only so the
    // banner below reports the right thing.
    const std::string name = v ? v : "PmSampling";
    if (name == "PcSampling")  return gpufl::ProfilingEngine::PcSampling;
    if (name == "SassMetrics") return gpufl::ProfilingEngine::SassMetrics;
    if (name == "RangeProfilerKernelReplay")
        return gpufl::ProfilingEngine::RangeProfilerKernelReplay;
    if (name == "Trace")       return gpufl::ProfilingEngine::Trace;
    return gpufl::ProfilingEngine::PmSampling;
}

// The loop has to outlast the window by a wide margin or the run ends first
// and the window closes at session stop, proving nothing. How many
// iterations that takes depends on the GPU and on the engine (kernel replay
// costs ~40x per iteration), so both are tunable for a faster card.
int EnvIntOr(const char* name, const int fallback) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return fallback;
    const int n = std::atoi(v);
    return n > 0 ? n : fallback;
}

}  // namespace

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "deep_window_demo";
    opts.log_path = "deep_window";
    opts.enable_debug_output = true;
    opts.profiling_engine = EngineFromEnv();
    // The point of the demo: engines stay idle until a window opens.
    opts.deep_window_only = true;

    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl" << std::endl;
        return 1;
    }

    std::cout << "=== Bounded deep window demo ===\n"
              << "engine        : "
              << gpufl::ProfilingEngineWireName(opts.profiling_engine) << "\n"
              << "deep_arm_mode : window-only\n"
              << std::endl;

    constexpr int n = 1 << 20;
    constexpr size_t bytes = n * sizeof(float);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemset(d_in, 0, bytes));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // The trigger fires as a tight burst at one iteration, not spread over
    // many. Spreading it across iterations would (correctly) open a fresh
    // window every time the previous one expired, and how many you get
    // would just track how slow the engine is - under kernel replay an
    // iteration costs ~40x what it does under PM sampling. A burst inside
    // a single iteration is over in microseconds, so every call after the
    // first is guaranteed to land while the window is open, which is the
    // invariant worth testing: a re-fire is ignored, never an extension.
    const int kIterations = EnvIntOr("GPUFL_DEMO_ITERATIONS", 3000);
    const int kDropsAt = kIterations / 6;
    const int kTriggerBurst = 50;
    const int kWindowMs = EnvIntOr("GPUFL_DEMO_WINDOW_MS", 1000);

    int trigger_calls = 0;
    int windows_opened = 0;
    int iterations_with_window_open = 0;
    int iterations_after_close = 0;
    bool was_open = false;
    bool fired = false;

    for (int iter = 0; iter < kIterations; ++iter) {
        const double tokens_per_sec = (iter < kDropsAt) ? 1800.0 : 850.0;

        if (tokens_per_sec < 1000.0 && !fired) {
            fired = true;
            for (int i = 0; i < kTriggerBurst; ++i) {
                gpufl::deepWindow(kWindowMs);
                ++trigger_calls;
            }
        }

        const bool window_open = gpufl::deepWindowActive();
        if (window_open && !was_open) {
            ++windows_opened;
            std::cout << "[iter " << iter << "] window opened (" << kWindowMs
                      << "ms)" << std::endl;
        }
        if (!window_open && was_open) {
            std::cout << "[iter " << iter
                      << "] window closed on its own; the loop keeps running"
                      << std::endl;
        }
        was_open = window_open;

        if (window_open) ++iterations_with_window_open;
        else if (windows_opened > 0) ++iterations_after_close;

        computeHeavy<<<blocks, threads>>>(d_out, d_in, n, 4000);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    const bool open_at_end = gpufl::deepWindowActive();
    std::cout << "\n--- summary ---\n"
              << "trigger calls              : " << trigger_calls
              << " (a burst inside one iteration)\n"
              << "windows opened             : " << windows_opened
              << " (expected 1)\n"
              << "iterations with window open: "
              << iterations_with_window_open << " of " << kIterations << "\n"
              << "iterations after close     : " << iterations_after_close
              << "\n"
              << "window still open at end   : " << (open_at_end ? "yes" : "no")
              << "\n\nCheck the session's deep_window event for the launches "
                 "it covered and the bound that closed it."
              << std::endl;

    int failures = 0;
    if (windows_opened != 1) {
        std::cerr << "[FAIL] expected exactly 1 window, saw " << windows_opened
                  << " - a re-fired trigger opened extra windows." << std::endl;
        ++failures;
    }
    if (open_at_end) {
        std::cerr << "[FAIL] the window never closed - it should have hit its "
                  << kWindowMs << "ms deadline." << std::endl;
        ++failures;
    }
    if (iterations_after_close == 0) {
        std::cerr << "[FAIL] no iterations ran after the window closed."
                  << std::endl;
        ++failures;
    }
    if (failures == 0) {
        std::cout << "\n[OK] window opened once, closed on its own bound, and "
                     "the workload outlived it."
                  << std::endl;
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    gpufl::shutdown();
    return failures == 0 ? 0 : 1;
}
