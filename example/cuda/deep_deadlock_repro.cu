// deep_deadlock_repro.cu
//
// Diagnostic repro for the "Deep + SASS + full CUPTI activity + concurrent
// first-launch" DEADLOCK that was first observed from PyTorch.
//
// ── What we are trying to answer ────────────────────────────────────────────
//
// A PyTorch EMNIST training run (see the C1M2 assignment notebook) hung when
// run under:
//     profiling_engine = Deep        (PC sampling + SASS metrics)
//     enable_stack_trace = true
//     enable_memory_tracking = true
//     a single gpufl.Scope("train_epoch") wrapping the whole training loop
//
// The empirical matrix in docs/compatibility/nvidia-deep-mode-matrix.md
// narrowed the trigger to THREE things happening at once:
//
//   (1) SASS is *armed* (Deep or SassMetrics) inside an active scope, AND
//   (2) the FULL CUPTI activity bundle is enabled alongside SASS
//       (KERNEL / MEMORY2 / SYNC / ... activity records), AND
//   (3) many *distinct* kernels are launched CONCURRENTLY from multiple
//       threads, so each kernel's first launch races to lazy-finalize its
//       CUDA module AND SASS-patch it at the same time.
//
// Under CUDA's default LAZY module loading, those concurrent first-launches
// invert CUPTI/driver lock acquisition order: the launching thread waits on a
// CUPTI/driver lock held by the profiler/callback path, which is itself
// waiting on the launcher → classic two-party deadlock, *inside* the training
// loop (not at shutdown).
//
// The open question this example exists to probe: is that deadlock
// PyTorch-specific (Python GIL handoff, DataLoader worker threads, cuDNN/
// cuBLAS internals), or is it a fundamental CUPTI property reproducible from
// pure C++? PyTorch's distinguishing trait is (3): hundreds of distinct
// kernels launched across threads. So this example deliberately recreates (3)
// in C++ — many distinct __global__ symbols, launched concurrently from a
// pool of std::threads — and combines it with (1) + (2).
//
// ── The defense being validated ─────────────────────────────────────────────
//
// gpufl commit afd6353 flipped the default so that, in any SASS profiler mode,
// the heavy CUPTI activity bundle (2) is DISABLED by default ("safe activity"
// policy); kernel rows then come from launch *callbacks* instead of activity
// records, which removes the lock-inversion window. So:
//
//   * Default (safe activity)                  → expected: COMPLETES.
//   * GPUFL_SASS_ALLOW_FULL_ACTIVITY=1 (unsafe) → expected: may DEADLOCK
//     on an affected GPU/driver, exactly like the original PyTorch run.
//
// Run it both ways to confirm the defense holds on your hardware:
//
//     deep_deadlock_repro                              # safe default — should finish
//     GPUFL_SASS_ALLOW_FULL_ACTIVITY=1 deep_deadlock_repro   # unsafe — may hang
//
// On Windows PowerShell:
//     $env:GPUFL_SASS_ALLOW_FULL_ACTIVITY=1; .\deep_deadlock_repro.exe
//
// ── Safety: this example can genuinely hang ─────────────────────────────────
//
// Because a real deadlock cannot be joined or unwound, a WATCHDOG thread
// arms a wall-clock timeout. If the workload doesn't finish in time, the
// watchdog prints a diagnosis and calls std::_Exit() to force the process
// down (a hung CUPTI/driver lock means no clean shutdown is possible). The
// timeout is generous; a healthy run finishes in a few seconds.
//
// Optional CLI args (order-independent):
//   threads=N     concurrent launcher threads          (default: 8)
//   steps=N       training "steps" per thread          (default: 40)
//   timeout=N     watchdog timeout in seconds           (default: 60)
//   engine=deep|sass|pcsampling|trace                   (default: deep)
//   noscope       do NOT wrap the loop in a scope (SASS never arms → control)

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "gpufl/core/monitor.hpp"
#include "gpufl/gpufl.hpp"

// ── 32 distinct kernels ─────────────────────────────────────────────────────
//
// Each GFL_KERNEL(i) is a separate __global__ symbol, so nvcc emits a
// distinct cubin entry point per index. CUPTI therefore sees 32 distinct PC
// ranges that each need first-launch module finalization + SASS patching —
// the diversity that makes the concurrent first-launch race possible. The
// body carries a unique constant `K` so the compiler can't merge them, and
// loops enough to give the scheduler a real window to interleave launches.
#define GFL_KERNEL(K)                                                       \
    __global__ void distinct_kernel_##K(float* a, int n) {                  \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                    \
        if (idx < n) {                                                      \
            float v = a[idx];                                              \
            for (int i = 0; i < 256; ++i) {                                \
                v = v * 1.000001f + float(K) * 0.5f + float(i) * 1e-6f;    \
            }                                                              \
            a[idx] = v;                                                    \
        }                                                                  \
    }

GFL_KERNEL(0)  GFL_KERNEL(1)  GFL_KERNEL(2)  GFL_KERNEL(3)
GFL_KERNEL(4)  GFL_KERNEL(5)  GFL_KERNEL(6)  GFL_KERNEL(7)
GFL_KERNEL(8)  GFL_KERNEL(9)  GFL_KERNEL(10) GFL_KERNEL(11)
GFL_KERNEL(12) GFL_KERNEL(13) GFL_KERNEL(14) GFL_KERNEL(15)
GFL_KERNEL(16) GFL_KERNEL(17) GFL_KERNEL(18) GFL_KERNEL(19)
GFL_KERNEL(20) GFL_KERNEL(21) GFL_KERNEL(22) GFL_KERNEL(23)
GFL_KERNEL(24) GFL_KERNEL(25) GFL_KERNEL(26) GFL_KERNEL(27)
GFL_KERNEL(28) GFL_KERNEL(29) GFL_KERNEL(30) GFL_KERNEL(31)

using KernelFn = void (*)(float*, int);

static KernelFn kDistinctKernels[] = {
    distinct_kernel_0,  distinct_kernel_1,  distinct_kernel_2,  distinct_kernel_3,
    distinct_kernel_4,  distinct_kernel_5,  distinct_kernel_6,  distinct_kernel_7,
    distinct_kernel_8,  distinct_kernel_9,  distinct_kernel_10, distinct_kernel_11,
    distinct_kernel_12, distinct_kernel_13, distinct_kernel_14, distinct_kernel_15,
    distinct_kernel_16, distinct_kernel_17, distinct_kernel_18, distinct_kernel_19,
    distinct_kernel_20, distinct_kernel_21, distinct_kernel_22, distinct_kernel_23,
    distinct_kernel_24, distinct_kernel_25, distinct_kernel_26, distinct_kernel_27,
    distinct_kernel_28, distinct_kernel_29, distinct_kernel_30, distinct_kernel_31,
};
static constexpr int kNumDistinctKernels =
    sizeof(kDistinctKernels) / sizeof(kDistinctKernels[0]);

// ── CLI parsing helpers ─────────────────────────────────────────────────────

static int intArg(int argc, char** argv, const char* key, int fallback) {
    const size_t klen = std::strlen(key);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], key, klen) == 0 && argv[i][klen] == '=') {
            return std::atoi(argv[i] + klen + 1);
        }
    }
    return fallback;
}

static bool hasFlag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

static gpufl::ProfilingEngine engineArg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "engine=", 7) == 0) {
            const char* v = argv[i] + 7;
            if (std::strcmp(v, "sass") == 0)       return gpufl::ProfilingEngine::SassMetrics;
            if (std::strcmp(v, "pcsampling") == 0) return gpufl::ProfilingEngine::PcSampling;
            if (std::strcmp(v, "trace") == 0)      return gpufl::ProfilingEngine::Trace;
            // default falls through to Deep below
        }
    }
    return gpufl::ProfilingEngine::Deep;
}

static const char* engineName(gpufl::ProfilingEngine e) {
    switch (e) {
        case gpufl::ProfilingEngine::Deep:        return "Deep";
        case gpufl::ProfilingEngine::SassMetrics: return "SassMetrics";
        case gpufl::ProfilingEngine::PcSampling:  return "PcSampling";
        case gpufl::ProfilingEngine::Trace:       return "Trace";
        default:                                  return "other";
    }
}

// ── Workload: one launcher thread ───────────────────────────────────────────
//
// Each thread owns its own device buffer + CUDA stream and walks the full set
// of distinct kernels every "step", starting from a thread-specific offset so
// the threads collide on *first launches* of different symbols at the same
// instant. `ready` + the spin-wait gate releases all threads simultaneously,
// maximizing the concurrent-first-launch overlap that the deadlock needs.
static void launcherThread(int threadId, int steps,
                           std::atomic<bool>& go,
                           std::atomic<int>& liveCount) {
    const int n = 1 << 18;  // 256K elements
    float* d_buf = nullptr;
    cudaStream_t stream{};
    if (cudaMalloc(&d_buf, n * sizeof(float)) != cudaSuccess) {
        liveCount.fetch_sub(1);
        return;
    }
    cudaMemset(d_buf, 0, n * sizeof(float));
    cudaStreamCreate(&stream);

    const dim3 block(256);
    const dim3 grid((n + block.x - 1) / block.x);

    // Release barrier: spin until main flips `go`, so every thread fires its
    // first kernel launch within microseconds of the others.
    while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    for (int s = 0; s < steps; ++s) {
        for (int k = 0; k < kNumDistinctKernels; ++k) {
            // Offset by threadId so threads hit *different* distinct symbols
            // first → maximal distinct-first-launch concurrency.
            int idx = (k + threadId) % kNumDistinctKernels;
            kDistinctKernels[idx]<<<grid, block, 0, stream>>>(d_buf, n);
        }
        // Periodic sync mimics a training step boundary and forces the
        // activity/callback path to flush under contention.
        cudaStreamSynchronize(stream);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_buf);
    liveCount.fetch_sub(1, std::memory_order_release);
}

int main(int argc, char** argv) {
    const int numThreads = intArg(argc, argv, "threads", 8);
    const int steps      = intArg(argc, argv, "steps", 40);
    const int timeoutSec = intArg(argc, argv, "timeout", 60);
    const bool useScope  = !hasFlag(argc, argv, "noscope");
    const gpufl::ProfilingEngine engine = engineArg(argc, argv);

    const char* fullActivity = std::getenv("GPUFL_SASS_ALLOW_FULL_ACTIVITY");
    const bool unsafeMode =
        fullActivity && (std::strcmp(fullActivity, "1") == 0 ||
                         std::strcmp(fullActivity, "true") == 0);

    std::cout << "=== gpufl Deep deadlock repro ===\n"
              << "  engine          : " << engineName(engine) << "\n"
              << "  threads         : " << numThreads << "\n"
              << "  steps/thread    : " << steps << "\n"
              << "  distinct kernels: " << kNumDistinctKernels << "\n"
              << "  scope-wrapped   : " << (useScope ? "yes" : "no (control)") << "\n"
              << "  CUPTI activity  : "
              << (unsafeMode ? "FULL (GPUFL_SASS_ALLOW_FULL_ACTIVITY=1 — UNSAFE, may hang)"
                            : "safe (default defense)")
              << "\n"
              << "  watchdog        : " << timeoutSec << "s\n"
              << "----------------------------------------\n"
              << std::flush;

    if (unsafeMode) {
        std::cout << "[!] Unsafe full-activity mode requested. On an affected\n"
                     "    GPU/driver this is expected to DEADLOCK inside the\n"
                     "    launch loop — the watchdog will force-exit.\n"
                  << std::flush;
    }

    // ── init gpufl with the exact risky combination from the PyTorch run ────
    gpufl::InitOptions opts;
    opts.app_name                 = "deep_deadlock_repro";
    opts.log_path                 = "deep_deadlock_repro";
    opts.profiling_engine         = engine;
    opts.enable_stack_trace       = true;   // matched the hung PyTorch config
    opts.enable_memory_tracking   = true;   // matched the hung PyTorch config
    opts.enable_source_collection = true;
    opts.system_sample_rate_ms    = 50;
    opts.continuous_system_sampling = false;  // scope-bracketed, like the repro
    opts.enable_debug_output      = true;

    if (!gpufl::init(opts)) {
        std::cerr << "gpufl::init failed (no GPU? stub mode?). Aborting.\n";
        return 1;
    }

    // ── Watchdog ────────────────────────────────────────────────────────────
    // A real deadlock can't be joined, so we time the whole workload from a
    // detached thread and hard-exit if it overruns.
    std::atomic<bool> finished{false};
    std::thread watchdog([&]() {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
        while (!finished.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() > deadline) {
                std::cerr
                    << "\n========================================================\n"
                       "[WATCHDOG] No progress for " << timeoutSec << "s — DEADLOCK.\n"
                       "  This reproduces the PyTorch Deep-mode hang in pure C++.\n"
                       "  Trigger present: SASS armed + concurrent distinct\n"
                       "  first-launches"
                    << (unsafeMode ? " + FULL CUPTI activity bundle.\n"
                                  : " (unexpected with the safe default!).\n")
                    << "  A hung CUPTI/driver lock cannot be unwound — forcing exit.\n"
                       "========================================================\n"
                    << std::flush;
                std::_Exit(42);  // distinctive code: "watchdog killed a hang"
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    // ── Run the concurrent workload, optionally inside a scope ──────────────
    std::atomic<bool> go{false};
    std::atomic<int> liveCount{numThreads};
    std::vector<std::thread> launchers;
    launchers.reserve(numThreads);

    auto runWorkload = [&]() {
        for (int t = 0; t < numThreads; ++t)
            launchers.emplace_back(launcherThread, t, steps, std::ref(go),
                                   std::ref(liveCount));
        // Fire the starting gun: all threads begin their first launches now.
        go.store(true, std::memory_order_release);
        for (auto& th : launchers) th.join();
    };

    auto t0 = std::chrono::steady_clock::now();
    if (useScope) {
        // SASS arms on scope entry — this is the window the deadlock needs.
        GFL_SCOPE("train_epoch") {
            runWorkload();
        }
    } else {
        runWorkload();
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    // Reached here → no deadlock.
    finished.store(true, std::memory_order_release);
    watchdog.join();

    const double secs =
        std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\n[OK] Workload completed in " << secs << "s — NO deadlock.\n";
    if (unsafeMode) {
        std::cout << "     (Full-activity mode did NOT hang on this GPU/driver —\n"
                     "      either this hardware is unaffected, or the window was\n"
                     "      not hit this run. Try more threads= / steps=.)\n";
    } else {
        std::cout << "     The safe-activity default held: Deep + SASS ran under\n"
                     "     concurrent distinct first-launches without hanging.\n";
    }

    gpufl::shutdown();
    gpufl::generateReport();
    std::cout << "Logs: " << opts.log_path << "\n";
    return 0;
}
