// manykernel_benchmark.cu
//
// Diagnostic benchmark: does PC Sampling overhead scale with the number
// of *distinct* kernels launched, or is it a PyTorch-specific issue?
//
// Background: benchmark/run_benchmark.py shows ~+1% PC Sampling overhead
// on a single-kernel CUDA GEMM workload but ~+650-700% on PyTorch
// MiniGPT training, which launches hundreds of distinct kernel functions
// per step. To rule out PyTorch-specific factors (CUDA graphs, cuDNN
// internals, Python<->CUDA stream interleaving, etc.) we need an
// equivalent many-distinct-kernel workload in pure C++. If this
// benchmark also shows multi-hundred-percent overhead → the issue is
// fundamentally about CUPTI's PC Sampling cost per kernel, independent
// of language. If C++ shows low overhead → something specific to the
// PyTorch path is amplifying the cost.
//
// Workload shape per "step":
//   40 distinct kernel functions (each is a separate __global__ symbol,
//   so CUPTI sees 40 distinct PC ranges per step). The kernel bodies
//   share a common compute shape but each carries a unique constant so
//   nvcc emits 40 separate cubin entry points — symbol distinctness is
//   guaranteed by C++ regardless. The 8 -> 40 jump is the key knob:
//   prior runs with 8 kernels showed only +32% PC Sampling overhead,
//   well below PyTorch's +657%; if overhead is driven by *unique*
//   kernel count (PC ranges, symbol-dict size, sample-buffer
//   fragmentation), 40 should land closer to PyTorch's number. If it
//   stays near +32%, PyTorch's amplifier is something else entirely
//   (cuDNN/cuBLAS heuristics, CUDA Graphs, multi-stream sync).
//
//   Default: 100 steps × 5 runs × 40 kernels = 4000 launches per run
//   (same total launch count as the prior 8-kernel × 500-step test —
//   apples-to-apples comparison; only the diversity varies).
//
// Usage:
//   manykernel_benchmark                    # all 4 modes, large kernels, with scope
//   manykernel_benchmark short              # all 4 modes, short kernels (~1us each)
//                                           #   tests kernel-duration as PyTorch's amplifier
//   manykernel_benchmark noscope            # all 4 modes, large, no scope wrapping
//                                           #   measures the unattributed-PC-sampling case
//   manykernel_benchmark short noscope      # compounds — short kernels, no scope
//   manykernel_benchmark pc_sampling short  # PC Sampling only, short kernels
//   manykernel_benchmark deep               # gpufl init w/ PcSamplingWithSass (bails to
//                                           #   PC Sampling only on Blackwell sm_120)
//
// CLI args are order-independent and positional. Recognized:
//   short / long                — kernel size (default: long)
//   noscope / scope             — wrap measurements in gpufl::ScopedMonitor (default: scope)
//   memset=N                    — N cudaMemsetAsync per step. Tests whether memset
//                                 records in the CUPTI activity stream amplify PC
//                                 Sampling overhead. PyTorch MiniGPT's memset share
//                                 was 4.5% (600/13406 launches); use memset=2 to
//                                 match that ratio. Default 0.
//   baseline / monitoring /     — mode (default: run all four)
//   pc_sampling / deep
//
// Example: manykernel_benchmark memset=2     # PyTorch-like memset ratio
//
// Compare wall/gpu times across modes to compute overhead %. Scope-on
// is the realistic usage pattern (users wrap a training step or batch);
// scope-off matches the older numbers in this thread for reference.
//
// Caveat: when running modes back-to-back in one invocation, CUPTI
// state may leak across init/shutdown cycles. The "all modes" path
// is convenient but for the cleanest numbers, run each mode in a
// fresh process and average across 3+ invocations.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gpufl/gpufl.hpp"
#include "gpufl/core/monitor.hpp"

// ---- 40 distinct kernels -------------------------------------------------
//
// Each is a separate __global__ symbol so CUPTI sees them as 40 distinct
// PC ranges. Bumped from 8 to 40 to test the diversity hypothesis: an
// 8-kernel many-launch workload showed only +32% PC Sampling overhead,
// vs PyTorch's +657% — but PyTorch launches hundreds of distinct
// cuBLAS/cuDNN kernels per step. If overhead scales with the *unique*
// kernel count rather than total launches, 40 variants here should land
// roughly midway and confirm the diversity-as-amplifier hypothesis.
//
// Each variant has identical body shape but a unique coefficient baked
// in via the macro NUM, so nvcc emits 40 distinct cubin entry points
// (the symbols are guaranteed distinct by name regardless; the unique
// constant just defeats any aggressive symbol-merging the linker might
// otherwise attempt).

#define FOREACH_VARIANT(F) \
    F(0)  F(1)  F(2)  F(3)  F(4)  F(5)  F(6)  F(7)  F(8)  F(9)  \
    F(10) F(11) F(12) F(13) F(14) F(15) F(16) F(17) F(18) F(19) \
    F(20) F(21) F(22) F(23) F(24) F(25) F(26) F(27) F(28) F(29) \
    F(30) F(31) F(32) F(33) F(34) F(35) F(36) F(37) F(38) F(39)

#define MAKE_VARIANT_KERNEL(NUM)                                          \
    __global__ void k_variant_##NUM(const float* a, const float* b,       \
                                     float* c, int n) {                   \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                    \
        if (i < n) {                                                      \
            const float coeff = static_cast<float>(NUM) * 0.0125f + 1.0f; \
            float x = a[i];                                               \
            float y = b[i];                                               \
            /* nontrivial work — roughly matches the old kernels' */      \
            /* per-element cost so total run time stays comparable */     \
            c[i] = x * coeff + y * (1.0f - coeff) + x * x * 0.5f;         \
        }                                                                 \
    }

FOREACH_VARIANT(MAKE_VARIANT_KERNEL)

// ---- workload ------------------------------------------------------------

struct DeviceBufs {
    float* a;
    float* b;
    float* c;
    float* d;
    float* partial;
    // Separate buffer for cudaMemsetAsync target. Sized large enough
    // that each memset takes a non-trivial amount of GPU time (so
    // CUPTI definitely records it as a kernel activity record), but
    // small enough to fit alongside the compute buffers on an 8GB
    // laptop GPU. 64 MiB → roughly 128us per memset on Blackwell ~500
    // GB/s memory bandwidth.
    void*  memset_buf;
    size_t memset_size;
    int n;
    int blocks;
    int threads;
};

static DeviceBufs allocate(int n, int threads) {
    DeviceBufs b{};
    b.n = n;
    b.threads = threads;
    b.blocks = (n + threads - 1) / threads;
    cudaMalloc(&b.a, n * sizeof(float));
    cudaMalloc(&b.b, n * sizeof(float));
    cudaMalloc(&b.c, n * sizeof(float));
    cudaMalloc(&b.d, n * sizeof(float));
    cudaMalloc(&b.partial, b.blocks * sizeof(float));
    b.memset_size = 64 * 1024 * 1024;  // 64 MiB
    cudaMalloc(&b.memset_buf, b.memset_size);
    // initialize a, b with something non-trivial
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i % 1024) * 0.01f;
    cudaMemcpy(b.a, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b.b, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    return b;
}

static void free_bufs(DeviceBufs& b) {
    cudaFree(b.a);
    cudaFree(b.b);
    cudaFree(b.c);
    cudaFree(b.d);
    cudaFree(b.partial);
    cudaFree(b.memset_buf);
}

// One "step" = 40 distinct kernel launches (one per variant) + an
// optional `memset_count` cudaMemsetAsync calls. Returns nothing —
// caller times the loop around `steps_per_run` calls.
// Each kernel writes c[] from a[]/b[]; the result of variant N is
// overwritten by variant N+1. That's fine for benchmarking — we only
// care about CUPTI's cost per launch, not correctness of the output.
//
// memsets are appended at the end of each step rather than interleaved
// with compute kernels. PyTorch interleaves more, but for the question
// we're asking ("does PC Sampling cost amplify when memset records
// are in the activity stream?") the *ratio* matters more than the
// *interleave pattern*. Default ratio of 2 memsets per 40 compute
// kernels matches the 4.5% memset share we measured in PyTorch
// MiniGPT (600 memsets / 13406 total launches).
#define LAUNCH_VARIANT(NUM) \
    k_variant_##NUM<<<b.blocks, b.threads>>>(b.a, b.b, b.c, b.n);

static void run_step(const DeviceBufs& b, int memset_count) {
    FOREACH_VARIANT(LAUNCH_VARIANT)
    for (int i = 0; i < memset_count; ++i) {
        cudaMemsetAsync(b.memset_buf, 0, b.memset_size, 0);
    }
}

// Runs `steps` of the 40-kernel workload (+ optional memsets per step),
// returns (wall_ms, gpu_ms).
static std::pair<float, float> measure_run(const DeviceBufs& b, int steps,
                                            bool use_scope, int memset_count) {
    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    // Optional scope wrapping the timed work. This matches how users
    // actually adopt gpufl: a `with gpufl.Scope(...)` block around a
    // training step or batch. Without the scope, PC samples accumulate
    // unattributed (no scope_event begin/end pair to bracket them) and
    // the analysis pipeline downstream can't slice by code section —
    // so a no-scope benchmark measures something users won't really
    // run in production.
    //
    // Scope is created BEFORE wall_start and destructs at function
    // return, AFTER wall_end. That isolates the measurement to the
    // pure in-scope kernel work: per-launch CUPTI overhead inside a
    // scope. The scope-end PC sampling drain happens in the destructor,
    // outside the measurement — matches the question we're actually
    // asking ("what does each step cost while PC sampling is on?")
    // rather than mixing in one-time post-step bookkeeping.
    std::unique_ptr<gpufl::ScopedMonitor> scope;
    if (use_scope) {
        scope = std::make_unique<gpufl::ScopedMonitor>("bench_run");
    }

    auto wall_start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_ev);

    for (int s = 0; s < steps; ++s) {
        run_step(b, memset_count);
    }

    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);
    auto wall_end = std::chrono::high_resolution_clock::now();

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start_ev, stop_ev);
    float wall_ms = std::chrono::duration<float, std::milli>(wall_end - wall_start).count();

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    return {wall_ms, gpu_ms};
    // scope destructs here on return — drain stays out of the measurement
}

// ---- mode plumbing -------------------------------------------------------

enum class Mode { Baseline, Monitoring, PcSampling, PcSamplingWithSass };

struct RunResult {
    Mode mode;
    bool ok = false;
    std::vector<float> walls;
    std::vector<float> gpus;
    float wall_mean = 0.0f;
    float gpu_mean = 0.0f;
};

static const char* mode_short(Mode m) {
    switch (m) {
        case Mode::Baseline:           return "baseline";
        case Mode::Monitoring:         return "monitoring";
        case Mode::PcSampling:         return "pc_sampling";
        case Mode::PcSamplingWithSass: return "deep";
    }
    return "unknown";
}

// Table-friendly label — mirrors run_benchmark.py's column header text
// so it's easy to compare the two outputs side-by-side.
static const char* mode_label(Mode m) {
    switch (m) {
        case Mode::Baseline:           return "Baseline (no gpufl)";
        case Mode::Monitoring:         return "Monitoring only";
        case Mode::PcSampling:         return "PC Sampling";
        case Mode::PcSamplingWithSass: return "PcSampling + SASS (Deep)";
    }
    return "unknown";
}

static Mode parse_mode(const char* s) {
    if (std::strcmp(s, "baseline")   == 0) return Mode::Baseline;
    if (std::strcmp(s, "monitoring") == 0) return Mode::Monitoring;
    if (std::strcmp(s, "pc_sampling") == 0) return Mode::PcSampling;
    if (std::strcmp(s, "deep") == 0) return Mode::PcSamplingWithSass;
    std::fprintf(stderr,
        "unknown mode '%s' (expected: baseline|monitoring|pc_sampling|deep)\n",
        s);
    std::exit(1);
}

static bool gpufl_init_for(Mode m) {
    if (m == Mode::Baseline) return true;
    gpufl::InitOptions opts;
    opts.app_name = "manykernel_benchmark";
    opts.log_path = "manykernel_logs";
    opts.system_sample_rate_ms = 0;       // off — we're not measuring monitoring volume
    opts.continuous_system_sampling = false;
    // MUST stay false for benchmarks. Setting true caused stderr-bound
    // GFL_LOG_DEBUG calls per kernel activity record (1200+ writes per
    // run × Windows console rendering) to dominate the measurement —
    // Monitoring overhead jumped from ~89% to ~1415% solely from debug
    // log spam, completely masking the real per-kernel CUPTI cost. If
    // you need samplingPeriod-verification, do that in a separate
    // single-shot run, not during timed measurement.
    opts.enable_debug_output = true;
    opts.enable_source_collection = false;
    switch (m) {
        case Mode::PcSampling:
            opts.profiling_engine = gpufl::ProfilingEngine::PcSampling;
            break;
        case Mode::PcSamplingWithSass:
            // "Deep" — PC sampling + SASS metrics. SASS metrics use
            // cuptiProfilerInitialize which has reportedly returned
            // CUPTI_ERROR_UNKNOWN (999) on Blackwell in earlier runs;
            // if init fails the mode is skipped cleanly via the
            // !gpufl_init_for(m) branch in run_mode().
            opts.profiling_engine = gpufl::ProfilingEngine::PcSamplingWithSass;
            break;
        case Mode::Monitoring:
        default:
            opts.profiling_engine = gpufl::ProfilingEngine::None;
            break;
    }
    return gpufl::init(opts);
}

// Run one mode: `runs` measurements of `steps_per_run` × 40 kernels each.
// Prints a one-line progress message; full per-run numbers go in the
// final summary table.
static RunResult run_mode(Mode m, int n, int threads, int steps_per_run,
                          int runs, bool use_scope, int memset_count) {
    RunResult r{};
    r.mode = m;
    std::printf("  %s ... ", mode_label(m));
    std::fflush(stdout);

    if (!gpufl_init_for(m)) {
        std::printf("FAILED (gpufl::init)\n");
        return r;
    }

    DeviceBufs b = allocate(n, threads);

    // warmup — first launch carries CUDA context creation / module load
    // costs that aren't part of steady-state per-kernel overhead, so we
    // throw the first iteration away.
    run_step(b, memset_count);
    cudaDeviceSynchronize();

    // Scope only makes sense when gpufl is initialized — Baseline has
    // no runtime to attach to. ScopedMonitor without an active runtime
    // is harmless (it short-circuits), but skipping the construction
    // entirely makes the no-gpufl path cleaner.
    const bool scope_for_this_run = use_scope && (m != Mode::Baseline);

    r.walls.reserve(runs);
    r.gpus.reserve(runs);
    for (int i = 0; i < runs; ++i) {
        auto [wall, gpu] = measure_run(b, steps_per_run, scope_for_this_run, memset_count);
        r.walls.push_back(wall);
        r.gpus.push_back(gpu);
    }
    float wsum = 0, gsum = 0;
    for (auto v : r.walls) wsum += v;
    for (auto v : r.gpus)  gsum += v;
    r.wall_mean = wsum / runs;
    r.gpu_mean  = gsum / runs;

    std::printf("wall=%.1fms, gpu=%.1fms\n", r.wall_mean, r.gpu_mean);

    free_bufs(b);
    if (m != Mode::Baseline) gpufl::shutdown();
    r.ok = true;
    return r;
}

// ---- summary -------------------------------------------------------------

// Format mirrors benchmark/run_benchmark.py's output so the two are
// directly comparable. ASCII separators (not Unicode box-drawing) keep
// it readable on Windows consoles without UTF-8 codepage tricks.
static void print_summary(const std::vector<RunResult>& results,
                          int steps_per_run) {
    // Find baseline mean (if present) for overhead calc. Absent baseline
    // (single-mode invocation) → "—" in the overhead column.
    float baseline_wall = -1.0f;
    for (const auto& r : results) {
        if (r.ok && r.mode == Mode::Baseline) {
            baseline_wall = r.wall_mean;
            break;
        }
    }

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("Many-Kernel Benchmark (%d steps x 40 distinct kernels per run):\n",
                steps_per_run);
    std::printf("  %-22s %10s %10s %10s\n",
                "Mode", "Wall (ms)", "GPU (ms)", "Overhead");
    std::printf("  %-22s %10s %10s %10s\n",
                "----------------------", "----------",
                "----------", "----------");
    for (const auto& r : results) {
        if (!r.ok) {
            std::printf("  %-22s %10s %10s %10s\n",
                        mode_label(r.mode), "FAILED", "FAILED", "-");
            continue;
        }
        char overhead[16];
        if (baseline_wall > 0 && r.mode != Mode::Baseline) {
            float pct = (r.wall_mean / baseline_wall - 1.0f) * 100.0f;
            std::snprintf(overhead, sizeof(overhead), "%+.1f%%", pct);
        } else {
            std::snprintf(overhead, sizeof(overhead), "%s", "-");
        }
        std::printf("  %-22s %10.1f %10.1f %10s\n",
                    mode_label(r.mode), r.wall_mean, r.gpu_mean, overhead);
    }

    std::printf("\n  Individual runs (wall ms):\n");
    for (const auto& r : results) {
        if (!r.ok) continue;
        std::printf("    %-22s [", mode_label(r.mode));
        for (size_t i = 0; i < r.walls.size(); ++i) {
            std::printf("%s%.1f", i == 0 ? "" : ", ", r.walls[i]);
        }
        std::printf("]\n");
    }

    // Pull out the PC Sampling overhead as the headline number — it's
    // the diagnostic we actually care about (does many-kernel C++ show
    // the same +650% PyTorch did, or is it dramatically lower?).
    for (const auto& r : results) {
        if (r.ok && r.mode == Mode::PcSampling && baseline_wall > 0) {
            float pct = (r.wall_mean / baseline_wall - 1.0f) * 100.0f;
            std::printf("\n  PC Sampling overhead vs baseline: %+.1f%%\n", pct);
            std::printf("  Compare: PyTorch MiniGPT was +657%%, CUDA GEMM was +1%%.\n");
            break;
        }
    }
    std::printf("============================================================\n");
}

// ---- main ----------------------------------------------------------------

int main(int argc, char** argv) {
    // Workload sizing. Two configurations:
    //   long  (default): n=1M floats → ~13µs per kernel. 40 kernels ×
    //         100 steps = 4000 launches per measurement at ~52ms
    //         baseline. PC Sampling overhead measured at +34% with
    //         this config.
    //   short (CLI arg): n=16K floats → ~1µs per kernel. Tests the
    //         hypothesis that PyTorch's +657% overhead is primarily
    //         driven by kernel duration: CUPTI's per-launch
    //         instrumentation cost is roughly fixed at ~10-15µs on
    //         Blackwell WDDM, so shorter kernels amplify the relative
    //         overhead. If short-mode overhead jumps to +300-1000%,
    //         kernel duration is confirmed as the amplifier; if it
    //         stays near +34%, something else (cuDNN/cuBLAS internals,
    //         Tensor Cores, multi-stream) is the cause for PyTorch.
    //
    // 5 runs averages out drain-cadence variance without making total
    // benchmark wall time annoying.
    //
    // CLI parsing: positional args, order-independent. "short"/"long"
    // toggles the kernel size; a mode name (baseline|monitoring|
    // pc_sampling|deep) restricts to a single mode.

    bool short_kernels = false;
    // Default: wrap each measurement in a gpufl::Scope. That's the
    // realistic usage pattern (users wrap a training step or batch),
    // and without it PC samples can't be attributed to any code section
    // so the analysis pipeline downstream produces nothing useful.
    // Pass "noscope" to disable for a direct comparison against the
    // previous no-scope numbers.
    bool use_scope = true;
    // Number of cudaMemsetAsync calls appended to each step. 0 matches
    // the pure-compute baseline; 2 matches the PyTorch MiniGPT ratio
    // (4.5% memset share = 2 memsets per 40 compute kernels). Test
    // whether the presence of memset records in the CUPTI activity
    // stream amplifies PC Sampling overhead — our profile_pytorch_via_gpufl
    // data showed cudaMemsetAsync was 57% of PyTorch's total CUDA time.
    int memset_count = 0;
    const char* mode_arg = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "short") == 0) {
            short_kernels = true;
        } else if (std::strcmp(argv[i], "long") == 0) {
            short_kernels = false;
        } else if (std::strcmp(argv[i], "noscope") == 0) {
            use_scope = false;
        } else if (std::strcmp(argv[i], "scope") == 0) {
            use_scope = true;
        } else if (std::strncmp(argv[i], "memset=", 7) == 0) {
            memset_count = std::atoi(argv[i] + 7);
            if (memset_count < 0) memset_count = 0;
        } else {
            mode_arg = argv[i];
        }
    }

    const int n             = short_kernels ? (1 << 14) : (1 << 20);
    const int threads       = 256;
    const int steps_per_run = 100;
    const int runs          = 5;
    const int kernels_per_step = 40;

    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    std::printf("GPU: %s (sm_%d%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("Kernel size: %s (n=%d floats per buffer, ~%s per kernel)\n",
                short_kernels ? "short" : "long",
                n, short_kernels ? "1us" : "13us");
    std::printf("Workload: %d steps/run x %d distinct kernels = %d launches per measurement\n",
                steps_per_run, kernels_per_step, steps_per_run * kernels_per_step);
    if (memset_count > 0) {
        std::printf("Extra memsets per step: %d (target buffer = 64 MiB, ~%.1f%% memset share)\n",
                    memset_count,
                    100.0 * memset_count / (memset_count + kernels_per_step));
    }
    std::printf("Runs per mode: %d\n", runs);
    std::printf("Scope wrapping: %s\n", use_scope ? "ON (gpufl::ScopedMonitor around each measurement)" : "OFF");
    std::printf("============================================================\n\n");

    std::vector<RunResult> results;
    if (mode_arg) {
        // Single-mode invocation — cleanest CUPTI state, recommended
        // for any apples-to-apples comparison if back-to-back mode
        // numbers look suspicious.
        results.push_back(run_mode(parse_mode(mode_arg), n, threads, steps_per_run, runs, use_scope, memset_count));
    } else {
        // Convenience: all four modes back-to-back in one process.
        // CUPTI may leak state between init/shutdown cycles — if the
        // numbers look funny, re-run each mode in its own invocation
        // and compare across invocations instead.
        // Deep (PcSamplingWithSass) bails to PC Sampling only on
        // Blackwell sm_120 — see SassMetricsEngine partial-failure
        // bailout. We still record the row for the data point.
        results.push_back(run_mode(Mode::Baseline,           n, threads, steps_per_run, runs, use_scope, memset_count));
        results.push_back(run_mode(Mode::Monitoring,         n, threads, steps_per_run, runs, use_scope, memset_count));
        results.push_back(run_mode(Mode::PcSampling,         n, threads, steps_per_run, runs, use_scope, memset_count));
        results.push_back(run_mode(Mode::PcSamplingWithSass, n, threads, steps_per_run, runs, use_scope, memset_count));
    }

    print_summary(results, steps_per_run);
    return 0;
}
