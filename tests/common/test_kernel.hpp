#pragma once

namespace gpufl::test {

/**
 * @brief Launches a small FMA-bound kernel that runs long enough for CUPTI to
 * gather useful data in any profiling engine.
 *
 * Used by the engine-coverage tests. The default iteration count is tuned so
 * the kernel executes in roughly 10 ms on a modern GPU — enough time for PC
 * Sampling to collect a handful of samples and for SASS / Range Profiler
 * replay passes to complete.
 *
 * No-op if CUDA is not available at runtime (returns without launching).
 *
 * @param iters   per-thread FMA iteration count (default ~2 million)
 * @param blocks  grid dim
 * @param threads block dim
 */
void RunTestKernel(int iters = 2'000'000, int blocks = 4, int threads = 256);

}  // namespace gpufl::test
