#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>

#include "gpufl/core/monitor.hpp"
#include "gpufl/gpufl.hpp"

namespace {

constexpr int kElementCount = 1 << 20;
constexpr int kBlockSize = 256;
constexpr int kLaunchesPerPhase = 8;
constexpr int kIterationsPerLaunch = 1024;

bool CheckHip(const hipError_t status, const char* what) {
    if (status == hipSuccess) return true;
    std::cerr << what << " failed: " << hipGetErrorString(status) << "\n";
    return false;
}

__global__ void sampleRowsWorkload(float* values, const int count,
                                   const int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    float x = values[index] +
              static_cast<float>((index & 1023) + 1) * 0.0001f;
    float y = static_cast<float>((threadIdx.x & 31) + 1) * 0.00001f;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        x = x * 1.0000001f + y;
        y = y * 0.9999999f + x * 0.000001f;
        if (x > 4096.0f) x -= 4096.0f;
    }
    values[index] = x + y;
}

bool RunPhase(float* values, const int launches, const int iterations) {
    const dim3 block(kBlockSize);
    const dim3 grid((kElementCount + block.x - 1) / block.x);
    for (int launch = 0; launch < launches; ++launch) {
        hipLaunchKernelGGL(sampleRowsWorkload, grid, block, 0, 0, values,
                           kElementCount, iterations);
    }
    return CheckHip(hipGetLastError(), "sampleRowsWorkload launch") &&
           CheckHip(hipDeviceSynchronize(), "sampleRowsWorkload sync");
}

}  // namespace

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "amd_pm_sampling_sample_rows";
    opts.log_path = "gfl_amd_pm_rows";
    opts.backend = gpufl::BackendKind::Amd;
    opts.profiling_engine = gpufl::ProfilingEngine::PmSampling;
    opts.pm_sampling_interval_us = 1000;
    opts.pm_sampling_max_samples = 4096;
    opts.pm_sampling_preset = "overview";
    opts.pm_sampling_metrics = {"GPUBusy"};
    opts.pm_sampling_scope_only = true;
    opts.continuous_system_sampling = false;
    opts.enable_debug_output = true;
    opts.enable_stack_trace = false;

    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl for AMD PM sampling\n";
        return 1;
    }

    const std::string engine =
        gpufl::Monitor::ResolvedProfilingEngineWireName();
    const bool engine_ok = engine == "amd.device_counting";
    std::cout << "=== GPUFL AMD PM Sample Rows ===\n"
              << "Resolved engine: " << engine << "\n";
    if (!engine_ok) {
        std::cerr << "Expected amd.device_counting; ROCprofiler device "
                     "counting may be unavailable\n";
    }

    float* device_values = nullptr;
    bool workload_ok =
        CheckHip(hipMalloc(&device_values,
                           static_cast<size_t>(kElementCount) * sizeof(float)),
                 "hipMalloc(device_values)");
    if (workload_ok) {
        workload_ok = CheckHip(
            hipMemset(device_values, 0,
                      static_cast<size_t>(kElementCount) * sizeof(float)),
            "hipMemset(device_values)");
    }

    // Warm up HIP and load the kernel before opening a measured scope. This
    // also gives ROCprofiler's deferred device-counting callback time to
    // accept the configured profile.
    if (workload_ok) workload_ok = RunPhase(device_values, 1, 64);

    const uint64_t rows_before = gpufl::Monitor::PmSampleRowsSeen();
    bool phase_a_ok = false;
    if (workload_ok) {
        GFL_SCOPE("pm_rows_phase_a") {
            phase_a_ok = RunPhase(device_values, kLaunchesPerPhase,
                                  kIterationsPerLaunch);
        }
    }
    const uint64_t rows_after_a = gpufl::Monitor::PmSampleRowsSeen();

    bool phase_b_ok = false;
    if (workload_ok && phase_a_ok) {
        GFL_SCOPE("pm_rows_phase_b") {
            phase_b_ok = RunPhase(device_values, kLaunchesPerPhase,
                                  kIterationsPerLaunch);
        }
    }
    const uint64_t rows_after_b = gpufl::Monitor::PmSampleRowsSeen();

    std::cout << "PM rows: before=" << rows_before
              << ", after phase A=" << rows_after_a
              << ", after phase B=" << rows_after_b << "\n";

    const bool phase_a_rows = rows_after_a > rows_before;
    const bool phase_b_rows = rows_after_b > rows_after_a;
    if (!phase_a_rows) {
        std::cerr << "Phase A did not emit a PM sample row\n";
    }
    if (!phase_b_rows) {
        std::cerr << "Phase B did not emit a PM sample row\n";
    }

    if (device_values != nullptr) {
        (void) hipFree(device_values);
    }

    gpufl::shutdown();
    gpufl::generateReport();

    const bool passed = engine_ok && workload_ok && phase_a_ok && phase_b_ok &&
                        phase_a_rows && phase_b_rows;
    if (!passed) return 2;

    std::cout << "\nPASS: both named scopes emitted GPUBusy sample rows.\n"
              << "Inspect logs with prefix " << opts.log_path
              << " for pm_sampling_config and pm_sample_batch events.\n";
    return 0;
}
