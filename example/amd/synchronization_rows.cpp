#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>

#include "gpufl/core/monitor.hpp"
#include "gpufl/gpufl.hpp"

namespace {

constexpr int kElementCount = 1 << 18;
constexpr int kBlockSize = 256;
constexpr int kIterations = 256;
constexpr uint64_t kExpectedRowsPerPhase = 4;
constexpr auto kDeliveryTimeout = std::chrono::seconds(2);

bool CheckHip(const hipError_t status, const char* what) {
    if (status == hipSuccess) return true;
    std::cerr << what << " failed: " << hipGetErrorString(status) << "\n";
    return false;
}

__global__ void synchronizationWorkload(float* values, const int count,
                                        const int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    float value = values[index] + static_cast<float>((index & 255) + 1);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        value = value * 1.000001f + 0.000001f;
        if (value > 4096.0f) value -= 4096.0f;
    }
    values[index] = value;
}

bool RunSynchronizationPhase(float* values, const hipStream_t producer,
                             const hipStream_t consumer,
                             const hipEvent_t event) {
    const dim3 block(kBlockSize);
    const dim3 grid((kElementCount + block.x - 1) / block.x);

    hipLaunchKernelGGL(synchronizationWorkload, grid, block, 0, producer,
                       values, kElementCount, kIterations);
    if (!CheckHip(hipGetLastError(), "producer workload launch")) return false;
    if (!CheckHip(hipEventRecord(event, producer), "hipEventRecord")) {
        return false;
    }
    if (!CheckHip(hipStreamWaitEvent(consumer, event, 0),
                  "hipStreamWaitEvent")) {
        return false;
    }

    hipLaunchKernelGGL(synchronizationWorkload, grid, block, 0, consumer,
                       values, kElementCount, kIterations);
    if (!CheckHip(hipGetLastError(), "consumer workload launch")) return false;

    if (!CheckHip(hipStreamSynchronize(consumer), "hipStreamSynchronize")) {
        return false;
    }
    if (!CheckHip(hipEventSynchronize(event), "hipEventSynchronize")) {
        return false;
    }
    return CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

uint64_t WaitForSynchronizationRows(const uint64_t minimum_rows) {
    const auto deadline = std::chrono::steady_clock::now() + kDeliveryTimeout;
    uint64_t rows = gpufl::Monitor::SynchronizationRowsSeen();
    while (rows < minimum_rows && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        rows = gpufl::Monitor::SynchronizationRowsSeen();
    }
    return rows;
}

}  // namespace

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "amd_synchronization_rows";
    opts.log_path = "gfl_amd_sync_rows";
    opts.backend = gpufl::BackendKind::Amd;
    opts.profiling_engine = gpufl::ProfilingEngine::Trace;
    opts.enable_synchronization = true;
    opts.enable_memory_tracking = false;
    opts.continuous_system_sampling = false;
    opts.enable_debug_output = true;
    opts.enable_stack_trace = false;

    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl for AMD synchronization tracing\n";
        return 1;
    }

    const std::string engine =
        gpufl::Monitor::ResolvedProfilingEngineWireName();
    const bool engine_ok = engine == "amd.buffer_tracing";
    std::cout << "=== GPUFL AMD Synchronization Rows ===\n"
              << "Resolved engine: " << engine << "\n";
    if (!engine_ok) {
        std::cerr << "Expected amd.buffer_tracing; ROCprofiler tracing may be unavailable\n";
    }

    float* device_values = nullptr;
    hipStream_t producer = nullptr;
    hipStream_t consumer = nullptr;
    hipEvent_t event = nullptr;

    bool workload_ok = CheckHip(
        hipMalloc(&device_values,
                  static_cast<size_t>(kElementCount) * sizeof(float)),
        "hipMalloc(device_values)");
    if (workload_ok) {
        workload_ok = CheckHip(
            hipMemset(device_values, 0,
                      static_cast<size_t>(kElementCount) * sizeof(float)),
            "hipMemset(device_values)");
    }
    if (workload_ok) {
        workload_ok = CheckHip(
            hipStreamCreateWithFlags(&producer, hipStreamNonBlocking),
            "hipStreamCreate(producer)");
    }
    if (workload_ok) {
        workload_ok = CheckHip(
            hipStreamCreateWithFlags(&consumer, hipStreamNonBlocking),
            "hipStreamCreate(consumer)");
    }
    if (workload_ok) {
        workload_ok = CheckHip(
            hipEventCreateWithFlags(&event, hipEventDisableTiming),
            "hipEventCreate");
    }

    // The first HIP calls load HSA. GPUFlight then retries its deferred
    // ROCprofiler context start from the collector thread.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const uint64_t initial_rows =
        gpufl::Monitor::SynchronizationRowsSeen();
    bool priming_ok = false;
    uint64_t rows_before = initial_rows;
    if (workload_ok) {
        priming_ok = RunSynchronizationPhase(device_values, producer,
                                             consumer, event);
        if (priming_ok) {
            rows_before = WaitForSynchronizationRows(
                initial_rows + kExpectedRowsPerPhase);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            rows_before = gpufl::Monitor::SynchronizationRowsSeen();
        }
    }
    const bool priming_rows =
        rows_before >= initial_rows + kExpectedRowsPerPhase;
    if (!priming_rows) {
        std::cerr << "Priming phase did not emit all synchronization rows\n";
    }
    workload_ok = workload_ok && priming_ok && priming_rows;

    bool phase_a_ok = false;
    if (workload_ok) {
        GFL_SCOPE("sync_rows_phase_a") {
            phase_a_ok = RunSynchronizationPhase(device_values, producer,
                                                 consumer, event);
        }
    }
    const uint64_t rows_after_a = WaitForSynchronizationRows(
        rows_before + kExpectedRowsPerPhase);

    bool phase_b_ok = false;
    if (workload_ok && phase_a_ok) {
        GFL_SCOPE("sync_rows_phase_b") {
            phase_b_ok = RunSynchronizationPhase(device_values, producer,
                                                 consumer, event);
        }
    }
    const uint64_t rows_after_b = WaitForSynchronizationRows(
        rows_after_a + kExpectedRowsPerPhase);

    std::cout << "Synchronization rows: before=" << rows_before
              << ", after phase A=" << rows_after_a
              << ", after phase B=" << rows_after_b << "\n";

    const bool phase_a_rows =
        rows_after_a >= rows_before + kExpectedRowsPerPhase;
    const bool phase_b_rows =
        rows_after_b >= rows_after_a + kExpectedRowsPerPhase;
    if (!phase_a_rows) {
        std::cerr << "Phase A did not emit all synchronization rows\n";
    }
    if (!phase_b_rows) {
        std::cerr << "Phase B did not emit all synchronization rows\n";
    }

    if (event != nullptr) (void)hipEventDestroy(event);
    if (consumer != nullptr) (void)hipStreamDestroy(consumer);
    if (producer != nullptr) (void)hipStreamDestroy(producer);
    if (device_values != nullptr) (void)hipFree(device_values);

    gpufl::shutdown();
    gpufl::generateReport();

    const bool passed = engine_ok && workload_ok && priming_ok &&
                        priming_rows && phase_a_ok && phase_b_ok &&
                        phase_a_rows && phase_b_rows;
    if (!passed) return 2;

    std::cout
        << "\nPASS: both phases emitted the four expected AMD synchronization rows.\n"
        << "Inspect logs with prefix " << opts.log_path
        << " for synchronization_event_batch events.\n";
    return 0;
}
