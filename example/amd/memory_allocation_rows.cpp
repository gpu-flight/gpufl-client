#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "gpufl/core/monitor.hpp"
#include "gpufl/gpufl.hpp"

namespace {

constexpr int kAllocationsPerPhase = 4;
constexpr uint64_t kExpectedRowsPerPhase = 2 * kAllocationsPerPhase;
constexpr auto kDeliveryTimeout = std::chrono::seconds(2);

bool CheckHip(const hipError_t status, const char* what) {
    if (status == hipSuccess) return true;
    std::cerr << what << " failed: " << hipGetErrorString(status) << "\n";
    return false;
}

bool RunAllocationPhase(const size_t base_bytes) {
    std::vector<void*> allocations;
    allocations.reserve(kAllocationsPerPhase);

    bool ok = true;
    for (int index = 0; index < kAllocationsPerPhase; ++index) {
        void* allocation = nullptr;
        const size_t bytes = base_bytes * static_cast<size_t>(index + 1);
        if (!CheckHip(hipMalloc(&allocation, bytes), "hipMalloc")) {
            ok = false;
            break;
        }
        allocations.push_back(allocation);
    }

    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        if (!CheckHip(hipFree(*it), "hipFree")) ok = false;
    }
    return ok;
}

uint64_t WaitForAllocationRows(const uint64_t minimum_rows) {
    const auto deadline = std::chrono::steady_clock::now() + kDeliveryTimeout;
    uint64_t rows = gpufl::Monitor::MemoryAllocRowsSeen();
    while (rows < minimum_rows && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        rows = gpufl::Monitor::MemoryAllocRowsSeen();
    }
    return rows;
}

}  // namespace

int main() {
    gpufl::InitOptions opts;
    opts.app_name = "amd_memory_allocation_rows";
    opts.log_path = "gfl_amd_memory_rows";
    opts.backend = gpufl::BackendKind::Amd;
    opts.profiling_engine = gpufl::ProfilingEngine::Trace;
    opts.enable_memory_tracking = true;
    opts.continuous_system_sampling = false;
    opts.enable_debug_output = true;
    opts.enable_stack_trace = false;

    if (!gpufl::init(opts)) {
        std::cerr << "Failed to initialize gpufl for AMD allocation tracing\n";
        return 1;
    }

    const std::string engine =
        gpufl::Monitor::ResolvedProfilingEngineWireName();
    const bool engine_ok = engine == "amd.buffer_tracing";
    std::cout << "=== GPUFL AMD Memory Allocation Rows ===\n"
              << "Resolved engine: " << engine << "\n";
    if (!engine_ok) {
        std::cerr << "Expected amd.buffer_tracing; ROCprofiler tracing may be unavailable\n";
    }

    // The first HIP call loads HSA. GPUFlight then retries its deferred
    // ROCprofiler context start from the collector thread.
    void* warmup = nullptr;
    bool workload_ok = CheckHip(hipMalloc(&warmup, 4096), "warmup hipMalloc");
    if (warmup != nullptr) {
        workload_ok = CheckHip(hipFree(warmup), "warmup hipFree") && workload_ok;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    bool priming_ok = false;
    uint64_t rows_before = gpufl::Monitor::MemoryAllocRowsSeen();
    if (workload_ok) {
        priming_ok = RunAllocationPhase(16 * 1024);
        if (priming_ok) {
            rows_before = WaitForAllocationRows(kExpectedRowsPerPhase);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            rows_before = gpufl::Monitor::MemoryAllocRowsSeen();
        }
    }
    const bool priming_rows = rows_before >= kExpectedRowsPerPhase;
    if (!priming_rows) {
        std::cerr << "Priming phase did not emit allocation and free rows\n";
    }
    workload_ok = workload_ok && priming_ok && priming_rows;

    bool phase_a_ok = false;
    if (workload_ok) {
        GFL_SCOPE("memory_rows_phase_a") {
            phase_a_ok = RunAllocationPhase(64 * 1024);
        }
    }
    const uint64_t rows_after_a =
        WaitForAllocationRows(rows_before + kExpectedRowsPerPhase);

    bool phase_b_ok = false;
    if (workload_ok && phase_a_ok) {
        GFL_SCOPE("memory_rows_phase_b") {
            phase_b_ok = RunAllocationPhase(128 * 1024);
        }
    }
    const uint64_t rows_after_b =
        WaitForAllocationRows(rows_after_a + kExpectedRowsPerPhase);

    std::cout << "Allocation rows: before=" << rows_before
              << ", after phase A=" << rows_after_a
              << ", after phase B=" << rows_after_b << "\n";

    const bool phase_a_rows =
        rows_after_a >= rows_before + kExpectedRowsPerPhase;
    const bool phase_b_rows =
        rows_after_b >= rows_after_a + kExpectedRowsPerPhase;
    if (!phase_a_rows) {
        std::cerr << "Phase A did not emit allocation rows\n";
    }
    if (!phase_b_rows) {
        std::cerr << "Phase B did not emit allocation rows\n";
    }

    gpufl::shutdown();
    gpufl::generateReport();

    const bool passed = engine_ok && workload_ok && priming_ok &&
                        priming_rows && phase_a_ok && phase_b_ok &&
                        phase_a_rows && phase_b_rows;
    if (!passed) return 2;

    std::cout << "\nPASS: both phases emitted AMD memory-allocation rows.\n"
              << "Inspect logs with prefix " << opts.log_path
              << " for memory_alloc_event_batch events.\n";
    return 0;
}
