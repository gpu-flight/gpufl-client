// Fixed-work decode-shaped target for the workload benchmark.
//
// Every configuration runs THIS binary with THESE arguments; only the rule
// configuration around it changes. Which config fires is decided by the
// threshold, not by reshaping the workload - a benchmark that slows the
// workload down to make the rule fire is measuring two changes at once.
//
// Fixed WORK (iterations), not fixed time: throughput and CPU time are only
// comparable when every run did the same thing. The kernel is deliberately
// small - a high iteration rate is the WORST case for per-iteration overhead,
// which is the thing being measured.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExtCounters.h>
#include <nvtx3/nvToolsExtSemanticsCounters.h>

__global__ void decode_step(float* x, int inner) {
    float v = x[threadIdx.x];
    for (int i = 0; i < inner; ++i) v = v * 1.0001f + 0.5f;
    x[threadIdx.x] = v;
}

int main(int argc, char** argv) {
    const long iters = argc > 1 ? std::atol(argv[1]) : 200000L;
    const int tokens_per_step = argc > 2 ? std::atoi(argv[2]) : 32;
    const int inner = argc > 3 ? std::atoi(argv[3]) : 200;

    nvtxDomainHandle_t domain = nvtxDomainCreateA("bench");
    nvtxSemanticsCounter_t sem = {};
    sem.header.structSize = sizeof(sem);
    sem.header.semanticId = NVTX_SEMANTIC_ID_COUNTERS_V1;
    sem.header.version = NVTX_COUNTER_SEMANTIC_VERSION;
    sem.flags = NVTX_COUNTER_FLAG_VALUETYPE_DELTA;
    sem.unit = "tokens";
    sem.unitScaleNumerator = 1;
    sem.unitScaleDenominator = 1;
    nvtxCounterAttr_t attr = {};
    attr.structSize = sizeof(attr);
    attr.name = "tokens";
    attr.counterId = NVTX_COUNTER_ID_NONE;
    attr.semantics = &sem.header;
    const uint64_t counter = nvtxCounterRegister(domain, &attr);

    float* buf = nullptr;
    cudaMalloc(&buf, 64 * sizeof(float));
    // Warm the context and JIT outside the timed region.
    for (int i = 0; i < 200; ++i) decode_step<<<1, 64>>>(buf, inner);
    cudaDeviceSynchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < iters; ++i) {
        decode_step<<<1, 64>>>(buf, inner);
        cudaDeviceSynchronize();
        nvtxCounterSampleInt64(domain, counter, tokens_per_step);
    }
    const auto t1 = std::chrono::steady_clock::now();
    cudaFree(buf);

    const double secs =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count();
    std::printf("WL iters=%ld secs=%.3f iters_per_sec=%.1f tokens_per_sec=%.0f\n",
                iters, secs, iters / secs,
                iters / secs * tokens_per_step);
    return 0;
}
