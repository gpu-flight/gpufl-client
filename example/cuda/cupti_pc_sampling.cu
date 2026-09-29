#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pcsampling.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <set>
#include <thread>
#include <string>
#include <utility>
#include <vector>
#include <cupti_pcsampling.h>

namespace {

void checkCuda(CUresult result, const char* what) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char* err = nullptr;
    cuGetErrorString(result, &err);
    std::fprintf(stderr, "CUDA error %s: %s\n", what, err ? err : "unknown");
    std::exit(1);
}

void checkCupti(CUptiResult result, const char* what) {
    if (result == CUPTI_SUCCESS) {
        return;
    }
    const char* err = nullptr;
    cuptiGetResultString(result, &err);
    std::fprintf(stderr, "CUPTI error %s: %s\n", what, err ? err : "unknown");
    std::exit(1);
}

void checkCudaRuntime(cudaError_t result, const char* what) {
    if (result == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "CUDA runtime error %s: %s\n", what, cudaGetErrorString(result));
    std::exit(1);
}

__global__ void busyKernel(int* data, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int v = data[idx];
        for (int i = 0; i < iters; ++i) {
            v = v * 2 + 1;
        }
        data[idx] = v;
    }
}

CUcontext ensureContext() {
    checkCuda(cuInit(0), "cuInit");

    CUcontext ctx = nullptr;
    checkCuda(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
    if (ctx) {
        return ctx;
    }

    CUdevice dev = 0;
    checkCuda(cuDeviceGet(&dev, 0), "cuDeviceGet");
    checkCuda(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    checkCuda(cuCtxPushCurrent(ctx), "cuCtxPushCurrent");
    return ctx;
}

struct PCSamplingBuffers {
    CUpti_PCSamplingData* data;
    CUpti_PCSamplingPCData* pcRecords;
};

void freePCSamplingBuffers(PCSamplingBuffers* buffers) {
    for (size_t i = 0; i < buffers->data->collectNumPcs; ++i) {
        std::free(buffers->pcRecords[i].stallReason);
    }
    std::free(buffers->pcRecords);
    std::free(buffers->data);
    std::free(buffers);
}

PCSamplingBuffers* configurePCSampling(CUcontext ctx) {
    const size_t kMaxPcs = 65536;
    PCSamplingBuffers* buffers = static_cast<PCSamplingBuffers*>(std::calloc(1, sizeof(PCSamplingBuffers)));
    if (!buffers) {
        std::fprintf(stderr, "Failed to allocate PCSamplingBuffers.\n");
        std::exit(1);
    }
    buffers->pcRecords = static_cast<CUpti_PCSamplingPCData*>(
        std::calloc(kMaxPcs, sizeof(CUpti_PCSamplingPCData)));
    if (!buffers->pcRecords) {
        std::fprintf(stderr, "Failed to allocate PC sampling records.\n");
        std::exit(1);
    }
    for (size_t i = 0; i < kMaxPcs; ++i) {
        buffers->pcRecords[i].size = sizeof(CUpti_PCSamplingPCData);
        buffers->pcRecords[i].stallReasonCount = 128;
        buffers->pcRecords[i].stallReason = static_cast<CUpti_PCSamplingStallReason*>(
            std::calloc(128, sizeof(CUpti_PCSamplingStallReason)));
    }

    buffers->data = static_cast<CUpti_PCSamplingData*>(
        std::calloc(1, sizeof(CUpti_PCSamplingData)));
    if (!buffers->data) {
        std::fprintf(stderr, "Failed to allocate PC sampling data.\n");
        std::exit(1);
    }
    buffers->data->size = sizeof(CUpti_PCSamplingData);
    buffers->data->collectNumPcs = kMaxPcs;
    buffers->data->pPcData = buffers->pcRecords;

    return buffers;
}

}  // namespace

int main() {
    checkCudaRuntime(cudaSetDevice(0), "cudaSetDevice");
    checkCudaRuntime(cudaFree(nullptr), "cudaFree (context init)");

    CUcontext ctx = ensureContext();

    // `buffers` becomes the SAMPLING_DATA_BUFFER. In KERNEL_SERIALIZED mode
    // CUPTI moves every finished kernel's records into it by itself, so
    // GetData needs a buffer of its own or it overwrites them.
    PCSamplingBuffers* buffers = configurePCSampling(ctx);
    PCSamplingBuffers* readBuffers = configurePCSampling(ctx);

    printf("Enabling PC sampling...\n"); fflush(stdout);
    CUpti_PCSamplingEnableParams enableParams = {};
    enableParams.size = sizeof(CUpti_PCSamplingEnableParams);
    enableParams.ctx = ctx;
    checkCupti(cuptiPCSamplingEnable(&enableParams), "cuptiPCSamplingEnable");

    // Re-set ALL attributes after enable, just in case
    CUpti_PCSamplingConfigurationInfo postEnableInfo[10] = {};
    postEnableInfo[0].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
    postEnableInfo[0].attributeData.collectionModeData.collectionMode = CUPTI_PC_SAMPLING_COLLECTION_MODE_KERNEL_SERIALIZED;
    postEnableInfo[1].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
    postEnableInfo[1].attributeData.samplingPeriodData.samplingPeriod = 10; // 2^10 = 1024 cycles
    postEnableInfo[2].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SCRATCH_BUFFER_SIZE;
    postEnableInfo[2].attributeData.scratchBufferSizeData.scratchBufferSize = 256 * 1024 * 1024;
    postEnableInfo[3].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE;
    postEnableInfo[3].attributeData.hardwareBufferSizeData.hardwareBufferSize = 256 * 1024 * 1024;
    postEnableInfo[4].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL;
    postEnableInfo[4].attributeData.enableStartStopControlData.enableStartStopControl = 1;
    postEnableInfo[5].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
    postEnableInfo[5].attributeData.samplingDataBufferData.samplingDataBuffer = buffers->data;
    postEnableInfo[6].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_OUTPUT_DATA_FORMAT;
    postEnableInfo[6].attributeData.outputDataFormatData.outputDataFormat = CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED;

    CUpti_PCSamplingConfigurationInfoParams postConfigParams = {};
    postConfigParams.size = sizeof(CUpti_PCSamplingConfigurationInfoParams);
    postConfigParams.ctx = ctx;
    postConfigParams.numAttributes = 7;
    postConfigParams.pPCSamplingConfigurationInfo = postEnableInfo;
    checkCupti(cuptiPCSamplingSetConfigurationAttribute(&postConfigParams),
               "cuptiPCSamplingSetConfigurationAttribute (post-enable)");

    for (size_t i = 0; i < 7; ++i) {
        if (postEnableInfo[i].attributeStatus != CUPTI_SUCCESS) {
            printf("Attribute %d (type %d) failed with status %d\n", (int)i, (int)postEnableInfo[i].attributeType, (int)postEnableInfo[i].attributeStatus);
        }
    }

    printf("Starting PC sampling...\n"); fflush(stdout);
    CUpti_PCSamplingStartParams startParams = {};
    startParams.size = sizeof(CUpti_PCSamplingStartParams);
    startParams.ctx = ctx;
    checkCupti(cuptiPCSamplingStart(&startParams), "cuptiPCSamplingStart");


    const int n = 1 << 20;
    int* data = nullptr;
    checkCudaRuntime(cudaMalloc(&data, n * sizeof(int)), "cudaMalloc");
    checkCudaRuntime(cudaMemset(data, 0, n * sizeof(int)), "cudaMemset");

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    busyKernel<<<grid, block>>>(data, n, 256);
    checkCudaRuntime(cudaDeviceSynchronize(), "cudaDeviceSynchronize (warmup)");

    for (int i = 0; i < 2000; ++i) {
        busyKernel<<<grid, block>>>(data, n, 2048);
    }
    checkCudaRuntime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    checkCudaRuntime(cudaFree(data), "cudaFree");

    printf("Stopping PC sampling...\n"); fflush(stdout);
    CUpti_PCSamplingStopParams stopParams = {};
    stopParams.size = sizeof(CUpti_PCSamplingStopParams);
    stopParams.ctx = ctx;
    checkCupti(cuptiPCSamplingStop(&stopParams), "cuptiPCSamplingStop");


    std::fprintf(stdout, "Getting PC sampling data...\n"); std::fflush(stdout);
    CUpti_PCSamplingGetDataParams getDataParams = {};
    getDataParams.size = sizeof(CUpti_PCSamplingGetDataParams);
    getDataParams.ctx = ctx;
    getDataParams.pcSamplingData = readBuffers->data;

    // Each call moves records out of the configured buffer (one set per
    // kernel launch), then the per-PC overflow of launches that did not fit.
    // Done when a call returns nothing.
    std::map<std::pair<std::string, uint64_t>, uint64_t> samplesByPc;
    std::set<uint32_t> correlationIds;
    size_t recordCount = 0;
    unsigned long long totalSamples = 0;
    for (;;) {
        for (size_t i = 0; i < readBuffers->data->collectNumPcs; ++i) {
            readBuffers->pcRecords[i].stallReasonCount = 128;
        }
        readBuffers->data->totalNumPcs = 0;
        checkCupti(cuptiPCSamplingGetData(&getDataParams), "cuptiPCSamplingGetData");
        const size_t pcCount = readBuffers->data->totalNumPcs;
        if (pcCount == 0) {
            break;
        }
        recordCount += pcCount;
        totalSamples += readBuffers->data->totalSamples;
        for (size_t i = 0; i < pcCount; ++i) {
            // Copies: the CUPTI structs are packed, and GCC refuses to bind
            // packed fields to references.
            const CUpti_PCSamplingPCData& pc = readBuffers->data->pPcData[i];
            const std::string functionName = pc.functionName ? pc.functionName : "<unknown>";
            const uint64_t pcOffset = pc.pcOffset;
            const uint32_t correlationId = pc.correlationId;
            uint64_t samples = 0;
            for (size_t j = 0; j < pc.stallReasonCount; ++j) {
                samples += pc.stallReason[j].samples;
            }
            samplesByPc[{functionName, pcOffset}] += samples;
            correlationIds.insert(correlationId);
        }
    }

    std::fprintf(stdout, "Disabling PC sampling...\n"); std::fflush(stdout);
    CUpti_PCSamplingDisableParams disableParams = {};
    disableParams.size = sizeof(CUpti_PCSamplingDisableParams);
    disableParams.ctx = ctx;
    checkCupti(cuptiPCSamplingDisable(&disableParams), "cuptiPCSamplingDisable");

    // Launches past the configured buffer share one correlation id.
    std::fprintf(stdout, "Collected %zu PC records, %zu correlation ids (totalSamples=%llu)\n",
                 recordCount, correlationIds.size(), totalSamples);
    for (const auto& [pc, samples] : samplesByPc) {
        std::fprintf(stdout, "  %s pcOffset=0x%llx samples=%llu\n", pc.first.c_str(),
                     static_cast<unsigned long long>(pc.second),
                     static_cast<unsigned long long>(samples));
    }

    freePCSamplingBuffers(readBuffers);
    freePCSamplingBuffers(buffers);

    std::fprintf(stdout, "PC sampling stopped.\n");
    return 0;
}
