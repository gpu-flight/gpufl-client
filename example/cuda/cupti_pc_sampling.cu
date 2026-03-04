#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_pcsampling.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <string>
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

struct PcSampleRecord {
    std::string functionName;
    uint64_t pcOffset;
    uint64_t samples;
    uint32_t correlationId;
};

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

    PCSamplingBuffers* buffers = configurePCSampling(ctx);

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
    getDataParams.pcSamplingData = buffers->data;
    checkCupti(cuptiPCSamplingGetData(&getDataParams), "cuptiPCSamplingGetData");

    std::fprintf(stdout, "Disabling PC sampling...\n"); std::fflush(stdout);
    CUpti_PCSamplingDisableParams disableParams = {};
    disableParams.size = sizeof(CUpti_PCSamplingDisableParams);
    disableParams.ctx = ctx;
    checkCupti(cuptiPCSamplingDisable(&disableParams), "cuptiPCSamplingDisable");
    std::fprintf(stdout, "PC sampling data: totalSamples=%llu dropped=%llu totalPCs=%zu remaining=%zu rangeId=%llu\n",
                    static_cast<unsigned long long>(buffers->data->totalSamples),
                    static_cast<unsigned long long>(buffers->data->droppedSamples),
                    buffers->data->totalNumPcs,
                    buffers->data->remainingNumPcs,
                    static_cast<unsigned long long>(buffers->data->rangeId));
    std::fflush(stdout);

    size_t pcCount = buffers->data->totalNumPcs;
    if (pcCount > buffers->data->collectNumPcs) {
        pcCount = buffers->data->collectNumPcs;
    }
    //
    std::vector<PcSampleRecord> records;
    records.reserve(pcCount);
    //
    for (size_t i = 0; i < pcCount; ++i) {
        const CUpti_PCSamplingPCData& pc = buffers->data->pPcData[i];
        uint64_t samples = 0;
        if (pc.stallReason) {
            for (size_t j = 0; j < pc.stallReasonCount; ++j) {
                samples += pc.stallReason[j].samples;
            }
        }
        PcSampleRecord rec;
        if (pc.functionName) {
            rec.functionName = pc.functionName;
        }
        rec.pcOffset = pc.pcOffset;
        rec.samples = samples;
        rec.correlationId = pc.correlationId;
        records.push_back(std::move(rec));
    }

     std::fprintf(stdout, "Collected %zu PC records\n", records.size());
     for (size_t i = 0; i < records.size(); ++i) {
         const PcSampleRecord& rec = records[i];
         std::fprintf(stdout, "  [%zu] %s pcOffset=0x%llx samples=%llu corr=%u\n",
                      i,
                      rec.functionName.empty() ? "<unknown>" : rec.functionName.c_str(),
                      static_cast<unsigned long long>(rec.pcOffset),
                      static_cast<unsigned long long>(rec.samples),
                      rec.correlationId);
     }

    size_t maxPcs = buffers->data->collectNumPcs;
    for (size_t i = 0; i < maxPcs; ++i) {
        if (buffers->pcRecords[i].stallReason) {
            std::free(buffers->pcRecords[i].stallReason);
        }
    }
    std::free(buffers->pcRecords);
    std::free(buffers->data);
    std::free(buffers);

    std::fprintf(stdout, "PC sampling stopped.\n");
    return 0;
}
