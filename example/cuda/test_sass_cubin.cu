#include <iostream>
#include <vector>
#include <map>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cupti.h>
#include <cupti_pcsampling.h>
#include <cupti_sass_metrics.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include "gpufl/backends/nvidia/sampler/cupti_sass.hpp"
#include "gpufl/core/debug_logger.hpp"

using namespace gpufl::nvidia;

// Global map to store captured cubins by their CRC
struct CubinInfo {
    std::vector<uint8_t> data;
    uint32_t crc;
};
std::map<uint32_t, CubinInfo> capturedCubins;

// CUPTI Resource Callback to capture cubins
void CUPTIAPI resourceCallback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata, CUpti_CallbackData *cbInfo) {
    if (domain == CUPTI_CB_DOMAIN_RESOURCE && 
        (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED || 
         cbid == CUPTI_CBID_RESOURCE_MODULE_PROFILED ||
         cbid == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING)) {
        auto *modData = static_cast<const CUpti_ModuleResourceData *>(cbdata);
        if (modData && modData->pCubin && modData->cubinSize > 0) {
            CUpti_GetCubinCrcParams params = {CUpti_GetCubinCrcParamsSize};
            params.cubinSize = modData->cubinSize;
            params.cubin = modData->pCubin;
            if (cuptiGetCubinCrc(&params) == CUPTI_SUCCESS) {
                auto& info = capturedCubins[params.cubinCrc];
                info.crc = params.cubinCrc;
                info.data.assign(reinterpret_cast<const uint8_t *>(modData->pCubin),
                               reinterpret_cast<const uint8_t *>(modData->pCubin) + modData->cubinSize);
                std::cout << "[CALLBACK] Captured Cubin with CRC: 0x" << std::hex << params.cubinCrc << std::dec 
                          << " Size: " << modData->cubinSize << " bytes" << std::endl;
            }

        }
         } else if (cbInfo->callbackSite == CUPTI_API_ENTER) {
             const char* img = nullptr;

             // Handle the two most common internal loaders
             if (cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData ||
                 cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx) {
                 auto params = (cuModuleLoadData_params *)(cbInfo->functionParams);
                 img = (const char*)params->image;
                 }
             else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary) {
                 auto params = (cuModuleLoadFatBinary_params *)(cbInfo->functionParams);
                 img = (const char*)params->fatCubin; // This is the start of the fatbin
             }

             if (img == nullptr) return;

             // --- THE PRINT LOGIC ---
             // 1. PTX Check (Starts with '.')
             if (img[0] == '.') {
                 printf("\n[SUCCESS] Captured PTX Source:\n%s\n", img);
             }
             // 2. Binary Check (Starts with ELF magic 0x7f)
             else if ((unsigned char)img[0] == 0x7f) {
                 printf("\n[INFO] Captured Binary CUBIN (SASS). Header: ELF\n");
             }
             // 3. Fatbinary Check (Starts with magic 0x11 0x47)
             else if ((unsigned char)img[0] == 0x11 && (unsigned char)img[1] == 0x47) {
                 printf("\n[INFO] Captured FATBINARY container. PTX is packed inside.\n");
             }

                      }
}

__global__ void simpleKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2 + 1;
    }
}

int main() {
    gpufl::DebugLogger::setEnabled(true);

    if (cuInit(0) != CUDA_SUCCESS) {
        std::cerr << "cuInit failed" << std::endl;
        return 1;
    }

    uint32_t deviceIndex = 0;
    cudaSetDevice(deviceIndex);

    // 1. Setup Resource Callback to capture cubin
    CUpti_SubscriberHandle subscriber;
    cuptiSubscribe(&subscriber, (CUpti_CallbackFunc) resourceCallback, nullptr);

    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_PROFILED);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING);

    // 2. Configure SASS metrics
    // None of these metrics return the cubin, but the SASS data struct includes cubinCrc.
    std::vector<std::string> metrics = {
        "smsp__sass_inst_executed",
        "smsp__sass_thread_inst_executed"
    };

    if (!CuptiSass::setMetricsConfig(deviceIndex, metrics)) {
        std::cerr << "Failed to set metrics config." << std::endl;
        return 1;
    }

    // 3. Enable Metrics
    if (!CuptiSass::enableMetrics()) {
        std::cerr << "Failed to enable metrics" << std::endl;
        return 1;
    }

    // 4. Launch kernel (this will trigger the resource callback)
    const int n = 1024;
    int *d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    simpleKernel<<<1, 256>>>(d_data, n);
    cudaDeviceSynchronize();

    // 5. Flush SASS metrics data
    auto records = CuptiSass::flushMetricsData();
    CuptiSass::disableMetrics();
    CuptiSass::unsetMetricsConfig(deviceIndex);

    std::cout << "\nAnalysis of SASS records with Cubin Correlation:" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    for (const auto& record : records) {
        std::cout << "Function: " << record.functionName 
                  << " (Cubin CRC: 0x" << std::hex << record.cubinCrc << ")"
                  << " PC Offset: 0x" << record.pcOffset << std::dec << std::endl;
        
        // Match with captured cubin
        if (capturedCubins.count(record.cubinCrc)) {
            const auto& info = capturedCubins[record.cubinCrc];
            std::cout << "  MATCHED with captured cubin (size: " << info.data.size() << " bytes)" << std::endl;
            
            // Try source correlation if possible (this uses cuptiGetSassToSourceCorrelation internally)
            auto source = CuptiSass::sampleSourceCorrelation(info.data.data(), info.data.size(), 
                                                           record.functionName.c_str(), record.pcOffset);
            if (!source.fileName.empty()) {
                std::cout << "  Source Correlation: " << source.fileName << ":" << source.lineNumber << std::endl;
            }
        } else {
            std::cout << "  NO cubin found for this CRC in local cache!" << std::endl;
        }

        for (const auto& val : record.values) {
            std::cout << "    Metric ID: " << val.metricId << " Value: " << val.value << std::endl;
        }
    }

    cuptiUnsubscribe(subscriber);
    cudaFree(d_data);
    return 0;
}
