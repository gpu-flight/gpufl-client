#pragma once

#include <cuda_runtime.h>
#include <cupti.h>
#include <string>
#include "gpufl/core/trace_type.hpp"

namespace gpufl {

    struct ActivityRecord {
        uint32_t deviceId;
        char name[128];
        TraceType type;
        cudaStream_t stream;
        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;
        int64_t cpuStartNs;
        int64_t apiStartNs;
        int64_t apiExitNs;
        int64_t durationNs;

        // Detailed metrics (optional)
        bool hasDetails;
        int gridX, gridY, gridZ;
        int blockX, blockY, blockZ;
        int dynShared;
        int staticShared;
        int localBytes;
        int constBytes;
        int numRegs;
        float occupancy;

        int maxActiveBlocks;
        unsigned int corrId;

        char sourceFile[256];
        uint32_t sourceLine;
        char functionName[256];
        uint32_t samplesCount;
        uint32_t stallReason;
        std::string reasonName;
        char deviceName[64]{};

        // SASS Metrics support
        uint32_t pcOffset;
        uint64_t metricValue;
        char metricName[64];

        char userScope[256]{};
        int scopeDepth{};

        size_t stackId{};

        // Memcpy / Memset specific
        uint64_t bytes;
        uint32_t copyKind; // CUpti_ActivityMemcpyKind
        uint32_t srcKind;  // CUpti_ActivityMemoryKind
        uint32_t dstKind;  // CUpti_ActivityMemoryKind
    };

    struct LaunchMeta {
        int64_t apiEnterNs = 0;
        int64_t apiExitNs  = 0;
        bool hasDetails = false;
        int gridX=0, gridY=0, gridZ=0;
        int blockX=0, blockY=0, blockZ=0;
        int dynShared=0, staticShared=0, localBytes=0, constBytes=0, numRegs=0;
        float occupancy=0.0f;
        int maxActiveBlocks=0;
        char name[128]{};
        char userScope[256]{};
        int scopeDepth{};
        size_t stackId{};
    };

    class ICuptiHandler {
    public:
        virtual ~ICuptiHandler() = default;
        virtual bool shouldHandle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) const = 0;
        virtual void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) = 0;
        virtual const char* getName() const = 0;
    };

} // namespace gpufl
