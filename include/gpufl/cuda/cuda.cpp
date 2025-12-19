#define GPUFL_EXPORTS
#include "gpufl/cuda/cuda.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/logger.hpp"

namespace gpufl::cuda {
    std::string dim3ToString(const dim3 v) {
        std::ostringstream oss;
        oss << "(" << v.x << "," << v.y << "," << v.z << ")";
        return oss.str();
    }

    const cudaDeviceProp& getDevicePropsCached(const int deviceId) {
        static std::mutex mu;
        static std::vector<cudaDeviceProp> cache;
        std::lock_guard<std::mutex> lk(mu);

        // Resize cache if needed (assuming max 32 GPUs to be safe)
        if (cache.empty()) cache.resize(32, {0});

        if (deviceId < 0 || deviceId >= 32) return cache[0];

        if (cache[deviceId].name[0] == 0) {
            cudaGetDeviceProperties(&cache[deviceId], deviceId);
        }
        return cache[deviceId];
    }
    KernelMonitor::KernelMonitor(std::string name,
                                 std::string tag,
                                 std::string grid, std::string block,
                                 const int dynShared, const int numRegs,
                                 const size_t staticShared, const size_t localBytes, const size_t constBytes,
                                 float occupancy, int maxActiveBlocks)
        : name_(std::move(name)), pid_(detail::getPid()), startTs_(detail::getTimestampNs()), tag_(std::move(tag)) {

        Runtime* rt = runtime();
        if (!rt || !rt->logger) return;

        KernelBeginEvent e;
        e.pid = pid_;
        e.app = rt->appName;
        e.name = name_;
        e.tag = std::move(tag);
        e.tsStartNs = startTs_; // Maps to user's 'tsStartNs'

        // Populate Launch Params
        e.grid = std::move(grid);
        e.block = std::move(block);
        e.dynSharedBytes = dynShared;
        e.numRegs = numRegs;
        e.staticSharedBytes = staticShared;
        e.localBytes = localBytes;
        e.constBytes = constBytes;

        e.occupancy = occupancy;
        e.maxActiveBlocks = maxActiveBlocks;

        if (rt->collector) e.devices = rt->collector->sampleAll();
        if (rt->hostCollector) e.host = rt->hostCollector->sample();

        rt->logger->logKernelBegin(e);
    }

    KernelMonitor::~KernelMonitor() {
        Runtime* rt = gpufl::runtime();
        if (!rt || !rt->logger) return;
        KernelEndEvent e;
        e.pid = pid_;
        e.app = rt->appName;
        e.name = name_;
        e.tsNs = detail::getTimestampNs();
        e.tag = tag_;
        e.devices = rt->collector ? rt->collector->sampleAll() : std::vector<DeviceSample>();
        rt->logger->logKernelEnd(e);
    }
}