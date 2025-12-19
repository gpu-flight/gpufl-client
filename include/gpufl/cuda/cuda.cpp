#define GPUFL_EXPORTS
#include "gpufl/cuda/cuda.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/logger.hpp"

namespace gpufl::cuda {

    KernelMonitor::KernelMonitor(std::string name,
                                 std::string tag,
                                 std::string grid, std::string block,
                                 const int dynShared, const int numRegs,
                                 const size_t staticShared, const size_t localBytes, const size_t constBytes)
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