#include "gpufl/cuda/cuda.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/logger.hpp"
#include <sstream>

namespace gpufl::cuda {

    std::string dim3ToString(dim3 v) {
        std::ostringstream oss;
        oss << "(" << v.x << "," << v.y << "," << v.z << ")";
        return oss.str();
    }

    void logKernelEvent(const std::string& kernelName,
                    int64_t ts_start_ns,
                    int64_t ts_end_ns,
                    dim3 grid,
                    dim3 block,
                    int dyn_shared_bytes,
                    const std::string& cuda_error,
                    const cudaFuncAttributes& attrs,
                    const std::string& tag) {
        Runtime* rt = gpufl::runtime();
        if (!rt || !rt->logger) return;

        KernelEvent e;
        e.pid = gpufl::detail::getPid();
        e.app = rt->appName;
        e.name = kernelName;

        e.tsStartNs = ts_start_ns;
        e.tsEndNs = ts_end_ns;
        e.durationNs = (ts_end_ns >= ts_start_ns) ? (ts_end_ns - ts_start_ns) : 0;

        e.grid = gpufl::cuda::dim3ToString(grid);
        e.block = gpufl::cuda::dim3ToString(block);

        e.dynSharedBytes = dyn_shared_bytes;
        e.numRegs = attrs.numRegs;
        e.staticSharedBytes = attrs.sharedSizeBytes;
        e.localBytes = attrs.localSizeBytes;
        e.constBytes = attrs.constSizeBytes;
        e.cudaError = cuda_error.empty() ? "no error" : cuda_error;
        e.tag = tag;

        const std::string devicesJson = rt->collector
            ? rt->collector->devicesInventoryJson()
            : "[]";

        rt->logger->logKernel(e, devicesJson);
    }

    const char* getCudaErrorString(const cudaError_t error) {
        return ::cudaGetErrorString(error);
    }
}