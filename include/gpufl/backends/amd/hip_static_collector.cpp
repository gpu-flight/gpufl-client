#if !(GPUFL_ENABLE_AMD && GPUFL_HAS_HIP)
#error \
    "hip_static_collector.cpp should only be compiled when GPUFL_ENABLE_AMD && GPUFL_HAS_HIP are true."
#endif

#include "gpufl/backends/amd/hip_static_collector.hpp"

#include <iomanip>
#include <sstream>

#include <hip/hip_runtime_api.h>

namespace gpufl::amd {
namespace {

std::string HipErrorToString(const hipError_t status) {
    const char* text = hipGetErrorString(status);
    return text ? std::string(text) : std::string("unknown HIP error");
}

std::string UuidToString(const hipUUID& uuid) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (unsigned char byte : uuid.bytes) {
        oss << std::setw(2) << static_cast<int>(byte);
    }
    return oss.str();
}

}  // namespace

bool HipStaticCollector::IsAvailable(std::string* reason) {
    if (reason) reason->clear();

    int count = 0;
    const hipError_t status = hipGetDeviceCount(&count);
    if (status != hipSuccess) {
        if (reason) *reason = "hipGetDeviceCount failed: " + HipErrorToString(status);
        (void)hipGetLastError();
        return false;
    }

    if (count <= 0) {
        if (reason) *reason = "HIP runtime initialized but found no devices.";
        return false;
    }

    return true;
}

HipStaticCollector::HipStaticCollector() = default;
HipStaticCollector::~HipStaticCollector() = default;

std::vector<GpuStaticDeviceInfo> HipStaticCollector::sampleAll() {
    std::vector<GpuStaticDeviceInfo> devices;

    int count = 0;
    const hipError_t countStatus = hipGetDeviceCount(&count);
    if (countStatus != hipSuccess || count <= 0) {
        (void)hipGetLastError();
        return devices;
    }

    devices.reserve(count);
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, i) != hipSuccess) {
            (void)hipGetLastError();
            continue;
        }

        GpuStaticDeviceInfo info{};
        info.id = i;
        info.name = prop.name;
        info.uuid = UuidToString(prop.uuid);
        info.vendor = "AMD";
        info.architecture = prop.gcnArchName;
        info.compute_major = prop.major;
        info.compute_minor = prop.minor;
        info.shared_mem_per_block = static_cast<int>(prop.sharedMemPerBlock);
        info.regs_per_block = prop.regsPerBlock;
        info.multi_processor_count = prop.multiProcessorCount;
        info.warp_size = prop.warpSize;

        devices.push_back(std::move(info));
    }

    return devices;
}

}  // namespace gpufl::amd
