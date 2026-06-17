#if !(GPUFL_ENABLE_METAL && GPUFL_HAS_METAL)
#error "metal_collector.mm requires GPUFL_ENABLE_METAL && GPUFL_HAS_METAL"
#endif

#include "gpufl/backends/metal/metal_collector.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace gpufl::metal {
namespace {

constexpr uint64_t kMiB = 1024ULL * 1024ULL;

std::string NSStringToString(NSString* value) {
    if (value == nil) return {};
    const char* utf8 = [value UTF8String];
    return utf8 ? std::string(utf8) : std::string{};
}

std::string RegistryUuid(id<MTLDevice> device) {
    std::ostringstream oss;
    oss << "metal-registry-0x" << std::hex << std::setw(16)
        << std::setfill('0') << [device registryID];
    return oss.str();
}

DeviceSample DeviceToSample(id<MTLDevice> device, int id) {
    DeviceSample sample{};
    sample.device_id = id;
    sample.name = NSStringToString([device name]);
    sample.uuid = RegistryUuid(device);
    sample.vendor = "Apple";

    const uint64_t totalBytes = [device recommendedMaxWorkingSetSize];
    const uint64_t usedBytes = [device currentAllocatedSize];
    sample.total_mib = static_cast<size_t>(totalBytes / kMiB);
    sample.used_mib = static_cast<size_t>(usedBytes / kMiB);
    sample.free_mib =
        sample.total_mib > sample.used_mib ? sample.total_mib - sample.used_mib : 0;
    if (sample.total_mib > 0) {
        sample.mem_util = static_cast<unsigned int>(
            std::min<uint64_t>(100, (100ULL * sample.used_mib) / sample.total_mib));
    }

    return sample;
}

GpuStaticDeviceInfo DeviceToStaticInfo(id<MTLDevice> device, int id) {
    GpuStaticDeviceInfo info{};
    info.id = id;
    info.name = NSStringToString([device name]);
    info.uuid = RegistryUuid(device);
    info.vendor = "Apple";
    info.architecture = "Metal";
    return info;
}

}  // namespace

MetalCollector::MetalCollector() : available_(IsAvailable(nullptr)) {}
MetalCollector::~MetalCollector() = default;

std::vector<DeviceSample> MetalCollector::sampleAll() {
    std::vector<DeviceSample> samples;
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices != nil && [devices count] > 0) {
            samples.reserve([devices count]);
            int idx = 0;
            for (id<MTLDevice> device in devices) {
                if (device != nil) samples.push_back(DeviceToSample(device, idx++));
            }
            [devices release];
            return samples;
        }
        [devices release];

        id<MTLDevice> defaultDevice = MTLCreateSystemDefaultDevice();
        if (defaultDevice != nil) {
            samples.push_back(DeviceToSample(defaultDevice, 0));
            [(id)defaultDevice release];
        }
    }
    return samples;
}

std::vector<GpuStaticDeviceInfo> MetalCollector::sampleStaticInfo() {
    std::vector<GpuStaticDeviceInfo> devicesInfo;
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices != nil && [devices count] > 0) {
            devicesInfo.reserve([devices count]);
            int idx = 0;
            for (id<MTLDevice> device in devices) {
                if (device != nil) {
                    devicesInfo.push_back(DeviceToStaticInfo(device, idx++));
                }
            }
            [devices release];
            return devicesInfo;
        }
        [devices release];

        id<MTLDevice> defaultDevice = MTLCreateSystemDefaultDevice();
        if (defaultDevice != nil) {
            devicesInfo.push_back(DeviceToStaticInfo(defaultDevice, 0));
            [(id)defaultDevice release];
        }
    }
    return devicesInfo;
}

bool MetalCollector::IsAvailable(std::string* reason) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices != nil && [devices count] > 0) {
            [devices release];
            return true;
        }
        [devices release];

        id<MTLDevice> defaultDevice = MTLCreateSystemDefaultDevice();
        if (defaultDevice != nil) {
            [(id)defaultDevice release];
            return true;
        }
    }

    if (reason) *reason = "Metal framework is present but no Metal device was found.";
    return false;
}

}  // namespace gpufl::metal
