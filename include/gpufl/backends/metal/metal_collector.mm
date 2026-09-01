#if !(GPUFL_ENABLE_METAL && GPUFL_HAS_METAL)
#error "metal_collector.mm requires GPUFL_ENABLE_METAL && GPUFL_HAS_METAL"
#endif

#include "gpufl/backends/metal/metal_collector.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

std::string RegistryId(id<MTLDevice> device) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0')
        << [device registryID];
    return oss.str();
}

std::string RegistryUuid(id<MTLDevice> device) {
    return "metal-registry-" + RegistryId(device);
}

std::string DeviceLocationName(const MTLDeviceLocation location) {
    switch (location) {
        case MTLDeviceLocationBuiltIn: return "built_in";
        case MTLDeviceLocationSlot: return "slot";
        case MTLDeviceLocationExternal: return "external";
        case MTLDeviceLocationUnspecified: return "unspecified";
    }
    return "unknown";
}

void AppendSupportedGpuFamilies(
    id<MTLDevice> device, std::vector<std::string>& families) {
    if (@available(macOS 10.15, *)) {
        for (int family = 1; family <= 10; ++family) {
            if ([device supportsFamily:static_cast<MTLGPUFamily>(1000 + family)]) {
                families.push_back("apple" + std::to_string(family));
            }
        }
        if ([device supportsFamily:static_cast<MTLGPUFamily>(2002)]) {
            families.push_back("mac2");
        }
        for (int family = 1; family <= 3; ++family) {
            if ([device supportsFamily:static_cast<MTLGPUFamily>(3000 + family)]) {
                families.push_back("common" + std::to_string(family));
            }
        }
        for (int family = 3; family <= 4; ++family) {
            if ([device supportsFamily:static_cast<MTLGPUFamily>(4998 + family)]) {
                families.push_back("metal" + std::to_string(family));
            }
        }
    }
}

void AppendCounterSetNames(
    id<MTLDevice> device, std::vector<std::string>& counterSets) {
    if (@available(macOS 10.15, *)) {
        NSArray<id<MTLCounterSet>>* sets = [device counterSets];
        for (id<MTLCounterSet> set in sets) {
            const std::string name = NSStringToString([set name]);
            if (!name.empty()) counterSets.push_back(name);
        }
    }
}

DeviceSample DeviceToSample(id<MTLDevice> device, int id) {
    DeviceSample sample{};
    sample.device_id = id;
    sample.name = NSStringToString([device name]);
    sample.uuid = RegistryUuid(device);
    sample.vendor = "Apple";

    auto& capabilities = sample.telemetry_capabilities;
    capabilities.available = {
        "process_allocated_mib", "recommended_max_working_set_mib"};
    capabilities.unavailable = {
        "gpu_util", "mem_util", "temp_c", "power_mw", "used_mib",
        "total_mib", "clock_sm", "fan_speed_pct", "temp_mem_c",
        "temp_junction_c", "voltage_mv", "energy_uj", "clock_mem",
        "pcie_bw_bps", "ecc_corrected", "ecc_uncorrected"};
    capabilities.allocation_scope = "current_process";
    capabilities.process_allocated_mib =
        static_cast<uint64_t>([device currentAllocatedSize]) / kMiB;
    capabilities.recommended_max_working_set_mib =
        [device recommendedMaxWorkingSetSize] / kMiB;
    if (@available(macOS 10.15, *)) {
        capabilities.memory_model = [device hasUnifiedMemory]
                                        ? "unified"
                                        : "discrete";
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

    auto& metal = info.metal;
    metal.available = true;
    metal.registry_id = RegistryId(device);
    if (@available(macOS 14.0, *)) {
        metal.architecture_name =
            NSStringToString([[device architecture] name]);
        if (!metal.architecture_name.empty()) {
            info.architecture = metal.architecture_name;
        }
    }
    metal.low_power = [device isLowPower];
    metal.headless = [device isHeadless];
    metal.removable = [device isRemovable];
    metal.recommended_max_working_set_bytes =
        [device recommendedMaxWorkingSetSize];
    metal.max_buffer_length_bytes = [device maxBufferLength];

    const MTLSize threads = [device maxThreadsPerThreadgroup];
    metal.max_threads_per_threadgroup = {
        static_cast<uint64_t>(threads.width),
        static_cast<uint64_t>(threads.height),
        static_cast<uint64_t>(threads.depth)};

    if (@available(macOS 10.15, *)) {
        metal.unified_memory = [device hasUnifiedMemory];
        metal.location = DeviceLocationName([device location]);
        metal.location_number = [device locationNumber];
        metal.max_transfer_rate_bps = [device maxTransferRate];
    }
    AppendSupportedGpuFamilies(device, metal.gpu_families);
    AppendCounterSetNames(device, metal.counter_sets);
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
