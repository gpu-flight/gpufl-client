#include "info_command.hpp"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

#include "gpufl/core/backend_factory.hpp"
#include "gpufl/core/json/json.hpp"
#include "gpufl/core/version.hpp"

#if GPUFL_HAS_CUDA
#include "gpufl/backends/nvidia/cuda_collector.hpp"
#endif

namespace gpufl::launcher {
namespace {

std::string architectureLabel(const GpuStaticDeviceInfo& device) {
    if (!device.architecture.empty()) return device.architecture;
    if (device.vendor == "NVIDIA" && device.compute_major > 0) {
        return "sm_" + std::to_string(device.compute_major) +
               std::to_string(device.compute_minor);
    }
    return {};
}

std::string formatCudaVersion(const int version) {
    if (version <= 0) return "n/a";
    std::ostringstream out;
    out << version / 1000 << '.' << (version % 1000) / 10;
    return out.str();
}

std::string formatBytes(const uint64_t bytes) {
    if (bytes == 0) return "n/a";
    constexpr double kib = 1024.0;
    constexpr double mib = kib * 1024.0;
    constexpr double gib = mib * 1024.0;

    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<uint64_t>(gib)) {
        out << static_cast<double>(bytes) / gib << " GiB";
    } else if (bytes >= static_cast<uint64_t>(mib)) {
        out << static_cast<double>(bytes) / mib << " MiB";
    } else if (bytes >= static_cast<uint64_t>(kib)) {
        out << static_cast<double>(bytes) / kib << " KiB";
    } else {
        out.unsetf(std::ios::floatfield);
        out << bytes << " B";
    }
    return out.str();
}

std::string formatKhz(const int khz) {
    if (khz <= 0) return "n/a";
    std::ostringstream out;
    out << std::fixed << std::setprecision(0)
        << static_cast<double>(khz) / 1000.0 << " MHz";
    return out.str();
}

std::string dimensions(const std::array<int, 3>& values) {
    if (values[0] <= 0 && values[1] <= 0 && values[2] <= 0) return "n/a";
    return std::to_string(values[0]) + " x " + std::to_string(values[1]) +
           " x " + std::to_string(values[2]);
}

void appendNullableInt(std::ostringstream& out, const int value) {
    if (value > 0) out << value;
    else out << "null";
}

void appendNullableUint64(std::ostringstream& out, const uint64_t value) {
    if (value > 0) out << value;
    else out << "null";
}

void appendDimensions(std::ostringstream& out,
                      const std::array<int, 3>& values) {
    if (values[0] <= 0 && values[1] <= 0 && values[2] <= 0) {
        out << "null";
        return;
    }
    out << '[' << values[0] << ',' << values[1] << ',' << values[2] << ']';
}

void appendDeviceJson(std::ostringstream& out,
                      const GpuStaticDeviceInfo& device) {
    const std::string architecture = architectureLabel(device);
    out << '{'
        << "\"deviceId\":" << device.id
        << ",\"name\":\"" << json::escape(device.name) << '"'
        << ",\"uuid\":\"" << json::escape(device.uuid) << '"'
        << ",\"vendor\":\"" << json::escape(device.vendor) << '"'
        << ",\"architecture\":";
    if (architecture.empty()) out << "null";
    else out << '"' << json::escape(architecture) << '"';

    out << ",\"computeCapability\":{"
        << "\"major\":" << device.compute_major
        << ",\"minor\":" << device.compute_minor << '}'
        << ",\"memory\":{"
        << "\"totalGlobalBytes\":";
    appendNullableUint64(out, device.total_global_mem);
    out << ",\"totalConstantBytes\":";
    appendNullableUint64(out, device.total_const_mem);
    out << ",\"l2CacheBytes\":";
    appendNullableInt(out, device.l2_cache_size);
    out << ",\"sharedPerBlockBytes\":";
    appendNullableInt(out, device.shared_mem_per_block);
    out << ",\"sharedPerBlockOptInBytes\":";
    appendNullableInt(out, device.shared_mem_per_block_optin);
    out << ",\"sharedPerSmBytes\":";
    appendNullableInt(out, device.shared_mem_per_multiprocessor);
    out << ",\"registersPerBlock\":";
    appendNullableInt(out, device.regs_per_block);
    out << ",\"registersPerSm\":";
    appendNullableInt(out, device.regs_per_multiprocessor);
    out << ",\"memoryClockRateKhz\":";
    appendNullableInt(out, device.memory_clock_rate_khz);
    out << ",\"memoryBusWidthBits\":";
    appendNullableInt(out, device.memory_bus_width_bits);

    out << "},\"executionLimits\":{"
        << "\"smCount\":";
    appendNullableInt(out, device.multi_processor_count);
    out << ",\"warpSize\":";
    appendNullableInt(out, device.warp_size);
    out << ",\"maxThreadsPerBlock\":";
    appendNullableInt(out, device.max_threads_per_block);
    out << ",\"maxThreadsPerSm\":";
    appendNullableInt(out, device.max_threads_per_multiprocessor);
    out << ",\"maxBlocksPerSm\":";
    appendNullableInt(out, device.max_blocks_per_multiprocessor);
    out << ",\"maxBlockDimensions\":";
    appendDimensions(out, device.max_threads_dim);
    out << ",\"maxGridDimensions\":";
    appendDimensions(out, device.max_grid_size);
    out << ",\"clockRateKhz\":";
    appendNullableInt(out, device.clock_rate_khz);
    out << ",\"asyncEngineCount\":";
    appendNullableInt(out, device.async_engine_count);

    out << "},\"features\":{"
        << "\"concurrentKernels\":"
        << (device.concurrent_kernels ? "true" : "false")
        << ",\"cooperativeLaunch\":"
        << (device.cooperative_launch ? "true" : "false")
        << ",\"unifiedAddressing\":"
        << (device.unified_addressing ? "true" : "false")
        << ",\"managedMemory\":"
        << (device.managed_memory ? "true" : "false")
        << ",\"memoryPools\":"
        << (device.memory_pools_supported ? "true" : "false")
        << ",\"clusterLaunch\":"
        << (device.cluster_launch ? "true" : "false")
        << ",\"tensorMapAccess\":"
        << (device.tensor_map_access_supported ? "true" : "false")
        << "}}";
}

InfoReport collectInfoReport(std::string* reason) {
    InfoReport report;
    BackendCollectors collectors = CreateBackendCollectors(BackendKind::Auto, reason);
    if (collectors.static_info_collector) {
        report.devices = collectors.static_info_collector->sampleStaticInfo();
    }

#if GPUFL_HAS_CUDA
    const bool has_nvidia = std::any_of(
        report.devices.begin(), report.devices.end(),
        [](const GpuStaticDeviceInfo& device) { return device.vendor == "NVIDIA"; });
    if (has_nvidia) {
        nvidia::EnrichCudaInfoCapabilities(report.devices);
        const nvidia::CudaRuntimeVersions versions =
            nvidia::QueryCudaRuntimeVersions();
        report.cuda_driver_version = versions.driver;
        report.cuda_runtime_version = versions.runtime;
    }
#endif
    return report;
}

}  // namespace

std::string renderInfoJson(const InfoReport& report) {
    std::ostringstream out;
    out << "{\"schemaVersion\":" << report.schema_version
        << ",\"gpuflVersion\":\"" << json::escape(kClientVersion) << '"'
        << ",\"wireVersion\":\"" << json::escape(kWireVersion) << '"'
        << ",\"cudaDriverVersion\":";
    appendNullableInt(out, report.cuda_driver_version);
    out << ",\"cudaRuntimeVersion\":";
    appendNullableInt(out, report.cuda_runtime_version);
    out << ",\"devices\":[";
    for (size_t i = 0; i < report.devices.size(); ++i) {
        if (i > 0) out << ',';
        appendDeviceJson(out, report.devices[i]);
    }
    out << "]}";
    return out.str();
}

std::string renderInfoText(const InfoReport& report) {
    std::ostringstream out;
    out << "GPUFlight " << kClientVersion << " (wire v" << kWireVersion << ")\n"
        << "CUDA driver:  " << formatCudaVersion(report.cuda_driver_version) << '\n'
        << "CUDA runtime: " << formatCudaVersion(report.cuda_runtime_version) << '\n'
        << "Devices:      " << report.devices.size() << "\n";

    for (const GpuStaticDeviceInfo& device : report.devices) {
        const std::string architecture = architectureLabel(device);
        out << "\nGPU " << device.id << ": " << device.name << '\n'
            << "  Vendor:                 " << device.vendor << '\n'
            << "  UUID:                   "
            << (device.uuid.empty() ? "n/a" : device.uuid) << '\n'
            << "  Architecture:           "
            << (architecture.empty() ? "n/a" : architecture) << '\n'
            << "  Compute capability:     " << device.compute_major << '.'
            << device.compute_minor << '\n'
            << "\n  Core and execution\n"
            << "    SM count:             " << device.multi_processor_count << '\n'
            << "    Warp size:            " << device.warp_size << '\n'
            << "    Max threads/block:    " << device.max_threads_per_block << '\n'
            << "    Max threads/SM:       "
            << device.max_threads_per_multiprocessor << '\n'
            << "    Max blocks/SM:        "
            << device.max_blocks_per_multiprocessor << '\n'
            << "    Max block dimensions: " << dimensions(device.max_threads_dim) << '\n'
            << "    Max grid dimensions:  " << dimensions(device.max_grid_size) << '\n'
            << "    Core clock:           " << formatKhz(device.clock_rate_khz) << '\n'
            << "    Async engines:        " << device.async_engine_count << '\n'
            << "\n  Memory and cache\n"
            << "    Global memory:        " << formatBytes(device.total_global_mem) << '\n'
            << "    Constant memory:      " << formatBytes(device.total_const_mem) << '\n'
            << "    L2 cache:             "
            << formatBytes(static_cast<uint64_t>(device.l2_cache_size)) << '\n'
            << "    Shared/block:         "
            << formatBytes(static_cast<uint64_t>(device.shared_mem_per_block)) << '\n'
            << "    Shared/block opt-in:  "
            << formatBytes(static_cast<uint64_t>(device.shared_mem_per_block_optin)) << '\n'
            << "    Shared/SM:            "
            << formatBytes(static_cast<uint64_t>(device.shared_mem_per_multiprocessor)) << '\n'
            << "    Registers/block:      " << device.regs_per_block << '\n'
            << "    Registers/SM:         " << device.regs_per_multiprocessor << '\n'
            << "    Memory clock:         "
            << formatKhz(device.memory_clock_rate_khz) << '\n'
            << "    Memory bus width:     " << device.memory_bus_width_bits << " bits\n"
            << "\n  Features\n"
            << "    Concurrent kernels:   "
            << (device.concurrent_kernels ? "yes" : "no") << '\n'
            << "    Cooperative launch:   "
            << (device.cooperative_launch ? "yes" : "no") << '\n'
            << "    Unified addressing:   "
            << (device.unified_addressing ? "yes" : "no") << '\n'
            << "    Managed memory:       "
            << (device.managed_memory ? "yes" : "no") << '\n'
            << "    Async memory pools:   "
            << (device.memory_pools_supported ? "yes" : "no") << '\n'
            << "    Cluster launch:       "
            << (device.cluster_launch ? "yes" : "no") << '\n'
            << "    Tensor Map access:    "
            << (device.tensor_map_access_supported ? "yes" : "no") << '\n';
    }
    return out.str();
}

int runInfo(const InfoArgs& args) {
    std::string reason;
    InfoReport report = collectInfoReport(&reason);
    if (report.devices.empty()) {
        std::fprintf(stderr, "gpufl info: no GPU devices were detected%s%s\n",
                     reason.empty() ? "" : ": ", reason.c_str());
        return 3;
    }

    if (args.device_id) {
        const int requested = *args.device_id;
        const auto it = std::find_if(
            report.devices.begin(), report.devices.end(),
            [requested](const GpuStaticDeviceInfo& device) {
                return device.id == requested;
            });
        if (it == report.devices.end()) {
            std::fprintf(stderr, "gpufl info: device %d was not found\n", requested);
            return 2;
        }
        report.devices = {*it};
    }

    const std::string output = args.json ? renderInfoJson(report)
                                         : renderInfoText(report);
    std::fputs(output.c_str(), stdout);
    std::fputc('\n', stdout);
    return 0;
}

}  // namespace gpufl::launcher
