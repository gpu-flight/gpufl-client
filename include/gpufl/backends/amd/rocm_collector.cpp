#if !(GPUFL_ENABLE_AMD && (GPUFL_HAS_ROCM_SMI || GPUFL_HAS_HIP))
#error \
    "rocm_collector.cpp should only be compiled when GPUFL_ENABLE_AMD and at least one AMD runtime capability is available."
#endif

#include "gpufl/backends/amd/rocm_collector.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include "gpufl/core/debug_logger.hpp"

#if GPUFL_HAS_HIP
#include <hip/hip_runtime_api.h>
#endif

#if GPUFL_HAS_ROCM_SMI
#include <rocm_smi/rocm_smi.h>
#endif

namespace gpufl::amd {
namespace {

constexpr uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr uint64_t kMicrowattsPerMilliwatt = 1000ull;
constexpr int64_t kMillidegreesPerDegree = 1000ll;

#if GPUFL_HAS_HIP
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

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool IsCpuLikeHipEntry(const hipDeviceProp_t& prop) {
    const std::string lowerName = ToLower(prop.name);
    return lowerName.find("ryzen") != std::string::npos ||
           lowerName.find("epyc") != std::string::npos ||
           lowerName.find("threadripper") != std::string::npos;
}

bool IsGpuHipDevice(const hipDeviceProp_t& prop) {
    const std::string arch = prop.gcnArchName;
    if (arch.empty() || arch.rfind("gfx", 0) != 0) {
        return false;
    }
    if (prop.multiProcessorCount <= 0 || prop.warpSize <= 0) {
        return false;
    }
    if (IsCpuLikeHipEntry(prop)) {
        return false;
    }
    return true;
}

GpuStaticDeviceInfo BuildStaticDeviceInfo(const int deviceId,
                                          const hipDeviceProp_t& prop) {
    GpuStaticDeviceInfo info{};
    info.id = deviceId;
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
    return info;
}

std::pair<bool, int> ProbeHipDeviceCount(std::string* reason) {
    if (reason) reason->clear();

    int count = 0;
    const hipError_t status = hipGetDeviceCount(&count);
    if (status != hipSuccess) {
        if (reason) *reason = "hipGetDeviceCount failed: " + HipErrorToString(status);
        (void)hipGetLastError();
        return {false, 0};
    }
    if (count <= 0) {
        if (reason) *reason = "HIP runtime initialized but found no devices.";
        return {false, 0};
    }
    return {true, count};
}
#endif

#if GPUFL_HAS_ROCM_SMI
std::string RsmiStatusToString(const rsmi_status_t status) {
    const char* text = nullptr;
    if (rsmi_status_string(status, &text) == RSMI_STATUS_SUCCESS && text) {
        return text;
    }
    return "unknown ROCm SMI error";
}

bool IsSuccess(const rsmi_status_t status) {
    return status == RSMI_STATUS_SUCCESS;
}

size_t ToMiB(const uint64_t bytes) {
    return static_cast<size_t>(bytes / kBytesPerMiB);
}

unsigned int ToMHz(const rsmi_frequencies_t& freqs) {
    if (freqs.num_supported == 0 || freqs.current >= freqs.num_supported) {
        return 0;
    }
    return static_cast<unsigned int>(freqs.frequency[freqs.current] / 1000000ull);
}

int DecodePciBusId(const uint64_t bdfid) {
    return static_cast<int>((bdfid >> 8) & 0xffu);
}

std::string UniqueIdToString(const uint64_t id) {
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << id;
    return oss.str();
}

unsigned int ReadTemperatureC(const uint32_t deviceIndex) {
    int64_t milliC = 0;
    rsmi_status_t status = rsmi_dev_temp_metric_get(
        deviceIndex, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &milliC);
    if (!IsSuccess(status)) {
        status = rsmi_dev_temp_metric_get(
            deviceIndex, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &milliC);
    }
    if (!IsSuccess(status) || milliC <= 0) return 0;
    return static_cast<unsigned int>(milliC / kMillidegreesPerDegree);
}

bool ReadThermalThrottle(const uint32_t deviceIndex) {
    int64_t currentMilliC = 0;
    int64_t criticalMilliC = 0;
    const rsmi_status_t currentStatus = rsmi_dev_temp_metric_get(
        deviceIndex, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &currentMilliC);
    const rsmi_status_t criticalStatus = rsmi_dev_temp_metric_get(
        deviceIndex, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CRITICAL, &criticalMilliC);
    if (!IsSuccess(currentStatus) || !IsSuccess(criticalStatus) ||
        criticalMilliC <= 0) {
        return false;
    }
    return currentMilliC >= criticalMilliC;
}

bool ReadPowerThrottle(const uint32_t deviceIndex) {
    uint64_t powerUw = 0;
    RSMI_POWER_TYPE powerType = RSMI_INVALID_POWER;
    uint64_t capUw = 0;

    const rsmi_status_t powerStatus =
        rsmi_dev_power_get(deviceIndex, &powerUw, &powerType);
    const rsmi_status_t capStatus =
        rsmi_dev_power_cap_get(deviceIndex, 0, &capUw);

    if (!IsSuccess(powerStatus) || !IsSuccess(capStatus) ||
        powerType == RSMI_INVALID_POWER || capUw == 0) {
        return false;
    }

    return powerUw >= capUw;
}

DeviceSample BuildTelemetrySample(const uint32_t deviceIndex) {
    DeviceSample sample{};
    sample.device_id = static_cast<int>(deviceIndex);
    sample.vendor = "AMD";

    char name[256] = {};
    if (IsSuccess(rsmi_dev_name_get(deviceIndex, name, sizeof(name))) &&
        name[0] != '\0') {
        sample.name = name;
    }

    uint64_t uniqueId = 0;
    if (IsSuccess(rsmi_dev_unique_id_get(deviceIndex, &uniqueId)) && uniqueId != 0) {
        sample.uuid = UniqueIdToString(uniqueId);
    }

    uint64_t totalBytes = 0;
    uint64_t usedBytes = 0;
    if (IsSuccess(
            rsmi_dev_memory_total_get(deviceIndex, RSMI_MEM_TYPE_VRAM, &totalBytes))) {
        sample.total_mib = ToMiB(totalBytes);
    }
    if (IsSuccess(
            rsmi_dev_memory_usage_get(deviceIndex, RSMI_MEM_TYPE_VRAM, &usedBytes))) {
        sample.used_mib = ToMiB(usedBytes);
    }
    if (sample.total_mib >= sample.used_mib) {
        sample.free_mib = sample.total_mib - sample.used_mib;
    }

    uint32_t gpuBusy = 0;
    if (IsSuccess(rsmi_dev_busy_percent_get(deviceIndex, &gpuBusy))) {
        sample.gpu_util = std::min(gpuBusy, 100u);
    }

    uint32_t memBusy = 0;
    if (IsSuccess(rsmi_dev_memory_busy_percent_get(deviceIndex, &memBusy))) {
        sample.mem_util = std::min(memBusy, 100u);
    }

    sample.temp_c = ReadTemperatureC(deviceIndex);

    uint64_t powerUw = 0;
    RSMI_POWER_TYPE powerType = RSMI_INVALID_POWER;
    if (IsSuccess(rsmi_dev_power_get(deviceIndex, &powerUw, &powerType)) &&
        powerType != RSMI_INVALID_POWER) {
        sample.power_mw =
            static_cast<unsigned int>(powerUw / kMicrowattsPerMilliwatt);
    }

    rsmi_frequencies_t gfxFreq{};
    if (IsSuccess(rsmi_dev_gpu_clk_freq_get(deviceIndex, RSMI_CLK_TYPE_SYS,
                                            &gfxFreq))) {
        sample.clock_gfx = ToMHz(gfxFreq);
        sample.clock_sm = sample.clock_gfx;
    }

    rsmi_frequencies_t memFreq{};
    if (IsSuccess(rsmi_dev_gpu_clk_freq_get(deviceIndex, RSMI_CLK_TYPE_MEM,
                                            &memFreq))) {
        sample.clock_mem = ToMHz(memFreq);
    }

    uint64_t pciBdf = 0;
    if (IsSuccess(rsmi_dev_pci_id_get(deviceIndex, &pciBdf))) {
        sample.pci_bus_id = DecodePciBusId(pciBdf);
    }

    uint64_t pcieSent = 0;
    uint64_t pcieReceived = 0;
    uint64_t maxPacketSize = 0;
    if (IsSuccess(rsmi_dev_pci_throughput_get(deviceIndex, &pcieSent,
                                              &pcieReceived, &maxPacketSize))) {
        sample.pcie_rx_bps = pcieSent;
        sample.pcie_tx_bps = pcieReceived;
    }

    sample.throttle_power = ReadPowerThrottle(deviceIndex);
    sample.throttle_thermal = ReadThermalThrottle(deviceIndex);
    return sample;
}

std::pair<bool, uint32_t> ProbeTelemetryDevices(std::string* reason) {
    if (reason) reason->clear();

    const rsmi_status_t initStatus = rsmi_init(0);
    if (!IsSuccess(initStatus)) {
        if (reason) *reason = "rsmi_init failed: " + RsmiStatusToString(initStatus);
        return {false, 0};
    }

    uint32_t count = 0;
    const rsmi_status_t countStatus = rsmi_num_monitor_devices(&count);
    rsmi_shut_down();

    if (!IsSuccess(countStatus)) {
        if (reason) {
            *reason =
                "rsmi_num_monitor_devices failed: " + RsmiStatusToString(countStatus);
        }
        return {false, 0};
    }
    if (count == 0) {
        if (reason) *reason = "ROCm SMI initialized but found no monitor devices.";
        return {false, 0};
    }
    return {true, count};
}
#endif

}  // namespace

bool RocmCollector::IsAvailable(std::string* reason) {
    std::string telemetryReason;
    std::string staticReason;
    const bool telemetryAvailable = IsTelemetryAvailable(&telemetryReason);
    const bool staticAvailable = IsStaticInfoAvailable(&staticReason);

    if (telemetryAvailable || staticAvailable) {
        if (reason) reason->clear();
        return true;
    }

    if (reason) {
        *reason = "AMD backend unavailable.";
        if (!telemetryReason.empty()) {
            *reason += " ROCm SMI: " + telemetryReason;
        }
        if (!staticReason.empty()) {
            *reason += " HIP: " + staticReason;
        }
    }
    return false;
}

bool RocmCollector::IsTelemetryAvailable(std::string* reason) {
#if GPUFL_HAS_ROCM_SMI
    return ProbeTelemetryDevices(reason).first;
#else
    if (reason) {
        *reason = "ROCm SMI support not compiled in.";
    }
    return false;
#endif
}

bool RocmCollector::IsStaticInfoAvailable(std::string* reason) {
#if GPUFL_HAS_HIP
    return ProbeHipDeviceCount(reason).first;
#else
    if (reason) {
        *reason = "HIP runtime support not compiled in.";
    }
    return false;
#endif
}

RocmCollector::RocmCollector() {
#if GPUFL_HAS_ROCM_SMI
    const rsmi_status_t initStatus = rsmi_init(0);
    if (IsSuccess(initStatus)) {
        uint32_t count = 0;
        const rsmi_status_t countStatus = rsmi_num_monitor_devices(&count);
        if (IsSuccess(countStatus) && count > 0) {
            telemetry_initialized_ = true;
            telemetry_device_count_ = count;
        } else {
            rsmi_shut_down();
        }
    }
#endif

#if GPUFL_HAS_HIP
    static_info_available_ = ProbeHipDeviceCount(nullptr).first;
    if (static_info_available_) {
        static_device_count_ = ProbeHipDeviceCount(nullptr).second;
    }
#endif
}

RocmCollector::~RocmCollector() {
#if GPUFL_HAS_ROCM_SMI
    if (telemetry_initialized_) {
        rsmi_shut_down();
        telemetry_initialized_ = false;
        telemetry_device_count_ = 0;
    }
#endif
    static_info_available_ = false;
    static_device_count_ = 0;
}

std::vector<DeviceSample> RocmCollector::sampleAll() {
    std::vector<DeviceSample> out;
    if (!telemetry_initialized_ || telemetry_device_count_ == 0) return out;

#if GPUFL_HAS_ROCM_SMI
    out.reserve(telemetry_device_count_);
    for (uint32_t i = 0; i < telemetry_device_count_; ++i) {
        out.push_back(BuildTelemetrySample(i));
    }
#endif
    return out;
}

std::vector<GpuStaticDeviceInfo> RocmCollector::sampleStaticInfo() {
    std::vector<GpuStaticDeviceInfo> devices;
    if (!static_info_available_ || static_device_count_ <= 0) return devices;

#if GPUFL_HAS_HIP
    devices.reserve(static_device_count_);
    for (int i = 0; i < static_device_count_; ++i) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, i) != hipSuccess) {
            (void)hipGetLastError();
            continue;
        }
        if (!IsGpuHipDevice(prop)) {
            GFL_LOG_DEBUG("[RocmCollector] skipping non-GPU HIP static device index=",
                          i, " name=", prop.name, " arch=", prop.gcnArchName);
            continue;
        }
        devices.push_back(BuildStaticDeviceInfo(i, prop));
    }
#endif
    return devices;
}

}  // namespace gpufl::amd
