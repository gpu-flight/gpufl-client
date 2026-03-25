#if !(GPUFL_ENABLE_AMD && GPUFL_HAS_ROCM_SMI)
#error \
    "rocm_collector.cpp should only be compiled when GPUFL_ENABLE_AMD && GPUFL_HAS_ROCM_SMI are true."
#endif

#include "rocm_collector.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#include <rocm_smi/rocm_smi.h>

namespace gpufl::amd {
namespace {

constexpr uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr uint64_t kMicrowattsPerMilliwatt = 1000ull;
constexpr int64_t kMillidegreesPerDegree = 1000ll;

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

size_t ToMiB(const uint64_t bytes) { return static_cast<size_t>(bytes / kBytesPerMiB); }

unsigned int ToMHz(const rsmi_frequencies_t& freqs) {
    if (freqs.num_supported == 0 || freqs.current >= freqs.num_supported) return 0;
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
    rsmi_status_t criticalStatus = rsmi_dev_temp_metric_get(
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

}  // namespace

bool RocmCollector::IsAvailable(std::string* reason) {
    if (reason) reason->clear();

    const rsmi_status_t initStatus = rsmi_init(0);
    if (!IsSuccess(initStatus)) {
        if (reason) *reason = "rsmi_init failed: " + RsmiStatusToString(initStatus);
        return false;
    }

    uint32_t count = 0;
    const rsmi_status_t countStatus = rsmi_num_monitor_devices(&count);
    rsmi_shut_down();

    if (!IsSuccess(countStatus)) {
        if (reason) {
            *reason =
                "rsmi_num_monitor_devices failed: " + RsmiStatusToString(countStatus);
        }
        return false;
    }

    if (count == 0) {
        if (reason) *reason = "ROCm SMI initialized but found no monitor devices.";
        return false;
    }

    return true;
}

RocmCollector::RocmCollector() {
    const rsmi_status_t initStatus = rsmi_init(0);
    if (!IsSuccess(initStatus)) return;

    uint32_t count = 0;
    const rsmi_status_t countStatus = rsmi_num_monitor_devices(&count);
    if (!IsSuccess(countStatus) || count == 0) {
        rsmi_shut_down();
        return;
    }

    initialized_ = true;
    deviceCount_ = count;
}

RocmCollector::~RocmCollector() {
    if (initialized_) {
        rsmi_shut_down();
        initialized_ = false;
        deviceCount_ = 0;
    }
}

std::vector<gpufl::DeviceSample> RocmCollector::sampleAll() {
    std::vector<gpufl::DeviceSample> out;
    if (!initialized_ || deviceCount_ == 0) return out;

    out.reserve(deviceCount_);
    for (uint32_t i = 0; i < deviceCount_; ++i) {
        gpufl::DeviceSample sample{};
        sample.device_id = static_cast<int>(i);
        sample.vendor = "AMD";

        char name[256] = {};
        if (IsSuccess(rsmi_dev_name_get(i, name, sizeof(name))) && name[0] != '\0') {
            sample.name = name;
        }

        uint64_t uniqueId = 0;
        if (IsSuccess(rsmi_dev_unique_id_get(i, &uniqueId)) && uniqueId != 0) {
            sample.uuid = UniqueIdToString(uniqueId);
        }

        uint64_t totalBytes = 0;
        uint64_t usedBytes = 0;
        if (IsSuccess(rsmi_dev_memory_total_get(i, RSMI_MEM_TYPE_VRAM, &totalBytes))) {
            sample.total_mib = ToMiB(totalBytes);
        }
        if (IsSuccess(rsmi_dev_memory_usage_get(i, RSMI_MEM_TYPE_VRAM, &usedBytes))) {
            sample.used_mib = ToMiB(usedBytes);
        }
        if (sample.total_mib >= sample.used_mib) {
            sample.free_mib = sample.total_mib - sample.used_mib;
        }

        uint32_t gpuBusy = 0;
        if (IsSuccess(rsmi_dev_busy_percent_get(i, &gpuBusy))) {
            sample.gpu_util = std::min(gpuBusy, 100u);
        }

        uint32_t memBusy = 0;
        if (IsSuccess(rsmi_dev_memory_busy_percent_get(i, &memBusy))) {
            sample.mem_util = std::min(memBusy, 100u);
        }

        sample.temp_c = ReadTemperatureC(i);

        uint64_t powerUw = 0;
        RSMI_POWER_TYPE powerType = RSMI_INVALID_POWER;
        if (IsSuccess(rsmi_dev_power_get(i, &powerUw, &powerType)) &&
            powerType != RSMI_INVALID_POWER) {
            sample.power_mw =
                static_cast<unsigned int>(powerUw / kMicrowattsPerMilliwatt);
        }

        rsmi_frequencies_t gfxFreq{};
        if (IsSuccess(rsmi_dev_gpu_clk_freq_get(i, RSMI_CLK_TYPE_SYS, &gfxFreq))) {
            sample.clock_gfx = ToMHz(gfxFreq);
            sample.clock_sm = sample.clock_gfx;
        }

        rsmi_frequencies_t memFreq{};
        if (IsSuccess(rsmi_dev_gpu_clk_freq_get(i, RSMI_CLK_TYPE_MEM, &memFreq))) {
            sample.clock_mem = ToMHz(memFreq);
        }

        uint64_t pciBdf = 0;
        if (IsSuccess(rsmi_dev_pci_id_get(i, &pciBdf))) {
            sample.pci_bus_id = DecodePciBusId(pciBdf);
        }

        uint64_t pcieSent = 0;
        uint64_t pcieReceived = 0;
        uint64_t maxPacketSize = 0;
        if (IsSuccess(rsmi_dev_pci_throughput_get(i, &pcieSent, &pcieReceived,
                                                  &maxPacketSize))) {
            sample.pcie_rx_bps = pcieSent;
            sample.pcie_tx_bps = pcieReceived;
        }

        sample.throttle_power = ReadPowerThrottle(i);
        sample.throttle_thermal = ReadThermalThrottle(i);

        out.push_back(std::move(sample));
    }

    return out;
}

}  // namespace gpufl::amd
