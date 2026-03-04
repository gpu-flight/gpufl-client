#if !(GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML)
#error \
    "nvml_collector.cpp should only be compiled when GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML are true."
#endif

#include "gpufl/backends/nvidia/nvml_collector.hpp"

#include <sstream>

// Define max links (Hopper/Ampere usually have max 12-18 links)
#define MAX_LVLINKS 18

namespace gpufl::nvidia {
std::string NvmlCollector::NvmlErrorToString(nvmlReturn_t r) {
    const char* s = nvmlErrorString(r);
    return s ? std::string(s) : std::string("unknown nvml error");
}

unsigned long long NvmlCollector::ToMiB(unsigned long long bytes) {
    return bytes / (1024ull * 1024ull);
}

bool NvmlCollector::IsAvailable(std::string* reason) {
    // If NVML is linked, the best probe is: can we init?
    nvmlReturn_t r = nvmlInit_v2();
    if (r != NVML_SUCCESS) {
        if (reason) *reason = "nvmlInit_v2 failed: " + NvmlErrorToString(r);
        return false;
    }
    nvmlShutdown();
    return true;
}

NvmlCollector::NvmlCollector() {
    nvmlReturn_t r = nvmlInit_v2();
    if (r != NVML_SUCCESS) return;

    initialized_ = true;

    r = nvmlDeviceGetCount_v2(&deviceCount_);
    if (r != NVML_SUCCESS) {
        deviceCount_ = 0;
    }
}

NvmlCollector::~NvmlCollector() {
    if (initialized_) {
        nvmlShutdown();
        initialized_ = false;
    }
}

std::vector<gpufl::DeviceSample> NvmlCollector::sampleAll() {
    std::vector<gpufl::DeviceSample> out;

    if (!initialized_ || deviceCount_ == 0) return out;
    out.reserve(deviceCount_);

    for (unsigned int i = 0; i < deviceCount_; ++i) {
        nvmlDevice_t dev{};
        nvmlReturn_t r = nvmlDeviceGetHandleByIndex_v2(i, &dev);
        if (r != NVML_SUCCESS) continue;

        gpufl::DeviceSample s{};
        s.device_id = static_cast<int>(i);

        char name[NVML_DEVICE_NAME_BUFFER_SIZE]{};
        char uuid[NVML_DEVICE_UUID_BUFFER_SIZE]{};
        nvmlPciInfo_t pci{};
        nvmlMemory_t mem{};
        nvmlUtilization_t util{};
        unsigned int tempC = 0;
        unsigned int powerMilliW = 0;
        unsigned int clkGfx = 0, clkSm = 0, clkMem = 0;

        nvmlDeviceGetName(dev, name, sizeof(name));
        nvmlDeviceGetUUID(dev, uuid, sizeof(uuid));
        nvmlDeviceGetPciInfo_v3(dev, &pci);

        s.name = name;
        s.uuid = uuid;
        s.vendor = "NVIDIA";
        s.pci_bus_id = static_cast<int>(pci.bus);

        if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
            s.total_mib = static_cast<long long>(ToMiB(mem.total));
            s.used_mib = static_cast<long long>(ToMiB(mem.used));
            s.free_mib = static_cast<long long>(ToMiB(mem.free));
        }

        if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
            s.gpu_util = static_cast<int>(util.gpu);
            s.mem_util = static_cast<int>(util.memory);
        }

        nvmlTemperature_t tempInfo{};
        tempInfo.version = nvmlTemperature_v1;
        tempInfo.sensorType = NVML_TEMPERATURE_GPU;
        if (nvmlDeviceGetTemperatureV(dev, &tempInfo) == NVML_SUCCESS) {
            s.temp_c = static_cast<int>(tempInfo.temperature);
        }

        if (nvmlDeviceGetPowerUsage(dev, &powerMilliW) == NVML_SUCCESS) {
            s.power_mw = static_cast<long long>(powerMilliW);
        }

        // Clocks (not all GPUs expose all clocks; ignore failures)
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &clkGfx) ==
            NVML_SUCCESS)
            s.clock_gfx = static_cast<int>(clkGfx);
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clkSm) == NVML_SUCCESS)
            s.clock_sm = static_cast<int>(clkSm);
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &clkMem) ==
            NVML_SUCCESS)
            s.clock_mem = static_cast<int>(clkMem);

        // Throttle Reasons
        unsigned long long reasons = 0;
        if (nvmlDeviceGetCurrentClocksEventReasons(dev, &reasons) ==
            NVML_SUCCESS) {
            // Check for Power Cap
            // nvmlClocksEventReasonSwPowerCap usually 0x0000000000000004ULL
            s.throttle_power = (reasons & nvmlClocksEventReasonSwPowerCap) != 0;

            // Check for Thermal (Software Thermal Slowdown)
            // nvmlClocksEventReasonSwThermalSlowdown usually
            // 0x0000000000000020ULL
            s.throttle_thermal =
                (reasons & nvmlClocksEventReasonSwThermalSlowdown) != 0;
        } else {
            s.throttle_power = false;
            s.throttle_thermal = false;
        }

        // ---------------------------------------------------------
        // NVLink Bandwidth Calculation
        // ---------------------------------------------------------

        std::vector<nvmlFieldValue_t> fields(2);
        fields[0].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
        fields[1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;

        if (nvmlReturn_t ret = nvmlDeviceGetFieldValues(dev, 2, fields.data());
            ret == NVML_SUCCESS) {
            unsigned long long rxKiB = (fields[0].nvmlReturn == NVML_SUCCESS)
                                           ? fields[0].value.ullVal
                                           : 0;
            unsigned long long txKiB = (fields[1].nvmlReturn == NVML_SUCCESS)
                                           ? fields[1].value.ullVal
                                           : 0;

            s.nvlink_rx_bps = rxKiB * 1024;
            s.nvlink_tx_bps = txKiB * 1024;
        } else {
            s.nvlink_rx_bps = 0;
            s.nvlink_tx_bps = 0;
        }

        // ---------------------------------------------------------
        // NVLink PCIe Throughput
        // ---------------------------------------------------------

        unsigned int pcieRx = 0;  // KB/s
        unsigned int pcieTx = 0;  // KB/s

        nvmlReturn_t r1 =
            nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_RX_BYTES, &pcieRx);
        nvmlReturn_t r2 =
            nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_TX_BYTES, &pcieTx);

        if (r1 == NVML_SUCCESS)
            s.pcie_rx_bps =
                static_cast<unsigned long long>(pcieRx) * 1024;  // KB/s -> B/s
        else
            s.pcie_rx_bps = 0;

        if (r2 == NVML_SUCCESS)
            s.pcie_tx_bps =
                static_cast<unsigned long long>(pcieTx) * 1024;  // KB/s -> B/s
        else
            s.pcie_tx_bps = 0;

        out.push_back(std::move(s));
    }

    return out;
}
}  // namespace gpufl::nvidia
