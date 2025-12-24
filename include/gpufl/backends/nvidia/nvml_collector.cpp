#if !(GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML)
#error "nvml_collector.cpp should only be compiled when GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML are true."
#endif

#include "gpufl/backends/nvidia/nvml_collector.hpp"

#include <sstream>

// Define max links (Hopper/Ampere usually have max 12-18 links)
#define MAX_LVLINKS 18

namespace gpufl::nvidia {
    std::string NvmlCollector::nvmlErrorToString(nvmlReturn_t r) {
        const char* s = nvmlErrorString(r);
        return s ? std::string(s) : std::string("unknown nvml error");
    }

    unsigned long long NvmlCollector::toMiB(unsigned long long bytes) {
        return bytes / (1024ull * 1024ull);
    }

    bool NvmlCollector::isAvailable(std::string* reason) {
        // If NVML is linked, the best probe is: can we init?
        nvmlReturn_t r = nvmlInit_v2();
        if (r != NVML_SUCCESS) {
            if (reason) *reason = "nvmlInit_v2 failed: " + nvmlErrorToString(r);
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
            s.deviceId = static_cast<int>(i);

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
            s.pciBusId = static_cast<int>(pci.bus);

            if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
                s.totalMiB = static_cast<long long>(toMiB(mem.total));
                s.usedMiB  = static_cast<long long>(toMiB(mem.used));
                s.freeMiB  = static_cast<long long>(toMiB(mem.free));
            }

            if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
                s.gpuUtil = static_cast<int>(util.gpu);
                s.memUtil = static_cast<int>(util.memory);
            }

            if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &tempC) == NVML_SUCCESS) {
                s.tempC = static_cast<int>(tempC);
            }

            if (nvmlDeviceGetPowerUsage(dev, &powerMilliW) == NVML_SUCCESS) {
                s.powerMw = static_cast<long long>(powerMilliW);
            }

            // Clocks (not all GPUs expose all clocks; ignore failures)
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &clkGfx) == NVML_SUCCESS) s.clockGfx = static_cast<int>(clkGfx);
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clkSm) == NVML_SUCCESS)       s.clockSm  = static_cast<int>(clkSm);
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &clkMem) == NVML_SUCCESS)     s.clockMem = static_cast<int>(clkMem);

            // Throttle Reasons
            unsigned long long reasons = 0;
            if (nvmlDeviceGetCurrentClocksThrottleReasons(dev, &reasons) == NVML_SUCCESS) {
                // Check for Power Cap (0x0000000000000004 usually, but check nvml.h constant)
                // NVML_CLOCKS_THROTTLE_REASON_SW_POWER_CAP
                s.throttlePower = (reasons & 0x0000000000000004ULL) != 0;

                // Check for Thermal (Hardware Slowdown or Thermal Caps)
                // NVML_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN | SW_THERMAL | HW_THERMAL
                bool therm = false;
                if (reasons & 0x0000000000000008ULL) therm = true;
                if (reasons & 0x0000000000000020ULL) therm = true;
                if (reasons & 0x0000000000000040ULL) therm = true;
                s.throttleThermal = therm;
            } else {
                s.throttlePower = false;
                s.throttleThermal = false;
            }

            // ---------------------------------------------------------
            // NVLink Bandwidth Calculation
            // ---------------------------------------------------------

            std::vector<nvmlFieldValue_t> fields(2);
            fields[0].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
            fields[1].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;

            if (nvmlReturn_t ret = nvmlDeviceGetFieldValues(dev, 2, fields.data()); ret == NVML_SUCCESS) {
                unsigned long long rxKiB = (fields[0].nvmlReturn == NVML_SUCCESS) ? fields[0].value.ullVal : 0;
                unsigned long long txKiB = (fields[1].nvmlReturn == NVML_SUCCESS) ? fields[1].value.ullVal : 0;

                s.nvlinkRxBps = rxKiB * 1024;
                s.nvlinkTxBps = txKiB * 1024;
            } else {
                s.nvlinkRxBps = 0;
                s.nvlinkTxBps = 0;
            }


            // ---------------------------------------------------------
            // NVLink PCIe Throughput
            // ---------------------------------------------------------

            unsigned int pcieRx = 0; // KB/s
            unsigned int pcieTx = 0; // KB/s

            nvmlReturn_t r1 = nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_RX_BYTES, &pcieRx);
            nvmlReturn_t r2 = nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_TX_BYTES, &pcieTx);

            if (r1 == NVML_SUCCESS) s.pcieRxBps = static_cast<unsigned long long>(pcieRx) * 1024; // KB/s -> B/s
            else s.pcieRxBps = 0;

            if (r2 == NVML_SUCCESS) s.pcieTxBps = static_cast<unsigned long long>(pcieTx) * 1024; // KB/s -> B/s
            else s.pcieTxBps = 0;

            out.push_back(std::move(s));
        }

        return out;
    }
} // namespace gpufl::nvidia
