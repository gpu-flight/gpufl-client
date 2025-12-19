#if !(GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML)
#error "nvml_collector.cpp should only be compiled when GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML are true."
#endif

#include "gpufl/backends/nvidia/nvml_collector.hpp"

#include <sstream>

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

    std::string NvmlCollector::devicesInventoryJson() {
        std::ostringstream oss;
        oss << "[";

        if (!initialized_ || deviceCount_ == 0) {
            oss << "]";
            return oss.str();
        }

        for (unsigned int i = 0; i < deviceCount_; ++i) {
            if (i) oss << ",";

            nvmlDevice_t dev{};
            nvmlReturn_t r = nvmlDeviceGetHandleByIndex_v2(i, &dev);
            if (r != NVML_SUCCESS) {
                oss << "{}";
                continue;
            }

            char name[NVML_DEVICE_NAME_BUFFER_SIZE]{};
            char uuid[NVML_DEVICE_UUID_BUFFER_SIZE]{};
            nvmlPciInfo_t pci{};
            nvmlMemory_t mem{};

            nvmlDeviceGetName(dev, name, sizeof(name));
            nvmlDeviceGetUUID(dev, uuid, sizeof(uuid));
            nvmlDeviceGetPciInfo_v3(dev, &pci);
            nvmlDeviceGetMemoryInfo(dev, &mem);

            oss << "{"
                << "\"id\":" << i
                << ",\"name\":\"" << name << "\""
                << ",\"uuid\":\"" << uuid << "\""
                << ",\"pci_bus\":" << static_cast<int>(pci.bus)
                << ",\"total_mib\":" << toMiB(mem.total)
                << "}";
        }

        oss << "]";
        return oss.str();
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
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &clkGfx) == NVML_SUCCESS) s.clockGfx = (int)clkGfx;
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clkSm) == NVML_SUCCESS)       s.clockSm  = (int)clkSm;
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &clkMem) == NVML_SUCCESS)     s.clockMem = (int)clkMem;

            out.push_back(std::move(s));
        }

        return out;
    }
} // namespace gpufl::nvidia
