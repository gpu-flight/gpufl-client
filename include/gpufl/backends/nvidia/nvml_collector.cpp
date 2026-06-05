#if !(GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML)
#error \
    "nvml_collector.cpp should only be compiled when GPUFL_ENABLE_NVIDIA && GPUFL_HAS_NVML are true."
#endif

#include "gpufl/backends/nvidia/nvml_collector.hpp"

#include <chrono>
#include <sstream>

#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/teardown_flag.hpp"

#ifdef _WIN32
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <cwchar>  // wcsstr — PDH engine-instance name filtering
#pragma comment(lib, "pdh.lib")
#endif

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

#ifdef _WIN32
    initPdh_();
#endif
#if defined(_WIN32) && GPUFL_HAS_NVAPI
    initNvapi_();
#endif
}

NvmlCollector::~NvmlCollector() {
#ifdef _WIN32
    cleanupPdh_();
#endif
#if defined(_WIN32) && GPUFL_HAS_NVAPI
    cleanupNvapi_();
#endif
    if (initialized_) {
        // Skip nvmlShutdown() during Windows injection process-exit teardown:
        // the driver is being torn down and nvmlShutdown() deadlocks against
        // it (process becomes unkillable). The OS reclaims the NVML handle at
        // exit anyway. Normal (Linux, or embedded-SDK mid-process) shutdown
        // still calls it. See gpufl/core/teardown_flag.hpp.
        if (!gpufl::detail::isProcessExitTeardown()) {
            nvmlShutdown();
        }
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
#if defined(_WIN32) && GPUFL_HAS_NVAPI
        // On Windows WDDM, NVML's util rates read 0% for CUDA workloads (both
        // gpu and memory). NVAPI's dynamic-pstates GPU/FB domains report
        // correctly on WDDM — and FB (memory-controller) util has no PDH
        // equivalent, so NVAPI is the only source for mem_util here. Only
        // backfill fields NVML left at 0, so correct TCC/datacenter values
        // (where NVML works) stay intact.
        if ((s.gpu_util == 0 || s.mem_util == 0) && nvapi_available_) {
            unsigned int nvapiGpu = 0, nvapiMem = 0;
            if (sampleUtilNvapi_(s.pci_bus_id, nvapiGpu, nvapiMem)) {
                if (s.gpu_util == 0) s.gpu_util = nvapiGpu;
                if (s.mem_util == 0) s.mem_util = nvapiMem;
            }
        }
#endif
#ifdef _WIN32
        // Last-resort gpu_util fallback: the Windows PDH GPU-engine counter
        // (reports correctly on WDDM). There is no PDH path for mem_util — no
        // such counter exists on Windows.
        if (s.gpu_util == 0 && pdh_available_) {
            s.gpu_util = sampleGpuUtilPdh_();
        }
#endif

        nvmlTemperature_t tempInfo{};
        tempInfo.version = nvmlTemperature_v1;
        tempInfo.sensorType = NVML_TEMPERATURE_GPU;
        if (nvmlDeviceGetTemperatureV(dev, &tempInfo) == NVML_SUCCESS) {
            s.temp_c = static_cast<int>(tempInfo.temperature);
        }

        if (nvmlDeviceGetPowerUsage(dev, &powerMilliW) == NVML_SUCCESS) {
            // Some Blackwell (RTX 50-series) drivers return cumulative energy
            // in millijoules instead of instantaneous power in milliwatts.
            // Detect this: any GPU drawing >1000 W is clearly wrong.
            if (powerMilliW > 1000000) {
                // Compute instantaneous power from energy delta
                static unsigned long long prevEnergy = 0;
                static std::chrono::steady_clock::time_point prevTime;
                auto now = std::chrono::steady_clock::now();
                if (prevEnergy > 0 && powerMilliW > prevEnergy) {
                    auto dtMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - prevTime).count();
                    if (dtMs > 0) {
                        // energy delta (mJ) / time delta (s) = power (mW)
                        s.power_mw = static_cast<long long>(
                            (powerMilliW - prevEnergy) * 1000ULL / static_cast<unsigned long long>(dtMs));
                    }
                }
                prevEnergy = powerMilliW;
                prevTime = now;
            } else {
                s.power_mw = static_cast<long long>(powerMilliW);
            }
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
        // Extended metrics (silently ignore failures — not all GPUs
        // support all sensors)
        // ---------------------------------------------------------

        // Fan speed (may fail on laptops / blower-less cards)
        unsigned int fanSpeed = 0;
        if (nvmlDeviceGetFanSpeed_v2(dev, 0, &fanSpeed) == NVML_SUCCESS)
            s.fan_speed_pct = fanSpeed;

        // Memory temperature — not available via NVML on NVIDIA GPUs
        // (only NVML_TEMPERATURE_GPU sensor exists). Left at 0.

        // Junction temperature — on NVIDIA, the GPU die sensor IS the
        // junction temp, so mirror temp_c.
        s.temp_junction_c = s.temp_c;

        // Cumulative energy consumption (millijoules → microjoules)
        {
            unsigned long long energyMj = 0;
            if (nvmlDeviceGetTotalEnergyConsumption(dev, &energyMj) ==
                NVML_SUCCESS)
                s.energy_uj = energyMj * 1000ULL;
        }

        // ECC error counters (only on GPUs with ECC — datacenter cards)
        {
            unsigned long long corrected = 0, uncorrected = 0;
            if (nvmlDeviceGetMemoryErrorCounter(
                    dev, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    NVML_VOLATILE_ECC, NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                    &corrected) == NVML_SUCCESS)
                s.ecc_corrected = corrected;
            if (nvmlDeviceGetMemoryErrorCounter(
                    dev, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    NVML_VOLATILE_ECC, NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                    &uncorrected) == NVML_SUCCESS)
                s.ecc_uncorrected = uncorrected;
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
#ifdef _WIN32
void NvmlCollector::initPdh_() {
    PDH_HQUERY query = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) return;

    PDH_HCOUNTER counter = nullptr;
    // English counter name — locale-independent.
    // engtype_3D = 3D/Compute engine (CUDA workloads).
    PDH_STATUS st = PdhAddEnglishCounterW(
        query,
        L"\\GPU Engine(*)\\Utilization Percentage",
        0, &counter);
    if (st != ERROR_SUCCESS) {
        PdhCloseQuery(query);
        return;
    }

    // First collect establishes baseline for rate counters.
    PdhCollectQueryData(query);

    pdh_query_ = query;
    pdh_gpu_counter_ = counter;
    pdh_available_ = true;
    GFL_LOG_DEBUG("[NVML] PDH GPU utilization counter initialized");
}

void NvmlCollector::cleanupPdh_() {
    if (pdh_query_) {
        PdhCloseQuery(static_cast<PDH_HQUERY>(pdh_query_));
        pdh_query_ = nullptr;
        pdh_gpu_counter_ = nullptr;
        pdh_available_ = false;
    }
}

unsigned int NvmlCollector::sampleGpuUtilPdh_() {
    auto query = static_cast<PDH_HQUERY>(pdh_query_);
    auto counter = static_cast<PDH_HCOUNTER>(pdh_gpu_counter_);

    if (PdhCollectQueryData(query) != ERROR_SUCCESS) return 0;

    // The wildcard counter returns one instance per (process, GPU engine).
    // We take the max utilization across the 3D/Compute engine instances
    // (the per-instance engine-type filtering happens in the loop below).
    DWORD bufSize = 0, itemCount = 0;
    PDH_STATUS st = PdhGetFormattedCounterArrayW(
        counter, PDH_FMT_DOUBLE, &bufSize, &itemCount, nullptr);
    if (st != PDH_MORE_DATA || bufSize == 0) return 0;

    std::vector<uint8_t> buf(bufSize);
    auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
    st = PdhGetFormattedCounterArrayW(
        counter, PDH_FMT_DOUBLE, &bufSize, &itemCount, items);
    if (st != ERROR_SUCCESS) return 0;

    double maxUtil = 0.0;
    for (DWORD i = 0; i < itemCount; ++i) {
        if (items[i].FmtValue.CStatus != PDH_CSTATUS_VALID_DATA) continue;
        // Count only the 3D/Compute engines — CUDA runs there. Skipping the
        // Copy/Video/Encode/Decode instances stops background video playback
        // from inflating gpu_util. Instance names look like
        // "pid_1234_..._engtype_3D" / "..._engtype_Compute".
        const wchar_t* name = items[i].szName;
        if (!name ||
            (!wcsstr(name, L"engtype_3D") && !wcsstr(name, L"engtype_Compute")))
            continue;
        if (items[i].FmtValue.doubleValue > maxUtil)
            maxUtil = items[i].FmtValue.doubleValue;
    }

    // PDH returns 0-100 as double; clamp and convert.
    if (maxUtil > 100.0) maxUtil = 100.0;
    return static_cast<unsigned int>(maxUtil);
}
#endif

#if defined(_WIN32) && GPUFL_HAS_NVAPI
void NvmlCollector::initNvapi_() {
    if (NvAPI_Initialize() != NVAPI_OK) return;

    NvPhysicalGpuHandle handles[NVAPI_MAX_PHYSICAL_GPUS] = {};
    NvU32 count = 0;
    if (NvAPI_EnumPhysicalGPUs(handles, &count) != NVAPI_OK) {
        NvAPI_Unload();
        return;
    }
    // Map each physical GPU by PCI bus id so we can line NVAPI handles up with
    // NVML's DeviceSample.pci_bus_id (NVAPI enum order != NVML index order).
    for (NvU32 i = 0; i < count; ++i) {
        NvU32 busId = 0;
        if (NvAPI_GPU_GetBusId(handles[i], &busId) == NVAPI_OK)
            nvapi_by_bus_[static_cast<unsigned int>(busId)] = handles[i];
    }
    nvapi_available_ = !nvapi_by_bus_.empty();
    if (nvapi_available_)
        GFL_LOG_DEBUG("[NVML] NVAPI dynamic-pstates utilization initialized");
    else
        NvAPI_Unload();
}

void NvmlCollector::cleanupNvapi_() {
    if (!nvapi_available_) return;
    nvapi_by_bus_.clear();
    nvapi_available_ = false;
    // Mirror the nvmlShutdown teardown guard: during Windows injection
    // process-exit the driver is being torn down and NvAPI_Unload can hang;
    // the OS reclaims everything at exit anyway. See teardown_flag.hpp.
    if (!gpufl::detail::isProcessExitTeardown()) {
        NvAPI_Unload();
    }
}

bool NvmlCollector::sampleUtilNvapi_(int pciBus, unsigned int& gpu,
                                     unsigned int& mem) {
    if (!nvapi_available_) return false;
    auto it = nvapi_by_bus_.find(static_cast<unsigned int>(pciBus));
    if (it == nvapi_by_bus_.end()) return false;

    NV_GPU_DYNAMIC_PSTATES_INFO_EX info{};
    info.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;
    if (NvAPI_GPU_GetDynamicPstatesInfoEx(it->second, &info) != NVAPI_OK)
        return false;

    // NVAPI's public header doesn't define named domain constants; the
    // utilization[] array is indexed by domain in a fixed order (see the
    // NvAPI_GPU_GetDynamicPstatesInfoEx docs in nvapi.h): 0=GPU (graphics
    // engine), 1=FB (frame buffer = graphics-memory bandwidth), 2=VID, 3=BUS.
    constexpr int kDomainGpu = 0;
    constexpr int kDomainFb  = 1;

    bool any = false;
    const auto& g = info.utilization[kDomainGpu];
    const auto& f = info.utilization[kDomainFb];
    if (g.bIsPresent) {
        gpu = g.percentage > 100u ? 100u : g.percentage;
        any = true;
    }
    if (f.bIsPresent) {  // FB = frame buffer = graphics-memory bandwidth util
        mem = f.percentage > 100u ? 100u : f.percentage;
        any = true;
    }
    return any;
}
#endif

}  // namespace gpufl::nvidia
