#ifndef GPUMON_CUDA_HPP
#define GPUMON_CUDA_HPP
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include "../core/common.hpp"

#ifdef GPUMON_ENABLE_NVML
    #include <nvml.h>
#endif
namespace gpumon {
    namespace backend {
        inline void initialize() {
            #ifdef GPUMON_ENABLE_NVML
                nvmlInit(); // Initialize NVML if enabled
            #endif
        }
        inline void shutdown() {
            #ifdef GPUMON_ENABLE_NVML
                nvmlShutdown();
            #endif
        }
        inline void synchronize() { cudaDeviceSynchronize(); }

        inline std::string formatUuid(const char* bytes) {
            std::ostringstream oss;
            oss << "GPU-";
            oss << std::hex << std::setfill('0');
            for (int i = 0; i < 16; ++i) {
                oss << std::setw(2) << (0xFF & bytes[i]);
                if (i == 3 || i == 5 || i == 7 || i == 9) oss << "-";
            }
            return oss.str();
        }

        // CACHE: Stores Name/UUID per device ID to avoid slow driver calls
        struct DeviceStaticInfo {
            std::string name;
            std::string uuid;
            int pciBusId;
            size_t totalMiB;
        };

        // Helper to get static info (cached)
        inline DeviceStaticInfo get_cached_static_info(int devId) {
            static std::map<int, DeviceStaticInfo> cache;
            static std::mutex cacheMutex;

            std::lock_guard<std::mutex> lock(cacheMutex);
            if (cache.find(devId) == cache.end()) {
                // Not in cache, query driver
                cudaDeviceProp p;
                if (cudaGetDeviceProperties(&p, devId) == cudaSuccess) {
                    cache[devId] = {
                        p.name,
                        formatUuid(p.uuid.bytes),
                        p.pciBusID,
                        p.totalGlobalMem / (1024 * 1024)
                    };
                } else {
                    cache[devId] = { "Unknown", "GPU-Unknown", -1, 0 };
                }
            }
            return cache[devId];
        }

        inline std::vector<detail::DeviceSnapshot> get_device_snapshots() {
            std::vector<detail::DeviceSnapshot> snapshots;

#ifdef GPUMON_ENABLE_NVML
            {
                unsigned int deviceCount = 0;
                if (nvmlDeviceGetCount(&deviceCount) != NVML_SUCCESS || deviceCount == 0) return snapshots;

                for (unsigned int i = 0; i < deviceCount; ++i) {
                    nvmlDevice_t nvDev;
                    if (nvmlDeviceGetHandleByIndex(i, &nvDev) != NVML_SUCCESS) continue;

                    detail::DeviceSnapshot snap;
                    snap.deviceId = static_cast<int>(i);

                    // 1. Identity (From NVML directly)
                    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
                    char uuid[NVML_DEVICE_UUID_BUFFER_SIZE];
                    nvmlPciInfo_t pci;

                    if (nvmlDeviceGetName(nvDev, name, sizeof(name)) == NVML_SUCCESS) snap.name = name;
                    if (nvmlDeviceGetUUID(nvDev, uuid, sizeof(uuid)) == NVML_SUCCESS) snap.uuid = uuid;
                    if (nvmlDeviceGetPciInfo(nvDev, &pci) == NVML_SUCCESS) snap.pciBusId = pci.bus;

                    // 2. Memory (Physical state, no context overhead)
                    nvmlMemory_t mem;
                    if (nvmlDeviceGetMemoryInfo(nvDev, &mem) == NVML_SUCCESS) {
                        snap.totalMiB = mem.total / (1024 * 1024);
                        snap.freeMiB  = mem.free  / (1024 * 1024);
                        snap.usedMiB  = mem.used  / (1024 * 1024);
                    }

                    // 3. Extended Telemetry
                    nvmlUtilization_t util;
                    if (nvmlDeviceGetUtilizationRates(nvDev, &util) == NVML_SUCCESS) {
                        snap.gpuUtil = util.gpu;
                        snap.memUtil = util.memory;
                    }
                    unsigned int val = 0;
                    if (nvmlDeviceGetTemperature(nvDev, NVML_TEMPERATURE_GPU, &val) == NVML_SUCCESS) snap.tempC = val;
                    if (nvmlDeviceGetPowerUsage(nvDev, &val) == NVML_SUCCESS) snap.powermW = val;
                    if (nvmlDeviceGetClockInfo(nvDev, NVML_CLOCK_GRAPHICS, &val) == NVML_SUCCESS) snap.clockGfx = val;
                    if (nvmlDeviceGetClockInfo(nvDev, NVML_CLOCK_SM, &val) == NVML_SUCCESS)       snap.clockSm = val;
                    if (nvmlDeviceGetClockInfo(nvDev, NVML_CLOCK_MEM, &val) == NVML_SUCCESS)      snap.clockMem = val;

                    snapshots.push_back(snap);
                }
                return snapshots;
            }
#else
            {
                int deviceCount = 0;
                if (cudaError_t err = cudaGetDeviceCount(&deviceCount); err != cudaSuccess || deviceCount == 0) return snapshots;

                // Save current device to restore later (polite behavior)
                int currentDevice = -1;
                cudaGetDevice(&currentDevice);

                for (int i = 0; i < deviceCount; ++i) {
                    cudaSetDevice(i);

                    // 1. Get Dynamic Data (Memory)
                    size_t free = 0, total = 0;
                    if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
                        cudaSetDevice(i); // This creates/activates context

                        detail::DeviceSnapshot snap;
                        snap.deviceId = i;

                        // Identity (Cached via CUDA Props)
                        auto staticInfo = get_cached_static_info(i);
                        snap.name = staticInfo.name;
                        snap.uuid = staticInfo.uuid;
                        snap.pciBusId = staticInfo.pciBusId;

                        // Memory (Context-aware)
                        size_t free = 0, total = 0;
                        if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
                            snap.freeMiB = free / (1024 * 1024);
                            snap.totalMiB = total / (1024 * 1024);
                            snap.usedMiB = snap.totalMiB - snap.freeMiB;
                        }

                        snapshots.push_back(snap);
                    }
                }
                if (currentDevice >= 0) cudaSetDevice(currentDevice);
                return snapshots;
            }
#endif
        }

        // Get info for the CURRENTLY active device (for Kernel Launch)
        inline detail::DeviceSnapshot get_current_device_snapshot() {
            int dev = 0;
            cudaGetDevice(&dev);

            detail::DeviceSnapshot snap;
            snap.deviceId = dev;

            // Static
            const auto staticInfo = get_cached_static_info(dev);
            snap.name = staticInfo.name;
            snap.uuid = staticInfo.uuid;
            snap.pciBusId = staticInfo.pciBusId;
            snap.totalMiB = staticInfo.totalMiB;

            return snap;
        }

        inline std::string formatDim3(const dim3& d) {
            std::ostringstream oss;
            oss << "(" << d.x << "," << d.y << "," << d.z << ")";
            return oss.str();
        }

        template <typename T>
        inline const cudaFuncAttributes& get_kernel_static_attrs(T kernel) {
            static cudaFuncAttributes attrs = {};
            static bool initialized = false;

            // Thread-safe initialization (C++11 standard guarantees)
            if (!initialized) {
                // If this fails (e.g. invalid kernel), attrs will remain 0-filled
                cudaFuncGetAttributes(&attrs, kernel);
                initialized = true;
            }
            return attrs;
        }
    }

    namespace detail {
        inline void logKernelEvent(
        const std::string& kernelName,
        const int64_t tsStartNs,
        const int64_t tsEndNs,
        const dim3& grid,
        const dim3& block,
        const size_t dynamicSharedMemBytes,
        const std::string& cudaError,
        const cudaFuncAttributes& attrs,
        const std::string& tag = "")
        {
            const State& state = getState();
            if (!state.initialized) return;
            const auto dev = backend::get_current_device_snapshot();

            std::ostringstream oss;
            oss << R"({"type":"kernel",)"
                << "\"pid\":" << state.pid << ","
                << R"("app":")" << escapeJson(state.appName) << "\","

                << "\"devices\":[{"
                << "\"id\":" << dev.deviceId << ","
                << R"("name":")" << escapeJson(dev.name) << "\","
                << R"("uuid":")" << dev.uuid << "\","
                << "\"pci_bus\":" << dev.pciBusId << ","
                << "\"total_mib\":" << dev.totalMiB
                << "}],"
                << R"("name":")" << escapeJson(kernelName) << "\","
                << "\"ts_start_ns\":" << tsStartNs << ","
                << "\"ts_end_ns\":" << tsEndNs << ","
                << "\"duration_ns\":" << (tsEndNs - tsStartNs) << ","
                << R"("grid":")" << backend::formatDim3(grid) << "\","
                << R"("block":")" << backend::formatDim3(block) << "\","
                << "\"dyn_shared_bytes\":" << dynamicSharedMemBytes << ","
                << "\"num_regs\":" << attrs.numRegs << ","
                << "\"static_shared_bytes\":" << attrs.sharedSizeBytes << ","
                << "\"local_bytes\":" << attrs.localSizeBytes << ","
                << "\"const_bytes\":" << attrs.constSizeBytes;

            if (!tag.empty()) oss << R"(,"tag":")" << escapeJson(tag) << "\"";

            oss << R"(,"cuda_error":")" << escapeJson(cudaError) << "\"}";
            writeLogLine(LogCategory::Kernel, oss.str());
        }

        inline const char* getCudaErrorString(const cudaError_t error) {
            return ::cudaGetErrorString(error);
        }
    }
}

#define GPUMON_LAUNCH(kernel, grid, block, sharedMem, stream, ...) \
    do { \
        int64_t _ts = gpumon::detail::getTimestampNs(); \
        kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
        cudaError_t _err = cudaGetLastError(); \
        cudaDeviceSynchronize(); \
        int64_t _te = gpumon::detail::getTimestampNs(); \
        /* Fetch Static Attributes (Cached) */ \
        auto& _attrs = gpumon::backend::get_kernel_static_attrs(kernel); \
        gpumon::detail::logKernelEvent(#kernel, _ts, _te, grid, block, sharedMem, gpumon::detail::getCudaErrorString(_err), _attrs); \
    } while(0)

// wraps a single kernel launch with custom tag
#define GPUMON_LAUNCH_TAGGED(tag, kernel, grid, block, sharedMem, stream, ...) \
    do { \
        int64_t _ts = gpumon::detail::getTimestampNs(); \
        kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
        cudaError_t _err = cudaGetLastError(); \
        cudaDeviceSynchronize(); \
        int64_t _te = gpumon::detail::getTimestampNs(); \
        /* Fetch Static Attributes (Cached) */ \
        auto& _attrs = gpumon::backend::get_kernel_static_attrs(kernel); \
        gpumon::detail::logKernelEvent(#kernel, _ts, _te, grid, block, sharedMem, gpumon::detail::getCudaErrorString(_err), _attrs, tag); \
    } while(0)

#endif
