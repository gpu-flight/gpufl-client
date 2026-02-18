#pragma once

#include <cuda_runtime.h>
#include <string>
#include "gpufl/core/trace_type.hpp"

namespace gpufl {

    struct MonitorOptions {
        bool collectKernelDetails = false;
        bool enableDebugOutput = false;
        bool enableStackTrace = false;
        bool isProfiling = false;
        int kernelSampleRateMs = 0;
    };

    enum class MonitorMode : unsigned int {
        None = 0,
        Monitoring = 1 << 0,
        Profiling = 1 << 1,

        Default = Monitoring
    };

    inline constexpr MonitorMode operator|(MonitorMode lhs, MonitorMode rhs) {
        using T = std::underlying_type_t<MonitorMode>;
        return static_cast<MonitorMode>(static_cast<T>(lhs) | static_cast<T>(rhs));
    }

    inline constexpr MonitorMode operator&(MonitorMode lhs, MonitorMode rhs) {
        using T = std::underlying_type_t<MonitorMode>;
        return static_cast<MonitorMode>(static_cast<T>(lhs) & static_cast<T>(rhs));
    }

    inline constexpr MonitorMode& operator|=(MonitorMode& lhs, MonitorMode rhs) {
        lhs = lhs | rhs;
        return lhs;
    }

    inline constexpr bool hasFlag(MonitorMode value, MonitorMode flag) {
        return (static_cast<unsigned int>(value) & static_cast<unsigned int>(flag)) != 0;
    }

    /**
     * @brief The central monitoring engine.
     */
    class Monitor {
    public:
        /**
         * @brief Initializes the monitoring engine.
         */
        static void Initialize(const MonitorOptions& opts = {});

        /**
         * @brief Shuts down the monitoring engine.
         */
        static void Shutdown();

        /**
         * @brief Starts global collection.
         */
        static void Start();

        /**
         * @brief Stops global collection.
         */
        static void Stop();

        /**
         * @brief Marks the start of a logical section.
         */
        static void PushRange(const char* name);

        /**
         * @brief Marks the end of the current section.
         */
        static void PopRange();

        /**
         * @brief Internal API for backends to record events.
         */
        static void RecordStart(const char* name, cudaStream_t stream, TraceType type, void** outHandle);
        static void RecordStop(void* handle, cudaStream_t stream);

        /**
         * @brief Profiler Scope Control
         */
        static void BeginProfilerScope(const char* name);
        static void EndProfilerScope(const char* name);

    private:
        Monitor() = delete;
    };

}