#pragma once

#include "gpufl/core/monitor.hpp"

namespace gpufl {

    /**
     * @brief Interface for backend-specific monitoring implementations.
     *
     * Backends implement this interface to provide platform-specific
     * kernel and event monitoring (e.g., CUPTI for NVIDIA, ROCTracer for AMD).
     */
    class IMonitorBackend {
    public:
        virtual ~IMonitorBackend() = default;

        /**
         * @brief initialize the monitoring backend with given options.
         * @param opts Configuration options for monitoring
         */
        virtual void initialize(const MonitorOptions& opts) = 0;

        /**
         * @brief shutdown the monitoring backend and release resources.
         */
        virtual void shutdown() = 0;

        /**
         * @brief start active monitoring/tracing.
         */
        virtual void start() = 0;

        /**
         * @brief stop active monitoring/tracing.
         */
        virtual void stop() = 0;

        virtual bool isMonitoringMode() = 0;

        virtual bool isProfilingMode() = 0;

        virtual void onScopeStart(const char* name) {}
        virtual void onScopeStop(const char* name) {}
    };

} // namespace gpufl
