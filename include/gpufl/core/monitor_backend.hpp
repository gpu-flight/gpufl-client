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
         * @brief Initialize the monitoring backend with given options.
         * @param opts Configuration options for monitoring
         */
        virtual void Initialize(const MonitorOptions& opts) = 0;

        /**
         * @brief Shutdown the monitoring backend and release resources.
         */
        virtual void Shutdown() = 0;

        /**
         * @brief Start active monitoring/tracing.
         */
        virtual void Start() = 0;

        /**
         * @brief Stop active monitoring/tracing.
         */
        virtual void Stop() = 0;
    };

} // namespace gpufl
