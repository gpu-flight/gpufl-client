#pragma once

#include <cuda_runtime.h>
#include <string>
#include "gpufl/core/trace_type.hpp"

namespace gpufl {

    struct MonitorOptions {
        bool collect_kernel_details = false;
        bool enable_debug_output = false;
    };

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

    private:
        Monitor() = delete;
    };

    /**
     * @brief RAII helper for manual range monitoring.
     */
    class ScopedRange {
    public:
        explicit ScopedRange(const char* name) {
            Monitor::PushRange(name);
        }

        ~ScopedRange() {
            Monitor::PopRange();
        }

        ScopedRange(const ScopedRange&) = delete;
        ScopedRange& operator=(const ScopedRange&) = delete;
    };

}