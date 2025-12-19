#pragma once
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <memory>

#include "gpufl/core/events.hpp"

namespace gpufl {

    struct DeviceSample;

    class Logger {
    public:
        struct Options {
            std::string basePath;
            std::size_t rotateBytes = 64 * 1024 * 1024; // 64 MiB default
            bool flushAlways = false;
            int systemSampleRateMs = 0;
        };

        Logger();
        ~Logger();

        bool open(const Options& opt);
        void close();

        // Lifecycle: writes to ALL active channels
        void logInit(const InitEvent& e) const;
        void logShutdown(const ShutdownEvent& e) const;

        // Kernel channel
        void logKernelBegin(const KernelBeginEvent& e) const;
        void logKernelEnd(const KernelEndEvent& e) const;
        void logKernelSample(const KernelSampleEvent& e) const;

        // Scope channel
        void logScopeBegin(const ScopeBeginEvent& e) const;
        void logScopeEnd(const ScopeEndEvent& e) const;
        void logScopeSample(const ScopeSampleEvent& e) const;

        // System channel
        void logSystemStart(const SystemStartEvent& e) const;
        void logSystemStop(const SystemStopEvent& e) const;
        void logSystemSample(const SystemSampleEvent& e) const;

        static std::string hostToJson(const HostSample& h);

    private:
        class LogChannel {
        public:
            LogChannel(std::string name, Options  opt);
            ~LogChannel();

            void write(const std::string& line);
            void close();
            bool isOpen() const;

        private:
            void ensureOpenLocked();
            void rotateLocked();
            [[nodiscard]] std::string makePathLocked() const;
            void closeLocked();

            std::string name_;
            Options opt_;

            std::ofstream stream_;
            int index_ = 0;
            size_t currentBytes_ = 0;

            mutable std::mutex mu_;
            bool opened_ = false;
        };

        Options opt_;

        // Channels for different event categories
        std::unique_ptr<LogChannel> chanKernel_;    // kernel
        std::unique_ptr<LogChannel> chanScope_;     // scope_begin/end/sample
        std::unique_ptr<LogChannel> chanSystem_;    // system_start/stop/sample
    };
}