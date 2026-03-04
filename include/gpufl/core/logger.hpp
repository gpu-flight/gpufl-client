#pragma once
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gpufl/core/events.hpp"

namespace gpufl {

struct DeviceSample;

class Logger {
   public:
    struct Options {
        std::string base_path;
        std::size_t rotate_bytes = 64 * 1024 * 1024;  // 64 MiB default
        bool flush_always = false;
        int system_sample_rate_ms = 0;
    };

    Logger();
    ~Logger();

    bool open(const Options& opt);
    void close();

    // Lifecycle: writes to ALL active channels
    void logInit(const InitEvent& e) const;
    void logShutdown(const ShutdownEvent& e) const;

    // Device channel (kernel, memcpy, memset)
    void logKernelEvent(const KernelEvent& e) const;
    void logMemcpyEvent(const MemcpyEvent& e) const;
    void logMemsetEvent(const MemsetEvent& e) const;

    // Scope channel
    void logScopeBegin(const ScopeBeginEvent& e) const;
    void logScopeEnd(const ScopeEndEvent& e) const;

    void logProfileSample(const ProfileSampleEvent& e) const;

    // System channel
    void logSystemStart(const SystemStartEvent& e) const;
    void logSystemStop(const SystemStopEvent& e) const;
    void logSystemSample(const SystemSampleEvent& e) const;

    static std::string hostToJson(const HostSample& h);

   private:
    class LogChannel {
       public:
        LogChannel(std::string name, Options opt);
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
        size_t current_bytes_ = 0;

        mutable std::mutex mu_;
        bool opened_ = false;
    };

    Options opt_;

    // Channels for different event categories
    std::unique_ptr<LogChannel>
        chanDevice_;  // kernel_event/memcpy_event/memset_event
    std::unique_ptr<LogChannel> chanScope_;   // scope_begin/end/sample
    std::unique_ptr<LogChannel> chanSystem_;  // system_start/stop/sample
};
}  // namespace gpufl