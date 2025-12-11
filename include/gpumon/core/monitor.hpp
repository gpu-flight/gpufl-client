#ifndef GPUMON_MONITOR_HPP
#define GPUMON_MONITOR_HPP
#include "common.hpp"
#include <thread>
#include <functional>
#include <utility>

namespace gpumon {
    // ============================================================================
    // Public Management API
    // ============================================================================

    // Initializes both the Logging System AND the GPU Backend
    inline bool init(const InitOptions& opts) {
        auto& state = detail::getState();
        std::lock_guard initLock(detail::getInitMutex());

        if (state.initialized) return false;

        state.appName = opts.appName;
        state.pid = detail::getPid();
        state.sampleIntervalMs = opts.sampleIntervalMs;

        // Call Backend Init (No-op for CUDA, but needed for pattern)
        backend::initialize();

        std::string basePath = opts.logPath;
        if (basePath.empty()) {
            basePath = "gpumon"; // Fallback to local CWD file if user provided nothing
        }

        // Helper lambda to open file
        auto openLog = [&](const LogCategory cat, const std::string &suffix) {
            const std::string path = detail::generateLogPath(basePath, suffix);
            state.logFiles[cat].open(path, std::ios::out | std::ios::app);
        };

        state.initialized = true; // Set true early so we can open files

        openLog(LogCategory::Kernel, "kernel");
        openLog(LogCategory::Scope,  "scope");
        openLog(LogCategory::System, "system");

        // 4. Log Init Event to ALL files (Meta)
        std::ostringstream oss;
        oss << R"({"type":"init",)"
            << "\"pid\":" << state.pid << ","
            << R"("app":")" << detail::escapeJson(state.appName) << "\","
            << "\"ts_ns\":" << detail::getTimestampNs()
            << "}";

        detail::writeLogLine(LogCategory::Meta, oss.str());

        return true;
    }

    inline void shutdown() {
        auto& state = detail::getState();
        if (!state.initialized) return;

        // Log shutdown event to ALL files (Meta)
        std::ostringstream oss;
        oss << R"({"type":"shutdown",)"
            << "\"pid\":" << state.pid << ","
            << R"("app":")" << detail::escapeJson(state.appName) << "\","
            << "\"ts_ns\":" << detail::getTimestampNs()
            << "}";
        detail::writeLogLine(LogCategory::Meta, oss.str());

        // Backend Shutdown (GENERIC CALL)
        backend::shutdown();
        std::lock_guard lock(state.logMutex);
        state.initialized = false;

        // Close all files
        for (auto& pair : state.logFiles) {
            if (pair.second.is_open()) pair.second.close();
        }
        state.logFiles.clear();
    }

    // ============================================================================
    // Scoped Monitor
    // ============================================================================

    class ScopedMonitor {
    public:
        explicit ScopedMonitor(std::string name, std::string tag = "")
            : name_(std::move(name)), tag_(std::move(tag)),
              tsStart_(detail::getTimestampNs()), stopSampling_(false) {

            const auto& state = detail::getState();
            if (!state.initialized) return;

            // 1. Log Scope Begin with initial snapshot
            auto snapshots = backend::get_device_snapshots();
            logScopeEvent(LogCategory::Scope, "scope_begin", tsStart_, snapshots);
            // 2. Start Background Sampler (if configured)
            if (state.sampleIntervalMs > 0) {
                samplerThread_ = std::thread(&ScopedMonitor::samplingLoop, this, state.sampleIntervalMs);
            }
        }

        ~ScopedMonitor() {
            // 1. Stop Sampler
            if (samplerThread_.joinable()) {
                stopSampling_ = true;
                samplerThread_.join();
            }

            const auto& state = detail::getState();
            if (!state.initialized) return;

            // 2. Synchronize GPU (GENERIC CALL)
            // Ensures we measure full GPU execution time, not just CPU launch time
            backend::synchronize();

            int64_t tsEnd = detail::getTimestampNs();

            // 3. Take Final Snapshot
            auto snapshots = backend::get_device_snapshots();

            // 4. Log Scope End
            logScopeEvent(LogCategory::Scope, "scope_end", tsEnd, snapshots, tsStart_);
        }

        // Disable copy/move
        ScopedMonitor(const ScopedMonitor&) = delete;
        ScopedMonitor& operator=(const ScopedMonitor&) = delete;

    private:
        std::string name_;
        std::string tag_;
        int64_t tsStart_;

        std::atomic<bool> stopSampling_;
        std::thread samplerThread_;

        // Helper to format and write the JSON
        void logScopeEvent(const LogCategory category, const char* type, const int64_t timestamp,
                      const std::vector<detail::DeviceSnapshot>& snapshots,
                      const int64_t startTime = 0) {

            const auto& state = detail::getState();
            std::ostringstream oss;

            oss << "{\"type\":\"" << type << "\","
                << "\"pid\":" << state.pid << ","
                << R"("app":")" << detail::escapeJson(state.appName) << "\","
                << R"("name":")" << detail::escapeJson(name_) << "\"";

            if (!tag_.empty()) {
                oss << R"(,"tag":")" << detail::escapeJson(tag_) << "\"";
            }

            if (std::string(type) == "scope_end") {
                oss << ",\"ts_start_ns\":" << startTime
                    << ",\"ts_end_ns\":" << timestamp
                    << ",\"duration_ns\":" << (timestamp - startTime);
            } else {
                oss << ",\"ts_ns\":" << timestamp;
            }

            detail::writeDeviceJson(oss, snapshots);
            oss << "}";

            detail::writeLogLine(category, oss.str());
        }

        void samplingLoop(uint32_t intervalMs) {
            while (!stopSampling_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
                if (stopSampling_) break;
                auto snapshots = backend::get_device_snapshots();

                int64_t now = detail::getTimestampNs();

                logScopeEvent(LogCategory::Scope, "scope_sample", now, snapshots);
            }
        }
    };

    // ============================================================================
    // System Monitor
    // ============================================================================
    class SystemMonitor {
    public:
        explicit SystemMonitor(uint32_t intervalMs = 1000)
            : intervalMs_(intervalMs), stop_(false) {

            worker_ = std::thread(&SystemMonitor::loop, this);
        }
        void await() {
            if (worker_.joinable()) {
                worker_.join();
            }
        }

        ~SystemMonitor() {
            stop_ = true;
            if (worker_.joinable()) worker_.join();
        }

    private:
        uint32_t intervalMs_;
        std::atomic<bool> stop_;
        std::thread worker_;
        std::string name_ = "system";

        void loop() {
            while (!stop_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs_));
                if (stop_) break;

                auto snapshots = backend::get_device_snapshots();

                const int64_t now = detail::getTimestampNs();
                const auto& state = detail::getState();

                std::ostringstream oss;
                oss << "{\"type\":\"system_sample\","
                    << R"("app":")" << detail::escapeJson(state.appName) << "\","
                    << R"("name":")" << detail::escapeJson(name_) << "\","
                    << "\"ts_ns\":" << now;

                detail::writeDeviceJson(oss, snapshots);
                oss << "}";

                detail::writeLogLine(LogCategory::System, oss.str());
            }
        }
    };


    inline void monitor(const std::string& name, const std::function<void()> &fn, const std::string& tag = "") {
        ScopedMonitor monitor(name, tag);
        fn();
    }
}

// ============================================================================
// Macros
// ============================================================================

#define GPUMON_SCOPE(name) \
if (gpumon::ScopedMonitor _gpumon_scope{name}; true)

#define GPUMON_SCOPE_TAGGED(name, tag) \
if (gpumon::ScopedMonitor _gpumon_scope{name, tag}; true)
#endif

#define GPUMON_SYSTEM_START(interval) \
gpumon::SystemMonitor _sys_mon{interval}; \
while(true) { std::this_thread::sleep_for(std::chrono::seconds(1)); }