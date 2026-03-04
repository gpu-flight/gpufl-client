#pragma once
#include <atomic>
#include <iostream>
#include <sstream>
#include <string>

namespace gpufl {

class DebugLogger {
   public:
    static void setEnabled(bool enabled);
    static bool isEnabled();

    template <typename... Args>
    static void log(const char* prefix, Args&&... args) {
        if (isEnabled()) {
            std::stringstream ss;
            ss << prefix;
            (ss << ... << std::forward<Args>(args));
            std::cout << ss.str() << std::endl;
        }
    }

    template <typename... Args>
    static void error(const char* prefix, Args&&... args) {
        // Errors might be shown even if debug is off?
        // The issue says "so many if statements like
        // backend->GetOptions().enable_debug_output then print out the logs" So
        // I should probably stick to what it was doing.
        if (isEnabled()) {
            std::stringstream ss;
            ss << prefix;
            (ss << ... << std::forward<Args>(args));
            std::cerr << ss.str() << std::endl;
        }
    }
};

#define GFL_LOG_DEBUG(...) ::gpufl::DebugLogger::log("[GPUFL] ", __VA_ARGS__)
#define GFL_LOG_ERROR(...) \
    ::gpufl::DebugLogger::error("[GPUFL-ERROR] ", __VA_ARGS__)

}  // namespace gpufl
