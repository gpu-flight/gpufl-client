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
        // Errors ALWAYS print, regardless of the debug-output flag.
        // The earlier behavior (gated on isEnabled()) silenced
        // legitimate failure diagnostics from init() / FileLogSink /
        // CUPTI, so users hit mysterious crashes (e.g. permission
        // denied on a cross-container volume mount with stale UID
        // ownership) with zero stderr output to diagnose from. Debug
        // verbosity is a knob for normal-running noise; failures are
        // not normal-running noise.
        std::stringstream ss;
        ss << prefix;
        (ss << ... << std::forward<Args>(args));
        std::cerr << ss.str() << std::endl;
    }
};

#define GFL_LOG_DEBUG(...) ::gpufl::DebugLogger::log("[GPUFL] ", __VA_ARGS__)
#define GFL_LOG_ERROR(...) \
    ::gpufl::DebugLogger::error("[GPUFL-ERROR] " __FILE__ ":", __LINE__, ": ", __VA_ARGS__)

}  // namespace gpufl
