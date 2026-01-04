#pragma once
#include <string>


namespace gpufl::core {
    /**
     * Captures the current call stack and returns it as a string.
     * Format: "main::FunctionA::FunctionB"
     * * @param skipFrames Number of top frames to skip (default 1 to skip GetCallStack itself)
     */
    std::string GetCallStack(int skipFrames = 1);
}
