#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gpufl {
namespace report {

// Per-function collected data used to generate performance hints.
// Stall keys are already resolved to display names before insertion.
struct FuncProfile {
    std::map<std::string, uint64_t> stalls;  // display-name → count
    uint64_t totalStalls   = 0;
    uint64_t warpInsts     = 0;
    uint64_t threadInsts   = 0;
    uint64_t globalSectors = 0;
    uint64_t idealSectors  = 0;
};

// Returns zero or more actionable hint strings for a single kernel/function profile.
std::vector<std::string> computeHints(const FuncProfile& fp);

}  // namespace report
}  // namespace gpufl
