#pragma once

#include <string>
#include <vector>

#include "cli_parse.hpp"
#include "gpufl/core/events.hpp"

namespace gpufl::launcher {

struct InfoReport {
    int schema_version = 1;
    int cuda_driver_version = 0;
    int cuda_runtime_version = 0;
    std::vector<GpuStaticDeviceInfo> devices;
};

// Rendering is separate from collection so the schema can be unit-tested on
// machines without a GPU.
std::string renderInfoJson(const InfoReport& report);
std::string renderInfoText(const InfoReport& report);

// Queries the local backend once and prints either text or JSON. Returns 0 on
// success, 2 for a missing requested device, and 3 when no GPU can be queried.
int runInfo(const InfoArgs& args);

}  // namespace gpufl::launcher
