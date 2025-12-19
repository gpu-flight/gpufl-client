#pragma once
#include <cstdint>
#include <string>

namespace gpufl::detail {
    int64_t getTimestampNs();
    int getPid();
    std::string toIso8601Utc();
}