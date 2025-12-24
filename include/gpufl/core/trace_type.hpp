#pragma once
#include <cstdint>

namespace gpufl {
    enum class TraceType : uint8_t {
        KERNEL,
        RANGE
    };
}