#pragma once
#include <string>
#include <cstdint>

namespace gpufl::nvidia {
    struct SourceCorrelation {
        std::string fileName;
        std::string dirName;
        uint32_t lineNumber = 0;
    };

    class CuptiSass {
    public:
        static SourceCorrelation sampleSourceCorrelation(const void* cubin, size_t cubinSize, const char* functionName, uint64_t pcOffset);
    };
}
