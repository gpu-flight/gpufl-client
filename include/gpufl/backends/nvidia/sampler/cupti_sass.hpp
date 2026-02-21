#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct CUpti_SassMetrics_Data;

namespace gpufl::nvidia {
    struct SourceCorrelation {
        std::string fileName;
        std::string dirName;
        uint32_t lineNumber = 0;
    };

    struct SassMetricData {
        uint32_t cubinCrc;
        uint32_t functionIndex;
        std::string functionName;
        uint32_t pcOffset;
        struct MetricValue {
            uint64_t metricId;
            uint64_t value;
        };
        std::vector<MetricValue> values;
    };

    class CuptiSass {
    public:
        static SourceCorrelation sampleSourceCorrelation(const void* cubin, size_t cubinSize, const char* functionName, uint64_t pcOffset);
        
        // SASS Metrics API wrapper
        static bool setMetricsConfig(uint32_t deviceIndex, const std::vector<std::string>& metricNames);
        static bool unsetMetricsConfig(uint32_t deviceIndex);
        static bool enableMetrics(void* ctx = nullptr);
        static bool disableMetrics(void* ctx = nullptr);
        static std::vector<SassMetricData> flushMetricsData(void* ctx = nullptr);
    };
}
