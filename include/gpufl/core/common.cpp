#include "gpufl/core/common.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

namespace gpufl::detail {
    int64_t getTimestampNs() {
        using namespace std::chrono;
        return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
    }


    int getPid() {
#ifdef _WIN32
        return static_cast<int>(GetCurrentProcessId());
#else
        return static_cast<int>(getpid());
#endif
    }

    std::string toIso8601Utc() {
        using namespace std::chrono;
        const auto now = system_clock::now();
        auto tt = system_clock::to_time_t(now);
#if defined(_WIN32)
        std::tm tm_utc;
        gmtime_s(&tm_utc, &tt);
        const auto* ptm = &tm_utc;
#else
        std::tm tm_utc;
        gmtime_r(&tt, &tm_utc);
        auto* ptm = &tm_utc;
#endif
        std::ostringstream oss;
        oss << std::put_time(ptm, "%Y-%m-%dT%H:%M:%SZ");
        return oss.str();
    }
}


