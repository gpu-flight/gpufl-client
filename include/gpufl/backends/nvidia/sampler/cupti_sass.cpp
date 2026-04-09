#include "cupti_sass.hpp"

#include <cupti_pcsampling.h>

#include <cstdlib>

#include "gpufl/core/debug_logger.hpp"

namespace gpufl::nvidia {
namespace {
void FreeCuptiCorrelationString(char* s) {
    if (!s) return;
#if defined(_WIN32) && defined(_DEBUG)
    // CUPTI may allocate these with a different CRT heap than the debug app.
    // Avoid invalid heap-pointer assertions in Debug on Windows.
    (void)s;
#else
    std::free(s);
#endif
}
}  // namespace

SourceCorrelation CuptiSass::sampleSourceCorrelation(const void* cubin,
                                                     size_t cubinSize,
                                                     const char* functionName,
                                                     uint64_t pcOffset) {
    CUpti_GetSassToSourceCorrelationParams params = {
        CUpti_GetSassToSourceCorrelationParamsSize};
    params.cubin = cubin;
    params.cubinSize = cubinSize;
    params.functionName = functionName;
    params.pcOffset = pcOffset;

    CUptiResult res = cuptiGetSassToSourceCorrelation(&params);
    if (res != CUPTI_SUCCESS) {
        const char* err;
        cuptiGetResultString(res, &err);
        GFL_LOG_ERROR("[SASS Metrics] cuptiGetSassToSourceCorrelation FAILED: ",
                      err);
        return {};
    }

    SourceCorrelation result = {};
    if (params.fileName) {
        result.fileName = params.fileName;
        FreeCuptiCorrelationString(params.fileName);
    }
    if (params.dirName) {
        result.dirName = params.dirName;
        FreeCuptiCorrelationString(params.dirName);
    }
    result.lineNumber = params.lineNumber;

    return result;
}
}  // namespace gpufl::nvidia
