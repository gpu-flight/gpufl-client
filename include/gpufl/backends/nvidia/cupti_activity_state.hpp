#pragma once

#include <cstdint>
#include <string>

namespace gpufl {

struct NvtxOpenRange {
    std::string name;
    std::string domain;
    uint64_t start_ts = 0;
};

void ResetCuptiActivityCompanionState();
void ClearExternalCorrelationState();

void StoreExternalCorrelation(uint32_t corr_id, uint8_t kind, uint64_t id);
bool LookupAndPopExternalCorrelation(uint32_t corr_id,
                                     uint8_t* kind_out,
                                     uint64_t* id_out);

void StoreSourceLocator(uint32_t id, const char* file_name,
                        uint32_t line_number);
void StoreFunctionName(uint32_t id, const char* name);
bool LookupSourceLocator(uint32_t id, std::string* file_name,
                         uint32_t* line_number);
bool LookupFunctionName(uint32_t id, std::string* name);

void StoreNvtxMarkerStart(uint32_t id, std::string name, std::string domain,
                          uint64_t start_ts);
bool PopNvtxMarker(uint32_t id, NvtxOpenRange* out);

void pushExternalCorrelation(uint32_t kind, uint64_t id);
void popExternalCorrelation(uint32_t kind);

}  // namespace gpufl
