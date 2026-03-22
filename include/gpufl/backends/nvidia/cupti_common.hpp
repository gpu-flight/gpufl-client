#pragma once

#include <cupti.h>

#include <utility>
#include <vector>

#define CUPTI_CHECK(call)                                            \
    do {                                                             \
        CUptiResult res = (call);                                    \
        if (res != CUPTI_SUCCESS) {                                  \
            const char* errStr;                                      \
            cuptiGetResultString(res, &errStr);                      \
            ::gpufl::DebugLogger::error("[GPUFL Monitor] ", errStr); \
        }                                                            \
    } while (0)

#define CUPTI_CHECK_RETURN(call, failMsg)                               \
    do {                                                                \
        CUptiResult res = (call);                                       \
        if (res != CUPTI_SUCCESS) {                                     \
            ::gpufl::DebugLogger::error("[GPUFL Monitor] ", (failMsg)); \
            return;                                                     \
        }                                                               \
    } while (0)

namespace gpufl {

/**
 * @brief Cubin binary data keyed by CRC, shared between CuptiBackend and
 * engines.
 */
struct CubinInfo {
    std::vector<uint8_t> data;
    uint64_t crc = 0;
};

struct LaunchMeta {
    int64_t api_enter_ns = 0;
    int64_t api_exit_ns = 0;
    bool has_details = false;
    int grid_x = 0, grid_y = 0, grid_z = 0;
    int block_x = 0, block_y = 0, block_z = 0;
    int dyn_shared = 0, static_shared = 0, local_bytes = 0, const_bytes = 0,
        num_regs = 0;
    float occupancy = 0.0f;
    int max_active_blocks = 0;
    char name[128]{};
    char user_scope[256]{};
    int scope_depth{};
    size_t stack_id{};
};

class ICuptiHandler {
   public:
    virtual ~ICuptiHandler() = default;
    virtual bool shouldHandle(CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid) const = 0;
    virtual void handle(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                        const void* cbdata) = 0;
    virtual const char* getName() const = 0;

    // Subscription requirements — used by CuptiBackend at
    // initialize()/shutdown()
    virtual std::vector<CUpti_CallbackDomain> requiredDomains() const {
        return {};
    }
    virtual std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
    requiredCallbacks() const {
        return {};
    }

    // Activity kind requirements — used by CuptiBackend at start()/stop()
    virtual std::vector<CUpti_ActivityKind> requiredActivityKinds() const {
        return {};
    }

    // Activity buffer processing — called by BufferCompleted for each record.
    // Returns true if the record was consumed (stops further dispatch).
    virtual bool handleActivityRecord(const CUpti_Activity* record,
                                      int64_t baseCpuNs, uint64_t baseCuptiTs) {
        return false;
    }
};

}  // namespace gpufl
