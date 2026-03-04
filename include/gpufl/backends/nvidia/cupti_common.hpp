#pragma once

#include <cuda_runtime.h>
#include <cupti.h>

#include <string>
#include <utility>
#include <vector>

#include "gpufl/core/trace_type.hpp"

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
struct ActivityRecord {
    uint32_t device_id;
    char name[128];
    TraceType type;
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    int64_t cpu_start_ns;
    int64_t api_start_ns;
    int64_t api_exit_ns;
    int64_t duration_ns;

    // Detailed metrics (optional)
    bool has_details;
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
    int dyn_shared;
    int static_shared;
    int local_bytes;
    int const_bytes;
    int num_regs;
    float occupancy;

    // Per-resource occupancy breakdown
    float reg_occupancy = 0.0f;    // occupancy limited only by register file
    float smem_occupancy = 0.0f;   // occupancy limited only by shared memory
    float warp_occupancy = 0.0f;   // occupancy limited only by warp count
    float block_occupancy = 0.0f;  // occupancy limited only by block count
    char limiting_resource[16]{};  // "warps"|"registers"|"shared_mem"|"blocks"

    int max_active_blocks;
    unsigned int corr_id;

    char source_file[256];
    uint32_t source_line;
    char function_name[256];
    uint32_t samples_count;
    uint32_t stall_reason;
    std::string reason_name;
    char device_name[64]{};

    // SASS Metrics support
    uint32_t pc_offset;
    uint64_t metric_value;
    char metric_name[64];

    char user_scope[256]{};
    int scope_depth{};

    size_t stack_id{};

    // Phase 1a: additional CUpti_ActivityKernel11 fields
    uint32_t local_mem_total =
        0;  // localMemoryTotal — total local mem across all threads
    uint8_t cache_config_requested = 0;  // cacheConfigRequested
    uint8_t cache_config_executed = 0;   // cacheConfigExecuted
    uint32_t shared_mem_executed =
        0;  // sharedMemoryExecuted — actual smem allocated by driver

    // Memcpy / Memset specific
    uint64_t bytes;
    uint32_t copy_kind;  // CUpti_ActivityMemcpyKind
    uint32_t src_kind;   // CUpti_ActivityMemoryKind
    uint32_t dst_kind;   // CUpti_ActivityMemoryKind
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
