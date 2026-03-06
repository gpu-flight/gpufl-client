#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"

#include <cstdio>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

namespace gpufl {
extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;
}

namespace gpufl {
namespace {
const char* CallbackSiteName(CUpti_ApiCallbackSite site) {
    switch (site) {
        case CUPTI_API_ENTER:
            return "ENTER";
        case CUPTI_API_EXIT:
            return "EXIT";
        default:
            return "UNKNOWN";
    }
}
}  // namespace

KernelLaunchHandler::KernelLaunchHandler(CuptiBackend* backend)
    : backend_(backend) {}

std::vector<CUpti_CallbackDomain> KernelLaunchHandler::requiredDomains() const {
    return {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_CB_DOMAIN_DRIVER_API};
}

std::vector<CUpti_ActivityKind> KernelLaunchHandler::requiredActivityKinds()
    const {
    return {CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL};
}

bool KernelLaunchHandler::shouldHandle(CUpti_CallbackDomain domain,
                                       CUpti_CallbackId cbid) const {
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        return cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
               cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000;
    }
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        return cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunch ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz;
    }
    return false;
}

void KernelLaunchHandler::handle(CUpti_CallbackDomain domain,
                                 CUpti_CallbackId cbid, const void* cbdata) {
    if (!backend_->IsActive()) return;

    auto* cbInfo = static_cast<const CUpti_CallbackData*>(cbdata);
    if (!cbInfo) {
        GFL_LOG_ERROR("[KernelLaunchHandler] cbInfo is null");
        return;
    }

    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        LaunchMeta meta{};
        meta.api_enter_ns = detail::GetTimestampNs();

        const char* nm =
            cbInfo->symbolName ? cbInfo->symbolName : cbInfo->functionName;
        if (!nm) nm = "kernel_launch";
        std::snprintf(meta.name, sizeof(meta.name), "%s", nm);

        if (backend_->GetOptions().enable_stack_trace) {
            const std::string trace = gpufl::core::GetCallStack(2);
            const std::string cleanTrace = detail::SanitizeStackTrace(trace);
            meta.stack_id =
                gpufl::StackRegistry::instance().getOrRegister(cleanTrace);
        } else {
            meta.stack_id = 0;
        }

        auto& stack = getThreadScopeStack();
        if (!stack.empty()) {
            std::string fullPath;
            for (size_t i = 0; i < stack.size(); ++i) {
                if (i > 0) fullPath += "|";
                fullPath += stack[i];
            }
            fullPath += "|";
            fullPath += meta.name;
            std::snprintf(meta.user_scope, sizeof(meta.user_scope), "%s",
                          fullPath.c_str());
            meta.scope_depth = stack.size();
        } else {
            std::string fullPath = "global|";
            fullPath += meta.name;
            std::snprintf(meta.user_scope, sizeof(meta.user_scope), "%s",
                          fullPath.c_str());
            meta.scope_depth = 0;
        }

        if (backend_->GetOptions().collect_kernel_details &&
            cbInfo->functionParams != nullptr) {
            if ((domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
                 cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) ||
                (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
                 cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)) {
                GFL_LOG_DEBUG("[KernelLaunchHandler] details path domain=",
                              static_cast<int>(domain), " cbid=",
                              static_cast<int>(cbid), " corr=",
                              cbInfo->correlationId, " params=",
                              cbInfo->functionParams);
                meta.has_details = true;
                const auto* params =
                    (cudaLaunchKernel_v7000_params*)(cbInfo->functionParams);
                meta.grid_x = params->gridDim.x;
                meta.grid_y = params->gridDim.y;
                meta.grid_z = params->gridDim.z;
                meta.block_x = params->blockDim.x;
                meta.block_y = params->blockDim.y;
                meta.block_z = params->blockDim.z;
                meta.dyn_shared = static_cast<int>(params->sharedMem);
                CalculateOccupancy(meta, params->func);
            }
        }

        std::lock_guard<std::mutex> lk(backend_->meta_mu_);
        auto& existing = backend_->meta_by_corr_[cbInfo->correlationId];
        if (existing.has_details && !meta.has_details) {
            GFL_LOG_DEBUG(
                "[DEBUG-CALLBACK] Skipping overwrite of rich metadata for "
                "CorrID ",
                cbInfo->correlationId, " by Driver API.");
        } else {
            existing = meta;
        }
    } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        const int64_t t = detail::GetTimestampNs();
        std::lock_guard<std::mutex> lk(backend_->meta_mu_);
        auto it = backend_->meta_by_corr_.find(cbInfo->correlationId);
        if (it != backend_->meta_by_corr_.end()) {
            it->second.api_exit_ns = t;
        }

    }
}

bool KernelLaunchHandler::handleActivityRecord(const CUpti_Activity* record,
                                               int64_t baseCpuNs,
                                               uint64_t baseCuptiTs) {
    if (!record) {
        GFL_LOG_ERROR("[KernelLaunchHandler] null activity record");
        return false;
    }
    GFL_LOG_DEBUG("[KernelLaunchHandler] activity begin kind=",
                  static_cast<int>(record->kind));
    if (record->kind != CUPTI_ACTIVITY_KIND_KERNEL &&
        record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        return false;
    }

    const auto* k = reinterpret_cast<const CUpti_ActivityKernel11*>(record);

    if (backend_->opts_.kernel_sample_rate_ms > 0) {
        uint64_t intervalNs =
            static_cast<uint64_t>(backend_->opts_.kernel_sample_rate_ms) *
            1000000ULL;
        uint64_t lastTs =
            backend_->last_kernel_end_ts_.load(std::memory_order_relaxed);
        if (k->start < lastTs + intervalNs) {
            GFL_LOG_DEBUG("[KernelLaunchHandler] activity throttled corr=",
                          k->correlationId, " start=", k->start, " last=",
                          lastTs, " intervalNs=", intervalNs);
            return true;  // within throttle window — consume but do not emit
        }
        backend_->last_kernel_end_ts_.store(k->start, std::memory_order_relaxed);
    }

    ActivityRecord out{};
    out.device_id = k->deviceId;
    out.stream =
        reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(k->streamId));
    out.type = TraceType::KERNEL;
    std::snprintf(out.name, sizeof(out.name), "%s",
                  (k->name ? k->name : "kernel"));
    out.cpu_start_ns = baseCpuNs + static_cast<int64_t>(k->start - baseCuptiTs);
    out.duration_ns = static_cast<int64_t>(k->end - k->start);
    out.dyn_shared = k->dynamicSharedMemory;
    out.static_shared = k->staticSharedMemory;
    out.num_regs = k->registersPerThread;
    out.has_details = false;

    // Phase 1a: always-on fields from CUpti_ActivityKernel11
    out.local_mem_total = k->localMemoryTotal;
    out.cache_config_requested = k->cacheConfig.config.requested;
    out.cache_config_executed = k->cacheConfig.config.executed;
    out.shared_mem_executed = k->sharedMemoryExecuted;

    {
        const uint64_t corr = k->correlationId;
        out.corr_id = corr;
        GFL_LOG_DEBUG("[KernelLaunchHandler] activity kernel corr=", corr,
                      " device=", k->deviceId, " stream=", k->streamId,
                      " start=", k->start, " end=", k->end);
        std::lock_guard lk(backend_->meta_mu_);
        if (auto it = backend_->meta_by_corr_.find(corr);
            it != backend_->meta_by_corr_.end()) {
            const LaunchMeta& m = it->second;
            out.scope_depth = m.scope_depth;
            out.stack_id = m.stack_id;
            std::copy(std::begin(m.user_scope), std::end(m.user_scope),
                      std::begin(out.user_scope));
            out.api_start_ns = m.api_enter_ns;
            out.api_exit_ns = m.api_exit_ns;
            if (m.has_details) {
                out.has_details = true;
                out.grid_x = m.grid_x;
                out.grid_y = m.grid_y;
                out.grid_z = m.grid_z;
                out.block_x = m.block_x;
                out.block_y = m.block_y;
                out.block_z = m.block_z;
                out.local_bytes = static_cast<int>(k->localMemoryPerThread);
                out.const_bytes = m.const_bytes;
                out.occupancy = m.occupancy;
                out.max_active_blocks = m.max_active_blocks;

                // Compute per-resource occupancy breakdown
                SmProps props = GetSMProps(out.device_id);
                int threadsPerBlock = out.block_x * out.block_y * out.block_z;
                int warpsPerBlock =
                    (threadsPerBlock + props.warpSize - 1) / props.warpSize;
                int maxWarpsPerSM = props.maxThreadsPerSM / props.warpSize;

                // Warp limit
                int warpBlocks =
                    (warpsPerBlock > 0) ? (maxWarpsPerSM / warpsPerBlock) : 0;

                // Hardware block count limit
                int blockBlocks = props.maxBlocksPerSM;

                // Register limit — registers are allocated per-warp in
                // multiples of 256 on modern NVIDIA architectures (sm_6x+).
                constexpr int kRegAllocGranularity = 256;
                int regsPerWarp = (warpsPerBlock > 0 && out.num_regs > 0)
                                      ? (((out.num_regs * props.warpSize) +
                                          kRegAllocGranularity - 1) /
                                         kRegAllocGranularity) *
                                            kRegAllocGranularity
                                      : 0;
                int regsPerBlock = regsPerWarp * warpsPerBlock;
                int regBlocks = (regsPerBlock > 0)
                                    ? (props.regsPerSM / regsPerBlock)
                                    : warpBlocks;

                // Shared memory limit
                int smemPerBlock = out.static_shared + out.dyn_shared;
                int smemBlocks_approx =
                    (smemPerBlock > 0) ? (props.sharedMemPerSM / smemPerBlock)
                                       : warpBlocks;
                int nonSmemLimit =
                    std::min({warpBlocks, regBlocks, blockBlocks});
                int smemBlocks;
                if (out.max_active_blocks > 0 && smemPerBlock > 0 &&
                    out.max_active_blocks < nonSmemLimit) {
                    smemBlocks = out.max_active_blocks;
                } else {
                    smemBlocks = smemBlocks_approx;
                }

                auto toOcc = [&](int blocks) -> float {
                    return (maxWarpsPerSM > 0 && warpsPerBlock > 0)
                               ? std::min(1.0f, static_cast<float>(
                                                    blocks * warpsPerBlock) /
                                                    maxWarpsPerSM)
                               : 0.0f;
                };
                out.warp_occupancy = toOcc(warpBlocks);
                out.reg_occupancy = toOcc(regBlocks);
                out.smem_occupancy = toOcc(smemBlocks);
                out.block_occupancy = toOcc(blockBlocks);

                struct {
                    float occ;
                    const char* name;
                } limiters[] = {
                    {out.warp_occupancy, "warps"},
                    {out.reg_occupancy, "registers"},
                    {out.smem_occupancy, "shared_mem"},
                    {out.block_occupancy, "blocks"},
                };
                const char* limiting = "warps";
                float minOcc = out.warp_occupancy;
                for (auto& l : limiters) {
                    if (l.occ < minOcc) {
                        minOcc = l.occ;
                        limiting = l.name;
                    }
                }
                std::snprintf(out.limiting_resource,
                              sizeof(out.limiting_resource), "%s", limiting);
            }
            backend_->meta_by_corr_.erase(it);
        }
    }

    g_monitorBuffer.Push(out);
    GFL_LOG_DEBUG("[KernelLaunchHandler] activity pushed corr=", out.corr_id,
                  " duration_ns=", out.duration_ns, " has_details=",
                  out.has_details);
    return true;
}

}  // namespace gpufl
