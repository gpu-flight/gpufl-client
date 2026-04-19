#include "gpufl/backends/nvidia/kernel_launch_handler.hpp"

#include <cstdio>

#include "gpufl/backends/nvidia/cupti_utils.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/scope_registry.hpp"
#include "gpufl/core/stack_registry.hpp"
#include "gpufl/core/stack_trace.hpp"

using gpufl::core::DemangleName;

namespace gpufl {

KernelLaunchHandler::KernelLaunchHandler(CuptiBackend* backend)
    : backend_(backend) {}

const std::string& KernelLaunchHandler::cachedDemangle(const char* mangled) {
    if (!mangled) {
        static const std::string fallback = "kernel_launch";
        return fallback;
    }
    std::lock_guard lk(demangle_mu_);
    auto it = demangle_cache_.find(mangled);
    if (it != demangle_cache_.end()) return it->second;
    auto [inserted, _] = demangle_cache_.emplace(mangled, DemangleName(mangled));
    return inserted->second;
}

std::vector<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>>
KernelLaunchHandler::requiredCallbacks() const {
    return {
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020},
        {CUPTI_CB_DOMAIN_RUNTIME_API,
         CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunch},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid},
        {CUPTI_CB_DOMAIN_DRIVER_API,
         CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync},
        {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel},
        {CUPTI_CB_DOMAIN_DRIVER_API,
         CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz},
    };
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
        const std::string& demangledName = cachedDemangle(nm);
        std::snprintf(meta.name, sizeof(meta.name), "%s", demangledName.c_str());

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
            if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
                cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
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
            } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
                       cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
                meta.has_details = true;
                const auto* params =
                    (cuLaunchKernel_params*)cbInfo->functionParams;
                meta.grid_x = params->gridDimX;
                meta.grid_y = params->gridDimY;
                meta.grid_z = params->gridDimZ;
                meta.block_x = params->blockDimX;
                meta.block_y = params->blockDimY;
                meta.block_z = params->blockDimZ;
                meta.dyn_shared = static_cast<int>(params->sharedMemBytes);
            }
        }

        // Store metadata — emit later from scope stop (PC Sampling path)
        // or handleActivityRecord (normal path).
        {
            std::lock_guard lk(backend_->meta_mu_);
            auto& existing = backend_->meta_by_corr_[cbInfo->correlationId];
            if (existing.has_details && !meta.has_details) {
                GFL_LOG_DEBUG(
                    "[DEBUG-CALLBACK] Skipping overwrite of rich metadata "
                    "for CorrID ",
                    cbInfo->correlationId, " by Driver API.");
            } else {
                existing = meta;
            }
        }
    } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
        const int64_t exitNs = detail::GetTimestampNs();
        std::lock_guard<std::mutex> lk(backend_->meta_mu_);
        auto it = backend_->meta_by_corr_.find(cbInfo->correlationId);
        if (it != backend_->meta_by_corr_.end()) {
            it->second.api_exit_ns = exitNs;
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
    if (record->kind != CUPTI_ACTIVITY_KIND_KERNEL &&
        record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        return false;
    }

    const auto* k = reinterpret_cast<const CUpti_ActivityKernel11*>(record);
    backend_->kernel_activity_seen_.fetch_add(1, std::memory_order_relaxed);

    const bool shouldThrottleKernels =
        backend_->opts_.kernel_sample_rate_ms > 0 &&
        backend_->opts_.profiling_engine != ProfilingEngine::PcSampling;

    if (shouldThrottleKernels) {
        uint64_t intervalNs =
            static_cast<uint64_t>(backend_->opts_.kernel_sample_rate_ms) *
            1000000ULL;
        uint64_t lastTs =
            backend_->last_kernel_end_ts_.load(std::memory_order_relaxed);
        if (k->start < lastTs + intervalNs) {
            GFL_LOG_DEBUG("[KernelLaunchHandler] activity throttled corr=",
                          k->correlationId, " start=", k->start,
                          " last=", lastTs, " intervalNs=", intervalNs);
            backend_->kernel_activity_throttled_.fetch_add(
                1, std::memory_order_relaxed);
            return true;  // within throttle window — consume but do not emit
        }
        backend_->last_kernel_end_ts_.store(k->start,
                                            std::memory_order_relaxed);
    }

    ActivityRecord out{};
    out.device_id = k->deviceId;
    out.stream = static_cast<StreamHandle>(k->streamId);
    out.type = TraceType::KERNEL;
    const std::string& demangledKernelName = cachedDemangle(k->name);
    std::snprintf(out.name, sizeof(out.name), "%s", demangledKernelName.c_str());
    out.cpu_start_ns = baseCpuNs + static_cast<int64_t>(k->start - baseCuptiTs);
    out.duration_ns = static_cast<int64_t>(k->end - k->start);
    out.dyn_shared = k->dynamicSharedMemory;
    out.static_shared = k->staticSharedMemory;
    out.num_regs = k->registersPerThread;
    out.has_details = false;

    // Phase 1a: always-on fields from CUpti_ActivityKernel11
    out.local_mem_total = k->localMemoryTotal;
    out.local_mem_per_thread =
        k->localMemoryPerThread;  // 0 = no register spill
    out.cache_config_requested = k->cacheConfig.config.requested;
    out.cache_config_executed = k->cacheConfig.config.executed;
    out.shared_mem_executed = k->sharedMemoryExecuted;

    {
        const uint64_t corr = k->correlationId;
        out.corr_id = corr;
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

                // Compute per-resource occupancy from activity record data
                // (registers, shared memory) and SM properties.
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
                int smemBlocks =
                    (smemPerBlock > 0) ? (props.sharedMemPerSM / smemPerBlock)
                                       : warpBlocks;

                out.max_active_blocks =
                    std::min({warpBlocks, regBlocks, blockBlocks, smemBlocks});

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
                out.occupancy = toOcc(out.max_active_blocks);

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
    backend_->kernel_activity_emitted_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

}  // namespace gpufl
