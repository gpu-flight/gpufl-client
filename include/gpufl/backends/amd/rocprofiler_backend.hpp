#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include "gpufl/backends/amd/engine/amd_profiling_engine.hpp"
#include "gpufl/core/activity_record.hpp"
#include "gpufl/core/monitor_backend.hpp"

namespace gpufl::amd {

class RocprofilerBackend final : public IMonitorBackend {
   public:
    RocprofilerBackend() = default;
    ~RocprofilerBackend() override = default;

    void initialize(const MonitorOptions& opts) override;
    void shutdown() override;
    void start() override;
    void stop() override;

    bool IsMonitoringMode() override { return initialized_.load(); }
    bool IsProfilingMode() override { return engine_ != nullptr; }

    void OnScopeStart(const char* name) override;
    void OnScopeStop(const char* name) override;
    void DrainProfilingData() override;
    void OnPerfScopeStart(const char* name) override;
    void OnPerfScopeStop(const char* name) override;

    void flushBuffers();

    // Expose context and agent for engine initialization
    rocprofiler_context_id_t context() const { return context_; }
    rocprofiler_agent_id_t primaryGpuAgent() const { return primary_gpu_agent_; }

    static bool IsAvailable(std::string* reason = nullptr);

   private:
    struct KernelMetadata {
        std::string name;
        uint32_t group_segment_size = 0;
        uint32_t private_segment_size = 0;
        uint32_t sgpr_count = 0;
        uint32_t arch_vgpr_count = 0;
        uint32_t accum_vgpr_count = 0;
    };

    struct ExternalScopeMetadata {
        std::string user_scope;
        int scope_depth = 0;
    };

    bool configureRocprofiler(const MonitorOptions& opts, std::string* reason);
    void resetToolState();
    bool registerTool(std::string* reason);

    int toolInitialize();
    void toolFinalize();
    void handleKernelDispatch(const rocprofiler_kernel_dispatch_info_t& info,
                              uint64_t start_timestamp,
                              uint64_t end_timestamp,
                              const rocprofiler_async_correlation_id_t& correlation_id);
    void handleMemoryCopy(const rocprofiler_buffer_tracing_memory_copy_record_t& data);
    void handleCodeObjectLoad(const rocprofiler_callback_tracing_code_object_load_data_t& data);

    std::string resolveKernelName(uint64_t kernel_id) const;
    int resolveDeviceId(rocprofiler_agent_id_t agent_id) const;
    uint32_t classifyMemcpyKind(rocprofiler_agent_id_t src_agent,
                                rocprofiler_agent_id_t dst_agent) const;

    static rocprofiler_tool_configure_result_t* configure(uint32_t version,
                                                          const char* runtime_version,
                                                          uint32_t priority,
                                                          rocprofiler_client_id_t* client_id);
    static int toolInitializeShim(rocprofiler_client_finalize_t finalize_func,
                                  void* tool_data);
    static void toolFinalizeShim(void* tool_data);
    static void callbackTracingShim(rocprofiler_callback_tracing_record_t record,
                                    rocprofiler_user_data_t* user_data,
                                    void* callback_data);
    static void bufferTracingShim(rocprofiler_context_id_t context,
                                  rocprofiler_buffer_id_t buffer_id,
                                  rocprofiler_record_header_t** headers,
                                  size_t num_headers,
                                  void* data,
                                  uint64_t drop_count);
    static rocprofiler_status_t queryAgentsShim(rocprofiler_agent_version_t version,
                                                const void** agents,
                                                size_t num_agents,
                                                void* user_data);

    mutable std::mutex kernel_meta_mutex_;
    std::unordered_map<uint64_t, KernelMetadata> kernel_metadata_;
    mutable std::unordered_map<std::string, std::string> demangle_cache_;
    mutable std::mutex external_scope_mutex_;
    std::unordered_map<uint64_t, ExternalScopeMetadata> external_scope_metadata_;

    mutable std::mutex agent_mutex_;
    std::unordered_map<uint64_t, int> gpu_device_ids_;
    std::unordered_map<uint64_t, rocprofiler_agent_type_t> agent_types_;

    // GPU architecture properties for occupancy calculation
    struct GpuArchProps {
        uint32_t wave_front_size = 64;
        uint32_t max_waves_per_cu = 0;
        uint32_t simd_per_cu = 0;
        uint32_t lds_size_bytes = 0;  // per CU
        uint32_t cu_count = 0;
        uint32_t workgroup_max_size = 0;
    };
    std::unordered_map<int, GpuArchProps> gpu_arch_props_;  // device_id -> props

    // Code object storage for ISA disassembly
    mutable std::mutex code_object_mutex_;
    std::unordered_set<uint64_t> enqueued_disasm_crcs_;

    MonitorOptions opts_{};
    rocprofiler_context_id_t context_{};
    rocprofiler_buffer_id_t buffer_{};
    rocprofiler_agent_id_t primary_gpu_agent_{};
    uint32_t client_handle_{0};
    rocprofiler_client_finalize_t client_finalize_{nullptr};

    std::unique_ptr<AmdProfilingEngine> engine_;

    std::atomic<bool> initialized_{false};
    std::atomic<bool> active_{false};
    std::atomic<bool> tool_registered_{false};
};

}  // namespace gpufl::amd
