#if !(GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK)
#error \
    "rocprofiler_backend.cpp should only be compiled when GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK are true."
#endif

#include "gpufl/backends/amd/rocprofiler_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/trace_type.hpp"

namespace gpufl {
extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;
}

namespace gpufl::amd {
namespace {

constexpr size_t kTraceBufferBytes = 1u << 20;

std::atomic<RocprofilerBackend*> g_pending_backend{nullptr};

thread_local std::vector<uint64_t> g_scope_external_stack;
thread_local std::vector<std::string> g_scope_name_stack;

std::string StatusToString(const rocprofiler_status_t status) {
    const char* name = rocprofiler_get_status_name(status);
    const char* text = rocprofiler_get_status_string(status);
    std::string out = name ? std::string(name) : std::string("ROCPROFILER_STATUS_ERROR");
    if (text && *text) out += ": " + std::string(text);
    return out;
}

bool CheckStatus(const rocprofiler_status_t status, const char* what,
                 std::string* reason = nullptr) {
    if (status == ROCPROFILER_STATUS_SUCCESS) return true;
    if (reason) {
        *reason = std::string(what) + " failed: " + StatusToString(status);
    }
    GFL_LOG_ERROR("[ROCProfilerBackend] ", what, " failed: ", StatusToString(status));
    return false;
}

uint32_t TruncateCorrelationId(const uint64_t value) {
    return static_cast<uint32_t>(value & 0xffffffffu);
}

const char* CopyKindName(const uint32_t kind) {
    switch (kind) {
        case 1:
            return "HtoD";
        case 2:
            return "DtoH";
        case 3:
            return "DtoD";
        case 4:
            return "HtoH";
        default:
            return "Unknown";
    }
}

}  // namespace

bool RocprofilerBackend::IsAvailable(std::string* reason) {
    uint32_t major = 0;
    uint32_t minor = 0;
    uint32_t patch = 0;
    const auto status = rocprofiler_get_version(&major, &minor, &patch);
    if (!CheckStatus(status, "rocprofiler_get_version", reason)) return false;
    if (reason) {
        *reason = "ROCprofiler-SDK " + std::to_string(major) + "." +
                  std::to_string(minor) + "." + std::to_string(patch) + " available";
    }
    return true;
}

void RocprofilerBackend::initialize(const MonitorOptions& opts) {
    if (initialized_.exchange(true)) return;

    opts_ = opts;
    std::string reason;
    if (!configureRocprofiler(opts, &reason)) {
        initialized_.store(false);
        GFL_LOG_ERROR("[ROCProfilerBackend] initialization failed: ", reason);
        return;
    }
}

bool RocprofilerBackend::configureRocprofiler(const MonitorOptions& opts,
                                              std::string* reason) {
    (void) opts;
    if (!IsAvailable(reason)) return false;
    if (!registerTool(reason)) return false;
    return true;
}

void RocprofilerBackend::resetToolState() {
    context_ = {};
    buffer_ = {};
    client_handle_ = 0;
    client_finalize_ = nullptr;
    tool_registered_.store(false);
    active_.store(false);
    {
        std::lock_guard<std::mutex> lock(kernel_meta_mutex_);
        kernel_metadata_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(external_scope_mutex_);
        external_scope_metadata_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(agent_mutex_);
        gpu_device_ids_.clear();
        agent_types_.clear();
    }
}

bool RocprofilerBackend::registerTool(std::string* reason) {
    g_pending_backend.store(this, std::memory_order_release);

    int init_status = 0;
    (void) rocprofiler_is_initialized(&init_status);

    const auto status = rocprofiler_force_configure(&RocprofilerBackend::configure);
    if (status != ROCPROFILER_STATUS_SUCCESS &&
        status != ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED) {
        g_pending_backend.store(nullptr, std::memory_order_release);
        return CheckStatus(status, "rocprofiler_force_configure", reason);
    }

    if (!tool_registered_.load(std::memory_order_acquire)) {
        if (reason) {
            if (status == ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED &&
                init_status != 0) {
                *reason =
                    "rocprofiler configuration is already locked by another tool or runtime";
            } else {
                *reason = "rocprofiler tool registration did not complete";
            }
        }
        return false;
    }

    if (reason) reason->clear();
    return true;
}

rocprofiler_tool_configure_result_t* RocprofilerBackend::configure(
    uint32_t,
    const char*,
    uint32_t priority,
    rocprofiler_client_id_t* client_id) {
    auto* backend = g_pending_backend.load(std::memory_order_acquire);
    if (!backend || priority > 0 || client_id == nullptr) return nullptr;

    client_id->name = "gpufl-amd";
    backend->client_handle_ = client_id->handle;

    static rocprofiler_tool_configure_result_t result = {
        sizeof(rocprofiler_tool_configure_result_t),
        &RocprofilerBackend::toolInitializeShim,
        &RocprofilerBackend::toolFinalizeShim,
        nullptr,
    };
    result.tool_data = backend;
    return &result;
}

int RocprofilerBackend::toolInitializeShim(
    const rocprofiler_client_finalize_t finalize_func,
    void* tool_data) {
    auto* backend = static_cast<RocprofilerBackend*>(tool_data);
    if (!backend) return -1;
    backend->client_finalize_ = finalize_func;
    return backend->toolInitialize();
}

void RocprofilerBackend::toolFinalizeShim(void* tool_data) {
    auto* backend = static_cast<RocprofilerBackend*>(tool_data);
    if (backend) backend->toolFinalize();
}

int RocprofilerBackend::toolInitialize() {
    std::string reason;

    if (!CheckStatus(rocprofiler_create_context(&context_), "rocprofiler_create_context",
                     &reason)) {
        return -1;
    }

    if (!CheckStatus(rocprofiler_query_available_agents(
                         ROCPROFILER_AGENT_INFO_VERSION_0,
                         &RocprofilerBackend::queryAgentsShim,
                         sizeof(rocprofiler_agent_t),
                         this),
                     "rocprofiler_query_available_agents",
                     &reason)) {
        return -1;
    }

    if (!CheckStatus(rocprofiler_create_buffer(context_, kTraceBufferBytes,
                                               kTraceBufferBytes / 2,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               &RocprofilerBackend::bufferTracingShim,
                                               this, &buffer_),
                     "rocprofiler_create_buffer", &reason)) {
        return -1;
    }

    const std::array<rocprofiler_tracing_operation_t, 1> code_object_ops = {
        static_cast<rocprofiler_tracing_operation_t>(
            ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER),
    };
    if (!CheckStatus(rocprofiler_configure_callback_tracing_service(
                         context_, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                         code_object_ops.data(), code_object_ops.size(),
                         &RocprofilerBackend::callbackTracingShim, this),
                     "rocprofiler_configure_callback_tracing_service(code_object)",
                     &reason)) {
        return -1;
    }

    if (!CheckStatus(rocprofiler_configure_buffer_tracing_service(
                         context_, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                         nullptr, 0, buffer_),
                     "rocprofiler_configure_buffer_tracing_service(kernel_dispatch)",
                     &reason)) {
        return -1;
    }

    if (!CheckStatus(rocprofiler_configure_buffer_tracing_service(
                         context_, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                         nullptr, 0, buffer_),
                     "rocprofiler_configure_buffer_tracing_service(memory_copy)",
                     &reason)) {
        return -1;
    }

    tool_registered_.store(true, std::memory_order_release);
    return 0;
}

void RocprofilerBackend::toolFinalize() {
    if (buffer_.handle != 0) {
        (void) rocprofiler_flush_buffer(buffer_);
        (void) rocprofiler_destroy_buffer(buffer_);
    }

    resetToolState();
    g_pending_backend.store(nullptr, std::memory_order_release);
}

void RocprofilerBackend::start() {
    if (!initialized_.load() || context_.handle == 0 || active_.load()) return;
    if (CheckStatus(rocprofiler_start_context(context_), "rocprofiler_start_context")) {
        active_.store(true);
    }
}

void RocprofilerBackend::stop() {
    if (!active_.exchange(false) || context_.handle == 0) return;
    (void) rocprofiler_stop_context(context_);
    flushBuffers();
}

void RocprofilerBackend::shutdown() {
    if (!initialized_.exchange(false)) return;

    stop();

    if (client_finalize_) {
        rocprofiler_client_id_t client_id = {
            sizeof(rocprofiler_client_id_t), "gpufl-amd", client_handle_};
        client_finalize_(client_id);
        client_finalize_ = nullptr;
    } else {
        toolFinalize();
    }
}

void RocprofilerBackend::flushBuffers() {
    if (buffer_.handle != 0) (void) rocprofiler_flush_buffer(buffer_);
}

void RocprofilerBackend::OnScopeStart(const char* name) {
    if (!active_.load() || context_.handle == 0 || name == nullptr) return;

    rocprofiler_thread_id_t tid{};
    if (rocprofiler_get_thread_id(&tid) != ROCPROFILER_STATUS_SUCCESS) return;

    static std::atomic<uint64_t> next_scope_external{1};
    const uint64_t external_value =
        next_scope_external.fetch_add(1, std::memory_order_relaxed);

    g_scope_name_stack.emplace_back(name);
    std::string scope_path;
    for (size_t i = 0; i < g_scope_name_stack.size(); ++i) {
        if (i > 0) scope_path += "|";
        scope_path += g_scope_name_stack[i];
    }

    {
        std::lock_guard<std::mutex> lock(external_scope_mutex_);
        external_scope_metadata_[external_value] = ExternalScopeMetadata{
            scope_path, static_cast<int>(g_scope_name_stack.size())};
    }

    g_scope_external_stack.push_back(external_value);
    rocprofiler_user_data_t user_data{};
    user_data.value = external_value;
    (void) rocprofiler_push_external_correlation_id(context_, tid, user_data);
}

void RocprofilerBackend::OnScopeStop(const char*) {
    if (!active_.load() || context_.handle == 0) return;

    rocprofiler_thread_id_t tid{};
    if (rocprofiler_get_thread_id(&tid) != ROCPROFILER_STATUS_SUCCESS) return;

    rocprofiler_user_data_t user_data{};
    (void) rocprofiler_pop_external_correlation_id(context_, tid, &user_data);
    if (!g_scope_external_stack.empty()) g_scope_external_stack.pop_back();
    if (!g_scope_name_stack.empty()) g_scope_name_stack.pop_back();
}

void RocprofilerBackend::callbackTracingShim(rocprofiler_callback_tracing_record_t record,
                                             rocprofiler_user_data_t*,
                                             void* callback_data) {
    auto* backend = static_cast<RocprofilerBackend*>(callback_data);
    if (!backend || record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
        record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD ||
        record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
        record.payload == nullptr) {
        return;
    }

    const auto* data =
        static_cast<const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
            record.payload);

    KernelMetadata meta{};
    meta.name = data->kernel_name ? data->kernel_name : "";
    meta.group_segment_size = data->group_segment_size;
    meta.private_segment_size = data->private_segment_size;
    meta.sgpr_count = data->sgpr_count;
    meta.arch_vgpr_count = data->arch_vgpr_count;
    meta.accum_vgpr_count = data->accum_vgpr_count;

    std::lock_guard<std::mutex> lock(backend->kernel_meta_mutex_);
    backend->kernel_metadata_[data->kernel_id] = std::move(meta);
}

void RocprofilerBackend::bufferTracingShim(rocprofiler_context_id_t,
                                           rocprofiler_buffer_id_t,
                                           rocprofiler_record_header_t** headers,
                                           const size_t num_headers,
                                           void* data,
                                           uint64_t) {
    auto* backend = static_cast<RocprofilerBackend*>(data);
    if (!backend || headers == nullptr) return;

    for (size_t i = 0; i < num_headers; ++i) {
        auto* header = headers[i];
        if (!header || header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
            header->payload == nullptr) {
            continue;
        }

        switch (header->kind) {
            case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH: {
                const auto* record =
                    static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                        header->payload);
                backend->handleKernelDispatch(record->dispatch_info,
                                              record->start_timestamp,
                                              record->end_timestamp,
                                              record->correlation_id);
                break;
            }
            case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY: {
                const auto* record =
                    static_cast<const rocprofiler_buffer_tracing_memory_copy_record_t*>(
                        header->payload);
                backend->handleMemoryCopy(*record);
                break;
            }
            default:
                break;
        }
    }
}

std::string RocprofilerBackend::resolveKernelName(const uint64_t kernel_id) const {
    std::lock_guard<std::mutex> lock(kernel_meta_mutex_);
    if (auto itr = kernel_metadata_.find(kernel_id); itr != kernel_metadata_.end() &&
        !itr->second.name.empty()) {
        return itr->second.name;
    }

    char buffer[64] = {};
    std::snprintf(buffer, sizeof(buffer), "kernel_%llu",
                  static_cast<unsigned long long>(kernel_id));
    return buffer;
}

rocprofiler_status_t RocprofilerBackend::queryAgentsShim(
    const rocprofiler_agent_version_t version,
    const void** agents,
    const size_t num_agents,
    void* user_data) {
    if (version != ROCPROFILER_AGENT_INFO_VERSION_0 || user_data == nullptr) {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    auto* backend = static_cast<RocprofilerBackend*>(user_data);
    std::lock_guard<std::mutex> lock(backend->agent_mutex_);
    backend->gpu_device_ids_.clear();
    backend->agent_types_.clear();

    for (size_t i = 0; i < num_agents; ++i) {
        const auto* agent = static_cast<const rocprofiler_agent_t*>(agents[i]);
        if (!agent) continue;

        backend->agent_types_[agent->id.handle] = agent->type;
        if (agent->type == ROCPROFILER_AGENT_TYPE_GPU) {
            backend->gpu_device_ids_[agent->id.handle] =
                std::max(agent->logical_node_type_id, 0);
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

int RocprofilerBackend::resolveDeviceId(const rocprofiler_agent_id_t agent_id) const {
    std::lock_guard<std::mutex> lock(agent_mutex_);
    if (auto itr = gpu_device_ids_.find(agent_id.handle); itr != gpu_device_ids_.end()) {
        return itr->second;
    }
    return 0;
}

uint32_t RocprofilerBackend::classifyMemcpyKind(const rocprofiler_agent_id_t src_agent,
                                                const rocprofiler_agent_id_t dst_agent) const {
    std::lock_guard<std::mutex> lock(agent_mutex_);
    const auto src_type = agent_types_.count(src_agent.handle) > 0
                              ? agent_types_.at(src_agent.handle)
                              : ROCPROFILER_AGENT_TYPE_NONE;
    const auto dst_type = agent_types_.count(dst_agent.handle) > 0
                              ? agent_types_.at(dst_agent.handle)
                              : ROCPROFILER_AGENT_TYPE_NONE;

    if (src_type == ROCPROFILER_AGENT_TYPE_CPU &&
        dst_type == ROCPROFILER_AGENT_TYPE_GPU) {
        return 1;
    }
    if (src_type == ROCPROFILER_AGENT_TYPE_GPU &&
        dst_type == ROCPROFILER_AGENT_TYPE_CPU) {
        return 2;
    }
    if (src_type == ROCPROFILER_AGENT_TYPE_GPU &&
        dst_type == ROCPROFILER_AGENT_TYPE_GPU) {
        return 3;
    }
    if (src_type == ROCPROFILER_AGENT_TYPE_CPU &&
        dst_type == ROCPROFILER_AGENT_TYPE_CPU) {
        return 4;
    }
    return 0;
}

void RocprofilerBackend::handleKernelDispatch(
    const rocprofiler_kernel_dispatch_info_t& info,
    const uint64_t start_timestamp,
    const uint64_t end_timestamp,
    const rocprofiler_async_correlation_id_t& correlation_id) {
    ActivityRecord out{};
    out.type = TraceType::KERNEL;
    out.device_id = static_cast<uint32_t>(resolveDeviceId(info.agent_id));
    out.stream = static_cast<StreamHandle>(info.queue_id.handle);
    out.cpu_start_ns = static_cast<int64_t>(start_timestamp);
    out.duration_ns =
        static_cast<int64_t>(end_timestamp >= start_timestamp ? end_timestamp - start_timestamp
                                                              : 0);
    out.api_start_ns = out.cpu_start_ns;
    out.api_exit_ns = out.cpu_start_ns + out.duration_ns;
    out.corr_id = TruncateCorrelationId(correlation_id.internal);
    out.has_details = true;
    out.grid_x = static_cast<int>(info.grid_size.x);
    out.grid_y = static_cast<int>(info.grid_size.y);
    out.grid_z = static_cast<int>(info.grid_size.z);
    out.block_x = static_cast<int>(info.workgroup_size.x);
    out.block_y = static_cast<int>(info.workgroup_size.y);
    out.block_z = static_cast<int>(info.workgroup_size.z);
    out.static_shared = static_cast<int>(info.group_segment_size);
    out.shared_mem_executed = info.group_segment_size;
    out.local_bytes = static_cast<int>(info.private_segment_size);
    out.local_mem_per_thread = info.private_segment_size;
    out.local_mem_total = info.private_segment_size *
                          std::max<uint32_t>(1, info.workgroup_size.x) *
                          std::max<uint32_t>(1, info.workgroup_size.y) *
                          std::max<uint32_t>(1, info.workgroup_size.z);

    const auto name = resolveKernelName(info.kernel_id);
    std::snprintf(out.name, sizeof(out.name), "%s", name.c_str());

    {
        std::lock_guard<std::mutex> lock(kernel_meta_mutex_);
        if (auto itr = kernel_metadata_.find(info.kernel_id);
            itr != kernel_metadata_.end()) {
            const auto& meta = itr->second;
            out.num_regs = static_cast<int>(
                meta.sgpr_count + meta.arch_vgpr_count + meta.accum_vgpr_count);
            if (out.static_shared == 0) out.static_shared = static_cast<int>(meta.group_segment_size);
            if (out.local_bytes == 0) out.local_bytes = static_cast<int>(meta.private_segment_size);
        }
    }

    if (correlation_id.external.value != 0) {
        std::lock_guard<std::mutex> lock(external_scope_mutex_);
        if (auto itr = external_scope_metadata_.find(correlation_id.external.value);
            itr != external_scope_metadata_.end() && !itr->second.user_scope.empty()) {
            std::snprintf(out.user_scope, sizeof(out.user_scope), "%s",
                          itr->second.user_scope.c_str());
            out.scope_depth = itr->second.scope_depth;
        }
    }

    g_monitorBuffer.Push(out);
}

void RocprofilerBackend::handleMemoryCopy(
    const rocprofiler_buffer_tracing_memory_copy_record_t& data) {
    ActivityRecord out{};
    out.type = TraceType::MEMCPY;
    out.device_id = static_cast<uint32_t>(resolveDeviceId(data.dst_agent_id));
    out.cpu_start_ns = static_cast<int64_t>(data.start_timestamp);
    out.duration_ns = static_cast<int64_t>(
        data.end_timestamp >= data.start_timestamp ? data.end_timestamp - data.start_timestamp
                                                   : 0);
    out.api_start_ns = out.cpu_start_ns;
    out.api_exit_ns = out.cpu_start_ns + out.duration_ns;
    out.bytes = data.bytes;
    out.copy_kind = classifyMemcpyKind(data.src_agent_id, data.dst_agent_id);
    out.corr_id = TruncateCorrelationId(data.correlation_id.internal);
    std::snprintf(out.name, sizeof(out.name), "%s", CopyKindName(out.copy_kind));

    if (data.correlation_id.external.value != 0) {
        std::lock_guard<std::mutex> lock(external_scope_mutex_);
        if (auto itr = external_scope_metadata_.find(data.correlation_id.external.value);
            itr != external_scope_metadata_.end() && !itr->second.user_scope.empty()) {
            std::snprintf(out.user_scope, sizeof(out.user_scope), "%s",
                          itr->second.user_scope.c_str());
            out.scope_depth = itr->second.scope_depth;
        }
    }

    g_monitorBuffer.Push(out);
}

}  // namespace gpufl::amd
