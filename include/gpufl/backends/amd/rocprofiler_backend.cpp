#if !(GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK)
#error \
    "rocprofiler_backend.cpp should only be compiled when GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK are true."
#endif

#include "gpufl/backends/amd/rocprofiler_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <optional>
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

#include <unistd.h>
#include <zlib.h>

#include "gpufl/backends/amd/engine/dispatch_counter_engine.hpp"
#include "gpufl/backends/amd/engine/device_counter_engine.hpp"
#include "gpufl/backends/amd/amd_capture_capabilities.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/deep_window.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/stack_trace.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/trace_type.hpp"

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

AmdResolvedProfilingPlan RocprofilerBackend::resolvedPlan() const {
    std::lock_guard lock(profiling_plan_mutex_);
    return resolved_plan_;
}

void RocprofilerBackend::setResolvedPlan(AmdResolvedProfilingPlan plan) {
    std::lock_guard lock(profiling_plan_mutex_);
    resolved_plan_ = std::move(plan);
}

std::string RocprofilerBackend::SelectedEngineWireName() const {
    return AmdSelectedPathWireName(resolvedPlan().selected_path);
}

void RocprofilerBackend::initialize(const MonitorOptions& opts) {
    if (initialized_.exchange(true)) return;

    opts_ = opts;
    kernel_rows_emitted_.store(0, std::memory_order_relaxed);
    memcpy_rows_emitted_.store(0, std::memory_order_relaxed);
    trace_records_dropped_.store(0, std::memory_order_relaxed);
    trace_records_queue_dropped_.store(0, std::memory_order_relaxed);
    trace_buffer_flush_failures_.store(0, std::memory_order_relaxed);
    trace_records_unattributed_.store(0, std::memory_order_relaxed);
    scope_correlation_failures_.store(0, std::memory_order_relaxed);
    next_scope_external_.store(1, std::memory_order_relaxed);
    {
        std::lock_guard lock(capture_capabilities_mutex_);
        capture_capabilities_session_id_.clear();
        capability_kernel_rows_baseline_ = 0;
        capability_memcpy_rows_baseline_ = 0;
        capability_pm_sample_rows_baseline_ = 0;
        capability_dropped_records_baseline_ = 0;
        capability_queue_dropped_records_baseline_ = 0;
        capability_buffer_flush_failures_baseline_ = 0;
        capability_unattributed_records_baseline_ = 0;
        capability_scope_correlation_failures_baseline_ = 0;
    }

    AmdProfilingSupport support;
    setResolvedPlan(ResolveAmdProfilingPlan(opts.profiling_engine, support));
    std::string reason;
    if (!configureRocprofiler(opts, &reason)) {
        support.buffer_tracing = false;
        setResolvedPlan(ResolveAmdProfilingPlan(opts.profiling_engine, support));
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
    start_requested_.store(false);
    next_start_retry_ns_.store(0);
    deferred_start_logged_.store(false);
    start_failure_logged_.store(false);
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
        gpu_arch_props_.clear();
        primary_gpu_agent_ = {};
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

    const std::array<rocprofiler_tracing_operation_t, 2> code_object_ops = {
        static_cast<rocprofiler_tracing_operation_t>(
            ROCPROFILER_CODE_OBJECT_LOAD),
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

    // Resolve the user-facing request to an AMD-native path before creating an
    // engine. Unsupported requests remain trace-capable and are reported as
    // explicit fallbacks rather than silently pretending the requested engine ran.
    std::optional<uint32_t> primary_device_id;
    {
        std::lock_guard<std::mutex> lock(agent_mutex_);
        if (const auto it = gpu_device_ids_.find(primary_gpu_agent_.handle);
            it != gpu_device_ids_.end()) {
            primary_device_id = static_cast<uint32_t>(it->second);
        }
    }

    AmdProfilingSupport support;
    support.dispatch_counting = primary_gpu_agent_.handle != 0 &&
                                primary_device_id.has_value();
    support.device_counting = primary_gpu_agent_.handle != 0 &&
                              primary_device_id.has_value();
    auto plan = ResolveAmdProfilingPlan(opts_.profiling_engine, support);
    setResolvedPlan(plan);

    if (plan.degraded) {
        GFL_LOG_WARN("[ROCProfilerBackend] Requested ",
                     AmdRequestIntentWireName(plan.requested_engine),
                     "; selected ", AmdSelectedPathWireName(plan.selected_path),
                     " (", plan.reason_code, ")");
    }

    if (plan.selected_path == AmdProfilingPath::DispatchCounting) {
        engine_ = std::make_unique<DispatchCounterEngine>();
        if (!engine_->initialize(context_, primary_gpu_agent_,
                                 *primary_device_id, opts_)) {
            GFL_LOG_ERROR(
                "[ROCProfilerBackend] Dispatch-counter initialization failed; "
                "continuing with ROCprofiler trace activity only");
            engine_.reset();
            support.dispatch_counting = false;
            setResolvedPlan(
                ResolveAmdProfilingPlan(opts_.profiling_engine, support));
        }
    } else if (plan.selected_path == AmdProfilingPath::DeviceCounting) {
        engine_ = std::make_unique<DeviceCounterEngine>();
        if (!engine_->initialize(context_, primary_gpu_agent_,
                                 *primary_device_id, opts_)) {
            GFL_LOG_ERROR(
                "[ROCProfilerBackend] Device-counter initialization failed; "
                "continuing with ROCprofiler trace activity only");
            engine_.reset();
            support.device_counting = false;
            setResolvedPlan(
                ResolveAmdProfilingPlan(opts_.profiling_engine, support));
        }
    }

    tool_registered_.store(true, std::memory_order_release);
    return 0;
}

void RocprofilerBackend::toolFinalize() {
    if (buffer_.handle != 0) {
        flushBuffers();
        (void) CheckStatus(rocprofiler_destroy_buffer(buffer_),
                           "rocprofiler_destroy_buffer");
    }

    resetToolState();
    g_pending_backend.store(nullptr, std::memory_order_release);
}

void RocprofilerBackend::start() {
    if (!initialized_.load(std::memory_order_acquire) ||
        context_.handle == 0) return;
    start_requested_.store(true, std::memory_order_release);
    (void) tryStartContext(true);
}

bool RocprofilerBackend::tryStartContext(const bool force) {
    if (!start_requested_.load(std::memory_order_acquire) ||
        !initialized_.load(std::memory_order_acquire) ||
        context_.handle == 0 ||
        active_.load(std::memory_order_acquire)) {
        return active_.load(std::memory_order_acquire);
    }

    const int64_t now_ns = detail::GetTimestampNs();
    if (!force &&
        now_ns < next_start_retry_ns_.load(std::memory_order_relaxed)) {
        return false;
    }

    std::lock_guard lock(start_stop_mutex_);
    if (!start_requested_.load(std::memory_order_acquire) ||
        active_.load(std::memory_order_acquire)) {
        return active_.load(std::memory_order_acquire);
    }

    const auto status = rocprofiler_start_context(context_);
    if (status == ROCPROFILER_STATUS_SUCCESS) {
        active_.store(true, std::memory_order_release);
        deferred_start_logged_.store(false, std::memory_order_relaxed);
        start_failure_logged_.store(false, std::memory_order_relaxed);
        if (engine_) engine_->start();
        return true;
    }

    // gpufl::init commonly runs before the application's first HIP call.
    // ROCprofiler cannot start until that call loads HSA, so retain the
    // configured context and retry from the collector or first scope.
    next_start_retry_ns_.store(now_ns + 10'000'000,
                               std::memory_order_relaxed);
    if (status == ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED) {
        if (!deferred_start_logged_.exchange(true,
                                             std::memory_order_relaxed)) {
            GFL_LOG_DEBUG(
                "[ROCProfilerBackend] deferring context start until HSA is "
                "loaded");
        }
    } else if (!start_failure_logged_.exchange(
                   true, std::memory_order_relaxed)) {
        GFL_LOG_ERROR("[ROCProfilerBackend] rocprofiler_start_context failed: ",
                      StatusToString(status));
    }
    return false;
}

void RocprofilerBackend::stop() {
    start_requested_.store(false, std::memory_order_release);
    std::lock_guard lock(start_stop_mutex_);
    if (!active_.exchange(false) || context_.handle == 0) return;
    if (engine_) engine_->stop();
    (void) CheckStatus(rocprofiler_stop_context(context_),
                       "rocprofiler_stop_context");
    flushBuffers();
}

void RocprofilerBackend::DrainProfilingData() {
    // SegmentRuntime calls this immediately before choosing a segment
    // boundary. Deliver the ROCprofiler activity buffer first so completed
    // kernels and copies enter g_monitorBuffer while the retiring segment is
    // still current; Monitor then drains that ring before publishing the next
    // segment. Without this flush, a quiet workload can retain activity below
    // the ROCprofiler watermark until a later segment or shutdown.
    if (active_.load(std::memory_order_acquire)) flushBuffers();
    if (engine_) engine_->drain();
}

void RocprofilerBackend::ServiceDeepWindow() {
    if (!initialized_.load(std::memory_order_acquire)) return;
    if (!active_.load(std::memory_order_acquire) &&
        !tryStartContext(false)) return;
    if (DeepWindow::HasPendingWork()) DeepWindow::ServicePending();
    if (engine_) engine_->service();
}

bool RocprofilerBackend::DeepEnginesPrepared() const {
    return engine_ != nullptr && engine_->isPrepared();
}

std::vector<std::string> RocprofilerBackend::OnDeepWindowStop(
    const char* name) {
    std::vector<std::string> armed;
    if (engine_ && engine_->isArmed()) {
        armed.emplace_back(SelectedEngineWireName());
    }
    OnScopeStop(name);
    return armed;
}

void RocprofilerBackend::OnPerfScopeStart(const char* name) {
    if (!active_.load(std::memory_order_acquire)) {
        (void) tryStartContext(true);
    }
    if (opts_.deep_arm_mode == DeepArmMode::WindowOnly) return;
    OnDeepWindowPerfStart(name);
}

void RocprofilerBackend::OnPerfScopeStop(const char* name) {
    if (opts_.deep_arm_mode == DeepArmMode::WindowOnly) return;
    OnDeepWindowPerfStop(name);
}

void RocprofilerBackend::OnDeepWindowPerfStart(const char* name) {
    if (!active_.load(std::memory_order_acquire)) {
        (void) tryStartContext(true);
    }
    if (engine_) engine_->onScopeStart(name);
}

void RocprofilerBackend::OnDeepWindowPerfStop(const char* name) {
    if (engine_) engine_->onScopeStop(name);
}

void RocprofilerBackend::emitCapabilities() {
    const Runtime* runtime_state = runtime();
    const auto segment =
        runtime_state ? runtime_state->acquireSegmentContext() : nullptr;
    if (!segment || !segment->logger) return;

    std::lock_guard lock(capture_capabilities_mutex_);
    if (capture_capabilities_session_id_ == segment->session_id) return;

    const auto delta = [](const uint64_t value, const uint64_t baseline) {
        return value >= baseline ? value - baseline : value;
    };
    const uint64_t kernel_rows =
        kernel_rows_emitted_.load(std::memory_order_relaxed);
    const uint64_t memcpy_rows =
        memcpy_rows_emitted_.load(std::memory_order_relaxed);
    const uint64_t dropped_records =
        trace_records_dropped_.load(std::memory_order_relaxed);
    const uint64_t queue_dropped_records =
        trace_records_queue_dropped_.load(std::memory_order_relaxed);
    const uint64_t buffer_flush_failures =
        trace_buffer_flush_failures_.load(std::memory_order_relaxed);
    const uint64_t unattributed_records =
        trace_records_unattributed_.load(std::memory_order_relaxed);
    const uint64_t scope_correlation_failures =
        scope_correlation_failures_.load(std::memory_order_relaxed);
    const uint64_t pm_sample_rows = Monitor::PmSampleRowsSeen();

    AmdCaptureCapabilityInput input;
    input.session_id = segment->session_id;
    input.ts_ns = detail::GetTimestampNs();
    input.plan = resolvedPlan();
    input.trace_configured =
        tool_registered_.load(std::memory_order_acquire);
    input.kernel_rows =
        delta(kernel_rows, capability_kernel_rows_baseline_);
    input.memcpy_rows =
        delta(memcpy_rows, capability_memcpy_rows_baseline_);
    input.profiling_sample_rows =
        engine_ && engine_->hasData() ? 1 : 0;
    input.pm_sample_rows =
        delta(pm_sample_rows, capability_pm_sample_rows_baseline_);
    input.dropped_trace_records =
        delta(dropped_records, capability_dropped_records_baseline_);
    input.dropped_client_records = delta(
        queue_dropped_records, capability_queue_dropped_records_baseline_);
    input.trace_buffer_flush_failures = delta(
        buffer_flush_failures, capability_buffer_flush_failures_baseline_);
    input.unattributed_trace_records = delta(
        unattributed_records, capability_unattributed_records_baseline_);
    input.scope_correlation_failures = delta(
        scope_correlation_failures,
        capability_scope_correlation_failures_baseline_);

    segment->logger->write(model::CaptureCapabilitiesModel(
        BuildAmdCaptureCapabilitiesEvent(input)));

    capability_kernel_rows_baseline_ = kernel_rows;
    capability_memcpy_rows_baseline_ = memcpy_rows;
    capability_pm_sample_rows_baseline_ = pm_sample_rows;
    capability_dropped_records_baseline_ = dropped_records;
    capability_queue_dropped_records_baseline_ = queue_dropped_records;
    capability_buffer_flush_failures_baseline_ = buffer_flush_failures;
    capability_unattributed_records_baseline_ = unattributed_records;
    capability_scope_correlation_failures_baseline_ =
        scope_correlation_failures;
    capture_capabilities_session_id_ = segment->session_id;
}

void RocprofilerBackend::shutdown() {
    if (!initialized_.exchange(false)) return;

    stop();
    emitCapabilities();

    if (engine_) {
        engine_->shutdown();
        engine_.reset();
    }

    if (client_finalize_) {
        rocprofiler_client_id_t client_id = {
            sizeof(rocprofiler_client_id_t), "gpufl-amd", client_handle_};
        client_finalize_(client_id);
        client_finalize_ = nullptr;
    } else {
        toolFinalize();
    }
}

bool RocprofilerBackend::flushBuffers() {
    if (buffer_.handle == 0) return true;
    if (CheckStatus(rocprofiler_flush_buffer(buffer_),
                    "rocprofiler_flush_buffer")) {
        return true;
    }
    trace_buffer_flush_failures_.fetch_add(1, std::memory_order_relaxed);
    return false;
}

void RocprofilerBackend::OnScopeStart(const char* name) {
    if (context_.handle == 0 || name == nullptr) return;
    if (!active_.load(std::memory_order_acquire) &&
        !tryStartContext(true)) return;

    rocprofiler_thread_id_t tid{};
    if (!CheckStatus(rocprofiler_get_thread_id(&tid),
                     "rocprofiler_get_thread_id(scope start)")) {
        scope_correlation_failures_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // Thread-local scope state can outlive a backend lifecycle when a user
    // shuts GPUFlight down inside an open scope. The metadata map is cleared
    // with the ROCprofiler context, so a missing top ID identifies a stale
    // stack and prevents it from contaminating the next session's scope path.
    if (!g_scope_external_stack.empty()) {
        std::lock_guard<std::mutex> lock(external_scope_mutex_);
        if (external_scope_metadata_.count(g_scope_external_stack.back()) == 0) {
            g_scope_external_stack.clear();
            g_scope_name_stack.clear();
        }
    }

    const uint64_t external_value =
        next_scope_external_.fetch_add(1, std::memory_order_relaxed);

    std::string scope_path;
    for (const auto& component : g_scope_name_stack) {
        if (!scope_path.empty()) scope_path += "|";
        scope_path += component;
    }
    if (!scope_path.empty()) scope_path += "|";
    scope_path += name;

    rocprofiler_user_data_t user_data{};
    user_data.value = external_value;
    if (!CheckStatus(rocprofiler_push_external_correlation_id(
                         context_, tid, user_data),
                     "rocprofiler_push_external_correlation_id")) {
        scope_correlation_failures_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    g_scope_name_stack.emplace_back(name);
    g_scope_external_stack.push_back(external_value);
    {
        std::lock_guard<std::mutex> lock(external_scope_mutex_);
        external_scope_metadata_[external_value] = ExternalScopeMetadata{
            std::move(scope_path), static_cast<int>(g_scope_name_stack.size())};
    }
}

void RocprofilerBackend::OnScopeStop(const char* name) {
    if (!active_.load() || context_.handle == 0) return;

    rocprofiler_thread_id_t tid{};
    if (!CheckStatus(rocprofiler_get_thread_id(&tid),
                     "rocprofiler_get_thread_id(scope stop)")) {
        scope_correlation_failures_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    rocprofiler_user_data_t user_data{};
    if (!CheckStatus(rocprofiler_pop_external_correlation_id(
                         context_, tid, &user_data),
                     "rocprofiler_pop_external_correlation_id")) {
        scope_correlation_failures_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    bool matched = !g_scope_external_stack.empty() &&
                   g_scope_external_stack.back() == user_data.value;
    if (name && !g_scope_name_stack.empty()) {
        matched = matched && g_scope_name_stack.back() == name;
    }
    if (!matched) {
        scope_correlation_failures_.fetch_add(1, std::memory_order_relaxed);
        GFL_LOG_ERROR(
            "[ROCProfilerBackend] scope correlation stack mismatch; "
            "discarding the local top to preserve push/pop depth");
    }
    if (!g_scope_external_stack.empty()) g_scope_external_stack.pop_back();
    if (!g_scope_name_stack.empty()) g_scope_name_stack.pop_back();
}

void RocprofilerBackend::callbackTracingShim(rocprofiler_callback_tracing_record_t record,
                                             rocprofiler_user_data_t*,
                                             void* callback_data) {
    auto* backend = static_cast<RocprofilerBackend*>(callback_data);
    if (!backend || record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
        record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD ||
        record.payload == nullptr) {
        return;
    }

    if (record.operation ==
        static_cast<rocprofiler_tracing_operation_t>(ROCPROFILER_CODE_OBJECT_LOAD)) {
        const auto* loadData =
            static_cast<const rocprofiler_callback_tracing_code_object_load_data_t*>(
                record.payload);
        backend->handleCodeObjectLoad(*loadData);
        return;
    }

    if (record.operation !=
        static_cast<rocprofiler_tracing_operation_t>(
            ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)) {
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
                                           const uint64_t drop_count) {
    auto* backend = static_cast<RocprofilerBackend*>(data);
    if (!backend) return;
    backend->trace_records_dropped_.fetch_add(drop_count,
                                              std::memory_order_relaxed);
    if (headers == nullptr) return;

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
        const auto& mangled = itr->second.name;
        // Check demangle cache (avoids repeated abi::__cxa_demangle calls)
        if (auto dit = demangle_cache_.find(mangled); dit != demangle_cache_.end()) {
            return dit->second;
        }
        auto demangled = core::DemangleName(mangled.c_str());
        demangle_cache_[mangled] = demangled;
        return demangled;
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
    backend->gpu_arch_props_.clear();
    backend->primary_gpu_agent_ = {};

    for (size_t i = 0; i < num_agents; ++i) {
        const auto* agent = static_cast<const rocprofiler_agent_t*>(agents[i]);
        if (!agent) continue;

        backend->agent_types_[agent->id.handle] = agent->type;
        if (agent->type == ROCPROFILER_AGENT_TYPE_GPU &&
            agent->logical_node_type_id >= 0) {
            const int device_id = agent->logical_node_type_id;
            backend->gpu_device_ids_[agent->id.handle] = device_id;

            // Track the first attributable GPU agent for profiling engine
            // initialization. Dispatch counting remains single-agent today.
            if (backend->primary_gpu_agent_.handle == 0) {
                backend->primary_gpu_agent_ = agent->id;
            }

            GpuArchProps props{};
            props.wave_front_size =
                agent->wave_front_size > 0 ? agent->wave_front_size : 64;
            props.max_waves_per_cu = agent->max_waves_per_cu;
            props.simd_per_cu = agent->simd_per_cu;
            props.lds_size_bytes = agent->lds_size_in_kb * 1024;
            props.cu_count = agent->cu_count;
            props.workgroup_max_size = agent->workgroup_max_size;
            backend->gpu_arch_props_[device_id] = props;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

AmdTraceEndpoint RocprofilerBackend::resolveTraceEndpoint(
    const rocprofiler_agent_id_t agent_id) const {
    std::lock_guard<std::mutex> lock(agent_mutex_);
    AmdTraceEndpoint endpoint;

    const auto type_it = agent_types_.find(agent_id.handle);
    if (type_it == agent_types_.end()) return endpoint;
    if (type_it->second == ROCPROFILER_AGENT_TYPE_CPU) {
        endpoint.kind = AmdTraceAgentKind::Cpu;
        return endpoint;
    }
    if (type_it->second != ROCPROFILER_AGENT_TYPE_GPU) return endpoint;

    endpoint.kind = AmdTraceAgentKind::Gpu;
    if (const auto device_it = gpu_device_ids_.find(agent_id.handle);
        device_it != gpu_device_ids_.end()) {
        endpoint.device_id = static_cast<uint32_t>(device_it->second);
    }
    return endpoint;
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

void RocprofilerBackend::handleCodeObjectLoad(
    const rocprofiler_callback_tracing_code_object_load_data_t& data) {

    std::vector<uint8_t> elf_bytes;

    if (data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
        const auto* ptr = reinterpret_cast<const uint8_t*>(data.memory_base);
        if (ptr && data.memory_size > 0)
            elf_bytes.assign(ptr, ptr + data.memory_size);
    } else if (data.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) {
        // Read code object from file descriptor at load_base offset with load_size
        if (data.storage_file >= 0 && data.load_size > 0) {
            elf_bytes.resize(data.load_size);
            auto saved = ::lseek(data.storage_file, 0, SEEK_CUR);
            if (::lseek(data.storage_file, static_cast<off_t>(data.load_base), SEEK_SET) >= 0) {
                auto rd = ::read(data.storage_file, elf_bytes.data(), data.load_size);
                if (rd < 0 || static_cast<uint64_t>(rd) != data.load_size)
                    elf_bytes.clear();
            } else {
                elf_bytes.clear();
            }
            if (saved >= 0) ::lseek(data.storage_file, saved, SEEK_SET);
        }
    }

    if (elf_bytes.size() < 4) return;

    // Verify ELF magic
    if (elf_bytes[0] != 0x7f || elf_bytes[1] != 'E' ||
        elf_bytes[2] != 'L' || elf_bytes[3] != 'F') {
        return;
    }

    uint64_t crc = ::crc32(0L, elf_bytes.data(),
                           static_cast<uInt>(elf_bytes.size()));

    {
        std::lock_guard<std::mutex> lock(code_object_mutex_);
        if (enqueued_disasm_crcs_.count(crc)) return;
        enqueued_disasm_crcs_.insert(crc);
    }

    GFL_LOG_DEBUG("[ROCProfilerBackend] code object loaded: id=", data.code_object_id,
                  " size=", elf_bytes.size(), " crc=", crc);

    Monitor::EnqueueCubinForDisassembly(crc, elf_bytes.data(), elf_bytes.size());
}

void RocprofilerBackend::handleKernelDispatch(
    const rocprofiler_kernel_dispatch_info_t& info,
    const uint64_t start_timestamp,
    const uint64_t end_timestamp,
    const rocprofiler_async_correlation_id_t& correlation_id) {
    const auto device_id =
        ResolveAmdKernelDeviceId(resolveTraceEndpoint(info.agent_id));
    if (!device_id.has_value()) {
        trace_records_unattributed_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    ActivityRecord out{};
    out.type = TraceType::KERNEL;
    out.device_id = *device_id;
    out.stream = static_cast<StreamHandle>(info.queue_id.handle);
    out.cpu_start_ns = static_cast<int64_t>(start_timestamp);
    out.duration_ns =
        static_cast<int64_t>(end_timestamp >= start_timestamp ? end_timestamp - start_timestamp
                                                              : 0);
    out.api_start_ns = out.cpu_start_ns;
    out.api_exit_ns = out.cpu_start_ns + out.duration_ns;
    out.corr_id = TruncateCorrelationId(correlation_id.internal);
    out.has_details = true;
    // ROCprofiler grid_size is total threads (not blocks like CUDA).
    // Convert to block count by dividing by workgroup size.
    const auto wg_x = (std::max)(info.workgroup_size.x, 1u);
    const auto wg_y = (std::max)(info.workgroup_size.y, 1u);
    const auto wg_z = (std::max)(info.workgroup_size.z, 1u);
    out.grid_x = static_cast<int>(info.grid_size.x / wg_x);
    out.grid_y = static_cast<int>(info.grid_size.y / wg_y);
    out.grid_z = static_cast<int>(info.grid_size.z / wg_z);
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
            out.arch_vgpr_count = static_cast<int>(meta.arch_vgpr_count);
            if (out.static_shared == 0) out.static_shared = static_cast<int>(meta.group_segment_size);
            if (out.local_bytes == 0) out.local_bytes = static_cast<int>(meta.private_segment_size);
        }
    }

    // Compute occupancy from architecture properties
    {
        int dev_id = static_cast<int>(out.device_id);
        auto pit = gpu_arch_props_.find(dev_id);
        if (pit != gpu_arch_props_.end()) {
            const auto& props = pit->second;
            int threadsPerBlock = out.block_x * out.block_y * out.block_z;
            if (threadsPerBlock > 0 && props.max_waves_per_cu > 0 && props.wave_front_size > 0) {
                int wavesPerBlock = (threadsPerBlock + static_cast<int>(props.wave_front_size) - 1)
                                    / static_cast<int>(props.wave_front_size);
                int maxWavesPerCU = static_cast<int>(props.max_waves_per_cu);

                // Wave/workgroup limit
                int waveBlocks = (wavesPerBlock > 0) ? (maxWavesPerCU / wavesPerBlock) : 0;

                // LDS (shared memory) limit
                int smemPerBlock = out.static_shared + out.dyn_shared;
                int ldsBytes = static_cast<int>(props.lds_size_bytes);
                int smemBlocks = (smemPerBlock > 0 && ldsBytes > 0) ? (ldsBytes / smemPerBlock) : waveBlocks;

                // VGPR register limit
                // Use arch_vgpr_count (not combined num_regs which includes SGPRs).
                // RDNA architectures have 1536 VGPRs per SIMD unit, allocated per-wave
                // in granularity of 16 VGPRs (RDNA 1/2) or 24 VGPRs (RDNA 3/4).
                // We use the max_waves_per_cu / simd_per_cu to derive VGPRs per SIMD:
                //   total_vgprs_per_simd = max_waves_per_simd * max_vgprs_per_wave
                // For RDNA 3/4: 1536 VGPRs per SIMD, 2 SIMDs per CU = 3072 VGPRs/CU.
                constexpr int kVgprAllocGranularity = 8;
                constexpr int kVgprsPerSimd = 1536;  // RDNA standard
                int vgprsPerThread = out.arch_vgpr_count;
                int vgprsAligned = (vgprsPerThread > 0)
                    ? (((vgprsPerThread + kVgprAllocGranularity - 1) / kVgprAllocGranularity)
                       * kVgprAllocGranularity)
                    : 0;
                int regBlocks = waveBlocks;  // default: not limited by registers
                if (vgprsAligned > 0 && props.simd_per_cu > 0) {
                    int totalVgprsPerCU = kVgprsPerSimd * static_cast<int>(props.simd_per_cu);
                    int regWavesPerCU = totalVgprsPerCU / vgprsAligned;
                    regBlocks = (wavesPerBlock > 0) ? (regWavesPerCU / wavesPerBlock) : waveBlocks;
                }

                // Overall occupancy = minimum of all limits
                int activeBlocks = (std::min)({waveBlocks, smemBlocks, regBlocks});
                activeBlocks = (std::max)(activeBlocks, 0);

                auto toOcc = [&](int blocks) -> float {
                    return (maxWavesPerCU > 0 && wavesPerBlock > 0)
                        ? (std::min)(1.0f, static_cast<float>(blocks * wavesPerBlock) / maxWavesPerCU)
                        : 0.0f;
                };

                out.occupancy = toOcc(activeBlocks);
                out.warp_occupancy = toOcc(waveBlocks);
                out.smem_occupancy = toOcc(smemBlocks);
                out.reg_occupancy = toOcc(regBlocks);
                out.block_occupancy = 1.0f;  // AMD doesn't have a hard block-per-CU limit like NVIDIA
                out.max_active_blocks = activeBlocks;

                struct { float occ; const char* name; } limiters[] = {
                    {out.warp_occupancy, "waves"},
                    {out.reg_occupancy, "registers"},
                    {out.smem_occupancy, "shared_mem"},
                };
                const char* limiting = "waves";
                float minOcc = out.warp_occupancy;
                for (auto& l : limiters) {
                    if (l.occ < minOcc) { minOcc = l.occ; limiting = l.name; }
                }
                std::snprintf(out.limiting_resource, sizeof(out.limiting_resource), "%s", limiting);
            }
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

    if (g_monitorBuffer.Push(out)) {
        kernel_rows_emitted_.fetch_add(1, std::memory_order_relaxed);
    } else {
        trace_records_queue_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
}

void RocprofilerBackend::handleMemoryCopy(
    const rocprofiler_buffer_tracing_memory_copy_record_t& data) {
    const auto source = resolveTraceEndpoint(data.src_agent_id);
    const auto destination = resolveTraceEndpoint(data.dst_agent_id);
    const auto device_id =
        ResolveAmdMemoryCopyDeviceId(source, destination);
    const bool host_only = source.kind == AmdTraceAgentKind::Cpu &&
                           destination.kind == AmdTraceAgentKind::Cpu;
    if (!device_id.has_value() && !host_only) {
        trace_records_unattributed_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    ActivityRecord out{};
    out.type = TraceType::MEMCPY;
    out.device_id = device_id.value_or(0);
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

    // Memory-kind wire values: 1=host, 2=device.
    out.src_kind = source.kind == AmdTraceAgentKind::Gpu ? 2 : 1;
    out.dst_kind = destination.kind == AmdTraceAgentKind::Gpu ? 2 : 1;

    if (data.correlation_id.external.value != 0) {
        std::lock_guard<std::mutex> lock(external_scope_mutex_);
        if (auto itr = external_scope_metadata_.find(data.correlation_id.external.value);
            itr != external_scope_metadata_.end() && !itr->second.user_scope.empty()) {
            std::snprintf(out.user_scope, sizeof(out.user_scope), "%s",
                          itr->second.user_scope.c_str());
            out.scope_depth = itr->second.scope_depth;
        }
    }

    if (g_monitorBuffer.Push(out)) {
        memcpy_rows_emitted_.fetch_add(1, std::memory_order_relaxed);
    } else {
        trace_records_queue_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace gpufl::amd
