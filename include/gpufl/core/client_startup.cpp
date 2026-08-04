#include "gpufl/core/client_startup.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "gpufl/gpufl.hpp"
#include "gpufl/backends/host_collector.hpp"
#include "gpufl/core/backend_factory.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"
#include "gpufl/core/model/system_event_model.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_configuration.hpp"
#include "gpufl/core/remote_config.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/segment_runtime.hpp"
#include "gpufl/core/session_bootstrap.hpp"
#include "gpufl/core/startup_configuration.hpp"

#if GPUFL_HAS_CUDA || defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace gpufl::detail {
namespace {

bool windowsInjectedProcess() {
#if defined(_WIN32)
    const char* injected = std::getenv(env::kInject);
    return injected && std::string(injected) == "1";
#else
    return false;
#endif
}

void configureEagerModuleLoading(const MonitorOptions& options) {
    // EAGER module loading is opt-in. It avoids the known SASS lazy-patching
    // deadlock, but carries a whole-process startup/memory cost, so it must run
    // before the first CUDA call and must never override a user's own setting.
    if (options.profiling_engine != ProfilingEngine::SassMetrics &&
        options.profiling_engine != ProfilingEngine::Deep) {
        return;
    }
    const char* knob_env = std::getenv(env::kEagerModuleLoading);
    const std::string knob = knob_env ? knob_env : "";
    const bool opted_in = knob == "1" || knob == "true" ||
                          knob == "yes" || knob == "on";
    if (!opted_in || std::getenv(env::kCudaModuleLoading) != nullptr) return;
#if defined(_WIN32)
    _putenv_s(env::kCudaModuleLoading, "EAGER");
#else
    setenv(env::kCudaModuleLoading, "EAGER", /*overwrite=*/0);
#endif
    GFL_LOG_DEBUG("[gpufl] CUDA_MODULE_LOADING=EAGER set "
                  "(GPUFL_EAGER_MODULE_LOADING opt-in) for SASS/Deep.");
}

void autoTuneKernelSampleRate(const InitOptions& init_options,
                              MonitorOptions& monitor_options) {
#if GPUFL_HAS_CUDA || defined(__CUDACC__)
    if (monitor_options.kernel_sample_rate_ms <= 0 ||
        monitor_options.kernel_sample_rate_ms >= 200 ||
        (monitor_options.profiling_engine != ProfilingEngine::SassMetrics &&
         monitor_options.profiling_engine != ProfilingEngine::Deep)) {
        return;
    }
    cudaDeviceProp prop{};
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return;
    }
    if (prop.major < 12 &&
        monitor_options.kernel_sample_rate_ms == init_options.kernel_sample_rate_ms) {
        GFL_LOG_DEBUG("[gpufl] Auto-tuning kernel_sample_rate_ms 50 -> 200 "
                      "on sm_", prop.major, prop.minor,
                      " (SASS metrics have significant per-launch overhead "
                      "on pre-sm_120 GPUs). Set the value explicitly to override.");
        monitor_options.kernel_sample_rate_ms = 200;
    }
#else
    (void)init_options;
    (void)monitor_options;
#endif
}

}  // namespace

class ClientStartup::State {
public:
    StartupSegmentationOptions segmentation;
    InitialSessionLoggingState logging;
    MonitorOptions monitor_options;
    InitEvent initial_event;
};

ClientStartup::ClientStartup(InitOptions& active_options)
    : options_(active_options), state_(std::make_unique<State>()) {}

ClientStartup::~ClientStartup() = default;

bool ClientStartup::start() {
    if (!resolveConfiguration()) return false;
    if (!createRuntime()) return false;

    launchVersionProbe();
    set_runtime(std::move(pending_runtime_));
    return startMonitor();
}

bool ClientStartup::resolveConfiguration() {
    resolveStartupOptions(options_);
    DebugLogger::setEnabled(options_.enable_debug_output);
    GFL_LOG_DEBUG("Initializing...");

    std::string error;
    if (!readStartupSegmentationOptions(state_->segmentation, error)) {
        GFL_LOG_ERROR(error);
        return false;
    }
    segmented_ = state_->segmentation.enabled();
    return true;
}

bool ClientStartup::createRuntime() {
    if (runtime()) {
        GFL_LOG_DEBUG("Runtime already exists, shutting down first...");
        shutdown();
    }

    pending_runtime_ = std::make_unique<Runtime>();
    pending_runtime_->app_name = options_.app_name.empty() ? "gpufl" : options_.app_name;
    pending_runtime_->session_id = GenerateSessionId();
    if (segmented_) {
        pending_runtime_->run_id = state_->segmentation.run_id;
        pending_runtime_->segment_index = 0;
        pending_runtime_->segment_every_ms =
            static_cast<int64_t>(state_->segmentation.segment_every_ms);
        pending_runtime_->segment_max_rows = state_->segmentation.segment_max_rows;
        pending_runtime_->run_roll_every_ms =
            static_cast<int64_t>(state_->segmentation.run_roll_every_ms);
        pending_runtime_->run_roll_max_bytes =
            state_->segmentation.run_roll_max_bytes;
    }
    pending_runtime_->logger = std::make_shared<Logger>();
    pending_runtime_->host_collector = std::make_unique<HostCollector>();
    return openInitialSessionLogging(*pending_runtime_, options_, segmented_,
                                     state_->logging);
}

void ClientStartup::launchVersionProbe() const {
    // Bounded, detached version discovery is advisory: offline/file-only use
    // must never block or fail initialization.
    std::string probe_url;
    if (const char* value = std::getenv(env::kBackendUrl)) probe_url = value;
    else if (const char* value = std::getenv(env::kRemoteConfig)) probe_url = value;
    if (probe_url.empty()) return;
    std::thread([url = std::move(probe_url), api_path = options_.api_path] {
        probeBackendVersion(url, api_path);
    }).detach();
}

bool ClientStartup::startMonitor() {
    GFL_LOG_DEBUG("Initializing Monitor (CUPTI)...");
    state_->monitor_options = buildMonitorOptions(options_);
    configureEagerModuleLoading(state_->monitor_options);
    autoTuneKernelSampleRate(options_, state_->monitor_options);
    Monitor::Initialize(state_->monitor_options);

    GFL_LOG_DEBUG("Starting Monitor...");
    Monitor::Start();
    GFL_LOG_DEBUG("Monitor started");
    return activateRuntime();
}

bool ClientStartup::activateRuntime() {
    Runtime* const active_runtime = runtime();
    const auto segment = active_runtime
        ? active_runtime->acquireSegmentContext("client_startup")
        : nullptr;
    if (!segment) {
        GFL_LOG_ERROR("Missing active segment context before job_start");
        return false;
    }

    configureCollectors(*active_runtime);
    emitInitialEvent(*active_runtime, *segment);
    Monitor::EmitSegmentMetadata();
    if (!startSegmentRuntime(*active_runtime)) {
        GFL_LOG_ERROR("Failed to start SegmentRuntime");
        shutdown();
        return false;
    }
    configureSampler(*active_runtime);
    startContinuousSampling(*active_runtime, *segment);
    return true;
}

void ClientStartup::configureCollectors(Runtime& active_runtime) const {
    std::string backend_reason;
    auto collectors = CreateBackendCollectors(options_.backend, &backend_reason);
    active_runtime.unified_gpu_collector = std::move(collectors.unified_collector);
    active_runtime.collector = std::move(collectors.telemetry_collector);
    active_runtime.static_info_collector =
        std::move(collectors.static_info_collector);
    if (!active_runtime.collector) {
        GFL_LOG_ERROR("Failed to initialize GPU backend: ", backend_reason);
    }
}

void ClientStartup::emitInitialEvent(Runtime& active_runtime,
                                     const SegmentContext& segment) {
    InitEvent event;
    event.pid = GetPid();
    event.session_id = segment.session_id;
    event.app = active_runtime.app_name;
    event.log_path = state_->logging.log_path;
    event.ts_ns = GetTimestampNs();
    if (active_runtime.collector) {
        event.devices = active_runtime.collector->sampleAll();
    }
    const bool skip_static_info = windowsInjectedProcess();
    if (active_runtime.static_info_collector && !skip_static_info) {
        event.gpu_static_device_infos =
            active_runtime.static_info_collector->sampleStaticInfo();
    } else if (skip_static_info) {
        GFL_LOG_DEBUG("Skipping CUDA static GPU inventory during Windows injection init.");
    }
    event.host = active_runtime.host_collector->sample();
    event.session_kind = ProfilingEngineSessionKind(state_->monitor_options.profiling_engine);
    event.profiling_engine = ProfilingEngineWireName(state_->monitor_options.profiling_engine);
    event.run_id = segment.run_id;
    event.segment_index = segment.segment_index;
    if (segment.run_part) {
        event.roll_chain_id = segment.run_part->roll_chain_id;
        event.previous_run_id = segment.run_part->previous_run_id;
        event.part_index = segment.run_part->part_index;
    }

    // The launcher tags each child of a multi-pass analysis through the
    // environment; absent fields preserve the ordinary single-pass wire form.
    if (const char* analysis_id = std::getenv(env::kAnalysisId);
        analysis_id && *analysis_id) {
        event.analysis_id = analysis_id;
        if (const char* pass_index = std::getenv(env::kPassIndex)) {
            event.pass_index = std::atoi(pass_index);
        }
        if (const char* pass_count = std::getenv(env::kPassCount)) {
            event.pass_count = std::atoi(pass_count);
        }
        GFL_LOG_DEBUG("Multi-pass: analysis_id=", event.analysis_id,
                      " pass ", event.pass_index, "/", event.pass_count);
    }

    segment.logger->write(model::InitEventModel(event));
    state_->initial_event = std::move(event);
}

bool ClientStartup::startSegmentRuntime(Runtime& active_runtime) {
    if (!segmented_) return true;

    SegmentRuntime::Options options;
    options.runtime = &active_runtime;
    options.logger_options = state_->logging.options;
    options.logger_options.on_serialized_bytes = {};
    options.init_template = state_->initial_event;
    options.segment_every_ms = active_runtime.segment_every_ms;
    options.segment_max_rows = active_runtime.segment_max_rows;
    options.run_roll_every_ms = active_runtime.run_roll_every_ms;
    options.run_roll_max_bytes = active_runtime.run_roll_max_bytes;
    active_runtime.segment_runtime =
        std::make_shared<SegmentRuntime>(std::move(options));
    return active_runtime.segment_runtime->start();
}

void ClientStartup::configureSampler(Runtime& active_runtime) const {
    if (options_.system_sample_rate_ms <= 0 || !active_runtime.collector) return;
    Runtime* const runtime_ptr = &active_runtime;
    active_runtime.sampler.configure(
        active_runtime.app_name,
        [runtime_ptr] {
            return runtime_ptr->acquireSegmentContext("sampler");
        },
        [runtime_ptr] {
            return runtime_ptr->peekSegmentContext();
        },
        active_runtime.collector, options_.system_sample_rate_ms,
        active_runtime.host_collector.get(),
        [runtime_ptr](const uint32_t index, const uint64_t rows,
                      const int64_t steady_ns, const int64_t event_ns) {
            if (runtime_ptr->segment_runtime) {
                runtime_ptr->segment_runtime->noteRows(
                    index, rows, steady_ns, event_ns);
            }
        });
}

void ClientStartup::startContinuousSampling(
    Runtime& active_runtime, const SegmentContext& segment) const {
    if (options_.continuous_system_sampling && segment.logger) {
        SystemStartEvent event;
        event.pid = GetPid();
        event.app = active_runtime.app_name;
        event.name = "sampling_start";
        event.session_id = segment.session_id;
        event.ts_ns = GetTimestampNs();
        if (active_runtime.collector) event.devices = active_runtime.collector->sampleAll();
        if (active_runtime.host_collector) event.host = active_runtime.host_collector->sample();
        segment.logger->write(model::SystemStartModel(event));
    }
    if (options_.continuous_system_sampling && options_.system_sample_rate_ms > 0 &&
        active_runtime.collector) {
        active_runtime.sampler.activate();
    }
}

}  // namespace gpufl::detail
