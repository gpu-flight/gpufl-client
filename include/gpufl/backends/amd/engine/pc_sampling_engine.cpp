#if !(GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK)
#error "pc_sampling_engine.cpp requires GPUFL_ENABLE_AMD && GPUFL_HAS_ROCPROFILER_SDK"
#endif

#include "gpufl/backends/amd/engine/pc_sampling_engine.hpp"

#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <unordered_map>

#include "gpufl/core/common.hpp"
#include "gpufl/core/debug_logger.hpp"
#include "gpufl/core/monitor.hpp"

namespace gpufl::amd {
namespace {

// Lossless, like rocprofiler-sdk's own PC sampling sample. A stochastic
// record is about 100 bytes; delivery starts at a quarter full.
constexpr size_t kBufferBytes = size_t{4} << 20;
constexpr size_t kBufferWatermark = kBufferBytes / 4;

// Rows per Monitor::PushProfileSamples call, bounding the staging vector.
constexpr size_t kRowsPerPush = 4096;

// rocprofiler-sdk delivers the buffer from its own thread, possibly after
// the engine is gone (its finalizer flushes every PC sampling buffer). The
// callback reaches the engine through this registry, never a raw pointer.
std::mutex g_live_engine_mutex;
PcSamplingEngine* g_live_engine = nullptr;

const char* StatusName(const rocprofiler_status_t status) {
    const char* name = rocprofiler_get_status_name(status);
    return name ? name : "ROCPROFILER_STATUS_ERROR";
}

rocprofiler_status_t CollectConfigurations(
    const rocprofiler_pc_sampling_configuration_t* configs,
    const size_t num_config, void* user_data) {
    auto* out = static_cast<
        std::vector<rocprofiler_pc_sampling_configuration_t>*>(user_data);
    if (out && configs) out->assign(configs, configs + num_config);
    return ROCPROFILER_STATUS_SUCCESS;
}

// "stochastic cycles 256..2147483648, host_trap us 1..1000000": what the
// agent lists, for the run's log. The qualification records it.
std::string DescribeConfigurations(
    const std::vector<rocprofiler_pc_sampling_configuration_t>& configs) {
    std::string out;
    for (const auto& config : configs) {
        if (!out.empty()) out += ", ";
        switch (config.method) {
            case ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC:
                out += "stochastic";
                break;
            case ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP:
                out += "host_trap";
                break;
            default:
                out += "method_" + std::to_string(config.method);
                break;
        }
        switch (config.unit) {
            case ROCPROFILER_PC_SAMPLING_UNIT_CYCLES:
                out += " cycles ";
                break;
            case ROCPROFILER_PC_SAMPLING_UNIT_TIME:
                out += " us ";
                break;
            case ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS:
                out += " instructions ";
                break;
            default:
                out += " unit_" + std::to_string(config.unit) + " ";
                break;
        }
        out += std::to_string(config.min_interval) + ".." +
               std::to_string(config.max_interval);
    }
    return out.empty() ? std::string("nothing") : out;
}

// Whether flushDisassembly has parsed this code object's ISA.
bool IsaParsed(const uint64_t crc) {
    return Monitor::LookupAmdIsaSource(crc, std::string(), 0, nullptr) !=
           DictionaryManager::IsaSourceLookup::NotParsed;
}

}  // namespace

// No rocprofiler-sdk call here: the last reference can drop on any thread,
// after the SDK has finalized. shutdown() delivered the samples.
PcSamplingEngine::~PcSamplingEngine() {
    std::lock_guard lock(g_live_engine_mutex);
    if (g_live_engine == this) g_live_engine = nullptr;
}

bool PcSamplingEngine::refuse(const AmdPcSamplingRefusal refusal,
                              std::string detail) {
    refusal_ = refusal;
    refusal_detail_ = std::move(detail);
    GFL_LOG_DEBUG("[PcSamplingEngine] not configured (",
                  AmdPcSamplingRefusalReason(refusal), ") ", refusal_detail_);
    return false;
}

bool PcSamplingEngine::initialize(const rocprofiler_context_id_t context,
                                  const rocprofiler_agent_id_t gpu_agent,
                                  const uint32_t gpu_device_id,
                                  const MonitorOptions&) {
    context_ = context;
    agent_ = gpu_agent;
    device_id_ = gpu_device_id;

    // What the agent lists at this moment; another process's configuration
    // shows as its single interval.
    std::vector<rocprofiler_pc_sampling_configuration_t> configs;
    rocprofiler_status_t status =
        rocprofiler_query_pc_sampling_agent_configurations(
            agent_, &CollectConfigurations, &configs);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        return refuse(ClassifyAmdPcSamplingStatus(AmdPcSamplingCall::Query,
                                                  StatusName(status)),
                      StatusName(status));
    }
    // Only an opted-in run gets here, so the listing always prints.
    GFL_LOG_INFO("[PcSamplingEngine] the GPU lists PC sampling: ",
                 DescribeConfigurations(configs));
    const rocprofiler_pc_sampling_configuration_t* stochastic = nullptr;
    for (const auto& config : configs) {
        if (config.method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC &&
            config.unit == ROCPROFILER_PC_SAMPLING_UNIT_CYCLES) {
            stochastic = &config;
            break;
        }
    }
    if (stochastic == nullptr) {
        return refuse(AmdPcSamplingRefusal::MethodUnavailable, std::string());
    }
    const auto interval = ResolveAmdPcSamplingInterval(
        stochastic->min_interval, stochastic->max_interval);
    if (!interval) {
        return refuse(AmdPcSamplingRefusal::ConfigurationFailed,
                      "no power-of-2 interval from " +
                          std::to_string(stochastic->min_interval) + " to " +
                          std::to_string(stochastic->max_interval) +
                          " cycles");
    }

    status = rocprofiler_create_buffer(
        context_, kBufferBytes, kBufferWatermark,
        ROCPROFILER_BUFFER_POLICY_LOSSLESS, &PcSamplingEngine::bufferCallback,
        nullptr, &buffer_);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        return refuse(AmdPcSamplingRefusal::ConfigurationFailed,
                      StatusName(status));
    }
    // Aggregation runs on a thread of its own, off the trace buffer's.
    rocprofiler_callback_thread_t thread{};
    if (rocprofiler_create_callback_thread(&thread) !=
            ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_assign_callback_thread(buffer_, thread) !=
            ROCPROFILER_STATUS_SUCCESS) {
        GFL_LOG_WARN(
            "[PcSamplingEngine] no dedicated callback thread; PC samples "
            "share the default delivery thread");
    }

    status = rocprofiler_configure_pc_sampling_service(
        context_, agent_, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC,
        ROCPROFILER_PC_SAMPLING_UNIT_CYCLES, *interval, buffer_, 0);
    if (status != ROCPROFILER_STATUS_SUCCESS) {
        return refuse(ClassifyAmdPcSamplingStatus(AmdPcSamplingCall::Configure,
                                                  StatusName(status)),
                      StatusName(status));
    }

    interval_ = *interval;
    // Rows carry the source line of their ISA row.
    Monitor::RetainAmdIsaLines();
    {
        std::lock_guard lock(g_live_engine_mutex);
        g_live_engine = this;
    }
    configured_.store(true, std::memory_order_release);
    GFL_LOG_INFO("[PcSamplingEngine] stochastic PC sampling every ",
                 interval_, " cycles (rocprofiler-sdk beta)");
    return true;
}

void PcSamplingEngine::start() {
    started_.store(true, std::memory_order_release);
}

void PcSamplingEngine::stop() {
    started_.store(false, std::memory_order_release);
}

void PcSamplingEngine::flush() {
    if (buffer_.handle == 0) return;
    if (const auto status = rocprofiler_flush_buffer(buffer_);
        status != ROCPROFILER_STATUS_SUCCESS) {
        GFL_LOG_DEBUG("[PcSamplingEngine] rocprofiler_flush_buffer: ",
                      StatusName(status));
    }
}

void PcSamplingEngine::drain() {
    if (!isPrepared()) return;
    flush();
    pushRows(false);
}

void PcSamplingEngine::finish() {
    if (!isPrepared()) return;
    // The context has stopped: deliver the last samples, then let the ISA
    // line tables catch up so these rows carry their source lines. A later
    // call finds nothing left unless the context ran again.
    flush();
    Monitor::FlushDisassemblyNow();
    pushRows(true);
}

void PcSamplingEngine::shutdown() {
    finish();
    std::lock_guard lock(g_live_engine_mutex);
    if (g_live_engine == this) g_live_engine = nullptr;
}

bool PcSamplingEngine::hasData() const {
    std::lock_guard lock(mutex_);
    return aggregator_.totals().samples > 0;
}

PcSamplingEngine::Stats PcSamplingEngine::stats() const {
    std::lock_guard lock(mutex_);
    Stats stats;
    stats.interval = interval_;
    stats.totals = aggregator_.totals();
    stats.lost = invalid_records_ + dropped_records_ + stats.totals.unmatched;
    return stats;
}

void PcSamplingEngine::onCodeObjectLoaded(const uint64_t code_object_id,
                                          const uint64_t crc,
                                          const uint8_t* data,
                                          const size_t size) {
    if (!isPrepared()) return;
    std::shared_ptr<const std::vector<AmdCodeObjectFunction>> functions;
    {
        std::lock_guard lock(mutex_);
        if (const auto it = functions_by_crc_.find(crc);
            it != functions_by_crc_.end()) {
            functions = it->second;
        }
    }
    if (!functions) {
        functions = std::make_shared<const std::vector<AmdCodeObjectFunction>>(
            ParseAmdCodeObjectFunctions(data, size));
    }
    std::lock_guard lock(mutex_);
    functions_by_crc_.emplace(crc, functions);
    aggregator_.addCodeObject(code_object_id, crc, std::move(functions));
}

void PcSamplingEngine::bufferCallback(rocprofiler_context_id_t,
                                      rocprofiler_buffer_id_t,
                                      rocprofiler_record_header_t** headers,
                                      const size_t num_headers, void*,
                                      const uint64_t drop_count) {
    std::lock_guard lock(g_live_engine_mutex);
    if (g_live_engine) g_live_engine->consume(headers, num_headers, drop_count);
}

uint32_t PcSamplingEngine::bucketId(const bool issued, const uint32_t value) {
    const uint32_t key = (issued ? 0x80000000u : 0u) | value;
    if (const auto it = bucket_ids_.find(key); it != bucket_ids_.end()) {
        return it->second;
    }
    const char* sdk_name = nullptr;
    if (issued) {
        sdk_name = rocprofiler_get_pc_sampling_instruction_type_name(
            static_cast<rocprofiler_pc_sampling_instruction_type_t>(value));
    } else {
        using Reason = rocprofiler_pc_sampling_instruction_not_issued_reason_t;
        sdk_name =
            rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
                static_cast<Reason>(value));
    }
    const auto id = static_cast<uint32_t>(bucket_names_.size());
    bucket_names_.push_back(AmdPcSamplingBucketName(issued, sdk_name, value));
    bucket_ids_.emplace(key, id);
    return id;
}

void PcSamplingEngine::consume(rocprofiler_record_header_t** headers,
                               const size_t num_headers,
                               const uint64_t drop_count) {
    std::lock_guard lock(mutex_);
    dropped_records_ += drop_count;
    if (headers == nullptr) return;
    for (size_t i = 0; i < num_headers; ++i) {
        const rocprofiler_record_header_t* header = headers[i];
        if (header == nullptr ||
            header->category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING) {
            continue;
        }
        if (header->kind !=
                ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE ||
            header->payload == nullptr) {
            // Invalid samples, and anything that isn't a stochastic sample.
            ++invalid_records_;
            continue;
        }
        const auto* sample =
            static_cast<const rocprofiler_pc_sampling_record_stochastic_v0_t*>(
                header->payload);
        // The instruction type means something only for an issued sample,
        // the reason only for one that waited.
        const bool issued = sample->wave_issued != 0;
        const uint32_t bucket =
            bucketId(issued, issued ? sample->inst_type
                                    : sample->snapshot.reason_not_issued);
        // ROCPROFILER_CODE_OBJECT_ID_NONE matches no code object: unmatched.
        aggregator_.addSample(
            sample->pc.code_object_id, sample->pc.code_object_offset, issued,
            bucket,
            static_cast<uint32_t>(sample->correlation_id.internal &
                                  0xffffffffu));
    }
}

void PcSamplingEngine::pushRows(const bool final) {
    std::vector<AmdPcSampleRow> rows;
    std::vector<std::string> names;
    {
        std::lock_guard lock(mutex_);
        std::unordered_map<uint64_t, bool> ready;
        rows = aggregator_.takeRows([&](const uint64_t crc) {
            if (final) return true;
            const auto [it, inserted] = ready.try_emplace(crc, false);
            if (inserted) it->second = IsaParsed(crc);
            return it->second;
        });
        if (rows.empty()) return;
        names = bucket_names_;
    }

    const int64_t ts_ns = detail::GetTimestampNs();
    std::vector<ProfileSampleInput> samples;
    samples.reserve(std::min(rows.size(), kRowsPerPush));
    for (const AmdPcSampleRow& row : rows) {
        ProfileSampleInput sample;
        sample.ts_ns = ts_ns;
        sample.corr_id = row.corr_id;
        sample.device_id = device_id_;
        // PushProfileSamples keys the function by the symbol before '@'.
        sample.function_key = row.function + "@";
        sample.pc_offset = row.pc_offset;
        sample.metric_name = names[row.bucket];
        sample.metric_value = row.samples;
        sample.stall_reason = 0;  // AMD rows carry names, no CUPTI number
        sample.sample_kind = 0;   // pc_sampling
        DictionaryManager::IsaSourceLine line;
        if (Monitor::LookupAmdIsaSource(row.code_object_crc, row.function,
                                        row.pc_offset, &line) ==
            DictionaryManager::IsaSourceLookup::Found) {
            sample.function_source_file = line.function_source;
            sample.source_file = line.source_file;
            sample.source_line = line.source_line;
        }
        samples.push_back(std::move(sample));
        if (samples.size() >= kRowsPerPush) {
            Monitor::PushProfileSamples(samples);
            samples.clear();
        }
    }
    Monitor::PushProfileSamples(samples);
}

}  // namespace gpufl::amd
