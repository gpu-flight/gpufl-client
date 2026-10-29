#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>

#include "gpufl/backends/amd/amd_pc_sampling_policy.hpp"
#include "gpufl/backends/amd/engine/amd_profiling_engine.hpp"

namespace gpufl::amd {

/// Stochastic PC sampling through rocprofiler-sdk's experimental, beta PC
/// sampling service. Configured once, in toolInitialize, on the primary GPU
/// agent, and only after the run opted in; host-trap is never configured.
/// Samples are aggregated per function, offset and bucket, and pushed as
/// profile_sample_batch rows (sample_kind 0, stall_reason 0) at each drain
/// and stop. A row whose code object has DWARF lines waits for that object's
/// ISA rows, which the collector disassembles as soon as it loads, so the
/// two share a function and a source line; the rest go out at once. The
/// engine never disassembles: that stays on the collector thread.
class PcSamplingEngine final : public AmdProfilingEngine {
   public:
    PcSamplingEngine() = default;
    ~PcSamplingEngine() override;

    bool initialize(rocprofiler_context_id_t context,
                    rocprofiler_agent_id_t gpu_agent,
                    uint32_t gpu_device_id,
                    const MonitorOptions& opts) override;
    void start() override;
    void stop() override;
    void drain() override;
    void finish() override;
    /// Delivers the last samples and pushes every row; afterwards the
    /// engine makes no rocprofiler-sdk call, which may be finalizing. Can
    /// wait up to two seconds for line tables, so the backend writes the
    /// capability row first.
    void shutdown() override;
    bool hasData() const override;
    bool isPrepared() const override {
        return configured_.load(std::memory_order_acquire) &&
               !finalized_.load(std::memory_order_acquire);
    }
    bool isArmed() const override {
        return isPrepared() && started_.load(std::memory_order_acquire);
    }
    void onCodeObjectLoaded(uint64_t code_object_id, uint64_t crc,
                            const uint8_t* data, size_t size) override;

    /// Why initialize() returned false, and what rocprofiler-sdk answered.
    AmdPcSamplingRefusal refusal() const { return refusal_; }
    const std::string& refusalDetail() const { return refusal_detail_; }

    struct Stats {
        uint64_t interval = 0;  // cycles
        AmdPcSamplingTotals totals;
        // Invalid records, records rocprofiler-sdk dropped, and samples that
        // matched no function.
        uint64_t lost = 0;
    };
    Stats stats() const;

   private:
    static void bufferCallback(rocprofiler_context_id_t context,
                               rocprofiler_buffer_id_t buffer_id,
                               rocprofiler_record_header_t** headers,
                               size_t num_headers, void* data,
                               uint64_t drop_count);
    void consume(rocprofiler_record_header_t** headers, size_t num_headers,
                 uint64_t drop_count);
    uint32_t bucketId(bool issued, uint32_t value);
    // False once shutdown() has run, and then no SDK call is made.
    bool flush();
    // Whether a code object's rows may go now: it has no line table to
    // wait for, or its ISA rows exist. mutex_ held.
    bool rowsReady(uint64_t crc) const;
    void pushRows(bool final);
    bool refuse(AmdPcSamplingRefusal refusal, std::string detail);

    rocprofiler_context_id_t context_{};
    rocprofiler_agent_id_t agent_{};
    rocprofiler_buffer_id_t buffer_{};
    uint32_t device_id_ = 0;
    uint64_t interval_ = 0;
    AmdPcSamplingRefusal refusal_ = AmdPcSamplingRefusal::ConfigurationFailed;
    std::string refusal_detail_;
    std::atomic<bool> configured_{false};
    std::atomic<bool> started_{false};

    // Serializes buffer flushes with shutdown(): the collector's drain must
    // not flush while, or after, rocprofiler-sdk finalizes.
    std::mutex sdk_mutex_;
    std::atomic<bool> finalized_{false};

    mutable std::mutex mutex_;  // guards everything below
    AmdPcSampleAggregator aggregator_;
    std::unordered_map<uint32_t, uint32_t> bucket_ids_;
    std::vector<std::string> bucket_names_;
    std::unordered_map<uint64_t, std::shared_ptr<const AmdCodeObject>>
        code_objects_by_crc_;
    uint64_t invalid_records_ = 0;
    uint64_t dropped_records_ = 0;
};

}  // namespace gpufl::amd
