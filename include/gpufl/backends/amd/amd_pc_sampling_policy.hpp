#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

// Decisions for AMD stochastic PC sampling that need no ROCprofiler SDK, so
// CPU tests cover them. The engine (engine/pc_sampling_engine.*) feeds them
// what rocprofiler-sdk returns.
namespace gpufl::amd {

// The pc_sampling capability row's mode when stochastic sampling runs.
inline constexpr const char* kAmdPcSamplingModeWireName =
    "rocprofiler_pc_sampling_stochastic";

// Every run samples every 2^20 cycles, clamped to what the agent lists. Not
// settable per run: it is rocprofv3's own example and the longest interval a
// public MI300X VF report listed.
inline constexpr uint64_t kAmdPcSamplingIntervalCycles = uint64_t{1} << 20;

// The power of 2 closest to kAmdPcSamplingIntervalCycles within
// [min_interval, max_interval], or nullopt when the range holds none.
std::optional<uint64_t> ResolveAmdPcSamplingInterval(uint64_t min_interval,
                                                     uint64_t max_interval);

// The engines of a pass's group, written as the launcher's
// "Trace+PcSampling" (separator '+') or GPUFL_ENGINE_COMBO's
// "Trace,PcSampling" (','): each name trimmed of spaces and tabs, empty
// names left out.
std::vector<std::string_view> SplitAmdEngineGroup(std::string_view engines,
                                                  char separator);

// How an AMD process treats PcSampling in a pass's engine group. The AMD
// backend runs one service beside tracing, and PC sampling can't share
// rocprofiler-sdk's context with counters: a group of PcSampling and Trace
// samples, and any other group keeps its first engine, so it samples only
// when PcSampling comes first. The launcher and the AMD backend both ask
// this, so the opt-in reaches exactly the processes that sample.
enum class AmdPcSamplingGroup {
    Absent,   // no PcSampling in the group
    Samples,  // the process samples PCs, once opted in
    Dropped,  // PcSampling is named, but another engine runs instead
};

AmdPcSamplingGroup ClassifyAmdPcSamplingGroup(std::string_view engines,
                                              char separator);

// A sample's bucket: amd_pcs_issued_<type> when the wave issued the sampled
// instruction, amd_pcs_not_issued_<reason> when it did not. sdk_name is what
// rocprofiler_get_pc_sampling_instruction_type_name or ..._not_issued_reason_
// name returned for the value; its ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_
// or ..._NOT_ISSUED_REASON_ prefix is dropped and the rest lower-cased. A
// value the SDK can't name becomes ..._unknown_<value>.
std::string AmdPcSamplingBucketName(bool issued, const char* sdk_name,
                                    uint32_t value);

// Why PC sampling did not run, as the pc_sampling row's reason code.
enum class AmdPcSamplingRefusal {
    // No --amd-pc-sampling=stochastic, or rocprofiler-sdk's beta switch off.
    NotOptedIn,
    // The agent lists no stochastic configuration (gfx1201, gfx90a).
    MethodUnavailable,
    // The driver has no PC sampling interface (an in-box amdgpu).
    DriverUnsupported,
    // Another process holds a different configuration, or ROCgdb runs.
    Unavailable,
    // Any other refusal.
    ConfigurationFailed,
    // Window-only deep arming (GPUFL_DEEP_ARM=window): rocprofiler-sdk
    // samples for as long as the context runs, so no window bounds it.
    WindowOnly,
    // The pass's '+' group also named an engine PC sampling can't share a
    // context with, and that engine ran (AmdPcSamplingGroup::Dropped).
    GroupUnsupported,
};

const char* AmdPcSamplingRefusalReason(AmdPcSamplingRefusal refusal);

// One configuration rocprofiler_query_pc_sampling_agent_configurations
// listed. While a process samples the GPU, the agent lists only that
// session's configuration, its interval as both bounds.
struct AmdPcSamplingListing {
    bool stochastic_cycles = false;  // the stochastic method, in cycles
    uint64_t min_interval = 0;
    uint64_t max_interval = 0;
};

struct AmdPcSamplingChoice {
    uint64_t interval = 0;  // cycles; 0 when refused
    AmdPcSamplingRefusal refusal = AmdPcSamplingRefusal::ConfigurationFailed;
    std::string detail;
};

// The stochastic interval to configure from what the agent lists, or why
// there is none. Another process's stochastic session is joined at its
// interval; another process's host-trap session makes the GPU unavailable,
// which is not the same as a GPU without stochastic sampling.
AmdPcSamplingChoice ChooseAmdPcSamplingConfiguration(
    const std::vector<AmdPcSamplingListing>& listed);

enum class AmdPcSamplingCall { Query, Configure };

// Classifies a refused rocprofiler_query_pc_sampling_agent_configurations
// or rocprofiler_configure_pc_sampling_service by the status's name
// (rocprofiler_get_status_name), so the rule needs no SDK headers.
AmdPcSamplingRefusal ClassifyAmdPcSamplingStatus(AmdPcSamplingCall call,
                                                 std::string_view status_name);

// A function in a loaded code object: an ELF STT_FUNC symbol of an
// executable section. Kernels and the device functions they call both
// count; the ISA rows label instructions by the same symbols.
struct AmdCodeObjectFunction {
    uint64_t vaddr = 0;
    uint64_t size = 0;
    std::string name;
};

// What PC sampling reads from a loaded AMDGPU code object (ELF64).
struct AmdCodeObject {
    // Sorted by address. A symbol without a size, as Tensile's assembly
    // kernels in rocBLAS and hipBLASLt have, extends to the next function
    // or to the end of its section.
    std::vector<AmdCodeObjectFunction> functions;
    // A .debug_line section, from which llvm-objdump -l gives the ISA rows
    // their source lines. Without one, no row has a line to wait for.
    bool line_info = false;
};

// Empty for anything that isn't a readable ELF64 file.
AmdCodeObject ParseAmdCodeObject(const uint8_t* data, size_t size);

struct AmdPcSampleLocation {
    const AmdCodeObjectFunction* function = nullptr;
    // Bytes from the function's first instruction, as the ISA rows count.
    uint32_t pc_offset = 0;
};

// rocprofiler-sdk reports a sample as its code object and the offset from
// that object's load base. AMDGPU code objects load their first segment at
// virtual address 0, so the offset is the ELF virtual address, as AMD's own
// rocprofv3 decoder reads it. nullopt when no function contains it.
std::optional<AmdPcSampleLocation> LocateAmdPcSample(
    const std::vector<AmdCodeObjectFunction>& functions,
    uint64_t code_object_offset);

// Totals since the session began.
struct AmdPcSamplingTotals {
    uint64_t samples = 0;     // attributed to a function: issued + waiting
    uint64_t issued = 0;
    uint64_t not_issued = 0;
    uint64_t unmatched = 0;   // unknown code object, or outside every function
};

// One row: a function, an offset in it and a bucket, with the samples it
// collected since the last take.
struct AmdPcSampleRow {
    uint64_t code_object_crc = 0;
    std::string function;
    uint32_t pc_offset = 0;
    uint32_t bucket = 0;    // the caller's bucket id, see addSample
    uint64_t samples = 0;
    uint32_t corr_id = 0;   // the latest launch sampled at this row
};

// Aggregates samples per code object, function, offset and bucket. Not
// thread-safe; the engine serializes it.
class AmdPcSampleAggregator {
   public:
    void addCodeObject(
        uint64_t code_object_id, uint64_t crc,
        std::shared_ptr<const std::vector<AmdCodeObjectFunction>> functions);

    // bucket is the caller's id for the sample's bucket name; rows carry it
    // back. Counting by id keeps string work off the per-sample path.
    void addSample(uint64_t code_object_id, uint64_t code_object_offset,
                   bool issued, uint32_t bucket, uint32_t corr_id);

    // Removes and returns, in a stable order, the rows of code objects for
    // which ready(crc) is true. Other rows stay for a later take.
    template <typename Ready>
    std::vector<AmdPcSampleRow> takeRows(Ready&& ready);

    // The code objects (by crc) that rows are waiting in.
    std::vector<uint64_t> waitingCodeObjects() const;

    const AmdPcSamplingTotals& totals() const { return totals_; }

   private:
    struct CodeObject {
        uint64_t crc = 0;
        std::shared_ptr<const std::vector<AmdCodeObjectFunction>> functions;
    };
    struct RowKey {
        uint64_t code_object_id = 0;
        uint32_t function = 0;  // index into the code object's functions
        uint32_t pc_offset = 0;
        uint32_t bucket = 0;
        bool operator==(const RowKey& other) const {
            return code_object_id == other.code_object_id &&
                   function == other.function &&
                   pc_offset == other.pc_offset && bucket == other.bucket;
        }
    };
    struct RowKeyHash {
        size_t operator()(const RowKey& key) const {
            uint64_t h = key.code_object_id * 0x9e3779b97f4a7c15ULL;
            h ^= (uint64_t{key.function} << 32 | key.pc_offset) +
                 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= uint64_t{key.bucket} + 0x9e3779b97f4a7c15ULL + (h << 6) +
                 (h >> 2);
            return static_cast<size_t>(h);
        }
    };
    struct RowValue {
        uint64_t samples = 0;
        uint32_t corr_id = 0;
    };

    std::unordered_map<uint64_t, CodeObject> code_objects_;
    std::unordered_map<RowKey, RowValue, RowKeyHash> rows_;
    AmdPcSamplingTotals totals_;
};

template <typename Ready>
std::vector<AmdPcSampleRow> AmdPcSampleAggregator::takeRows(Ready&& ready) {
    std::vector<AmdPcSampleRow> out;
    for (auto it = rows_.begin(); it != rows_.end();) {
        const CodeObject& code_object =
            code_objects_.at(it->first.code_object_id);
        if (!ready(code_object.crc)) {
            ++it;
            continue;
        }
        AmdPcSampleRow row;
        row.code_object_crc = code_object.crc;
        row.function = (*code_object.functions)[it->first.function].name;
        row.pc_offset = it->first.pc_offset;
        row.bucket = it->first.bucket;
        row.samples = it->second.samples;
        row.corr_id = it->second.corr_id;
        out.push_back(std::move(row));
        it = rows_.erase(it);
    }
    std::sort(out.begin(), out.end(),
              [](const AmdPcSampleRow& a, const AmdPcSampleRow& b) {
                  return std::tie(a.code_object_crc, a.function, a.pc_offset,
                                  a.bucket) < std::tie(b.code_object_crc,
                                                       b.function, b.pc_offset,
                                                       b.bucket);
              });
    return out;
}

}  // namespace gpufl::amd
