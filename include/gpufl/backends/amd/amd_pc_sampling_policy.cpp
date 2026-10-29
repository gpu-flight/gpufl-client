#include "gpufl/backends/amd/amd_pc_sampling_policy.hpp"

#include <iterator>
#include <limits>
#include <unordered_set>

namespace gpufl::amd {
namespace {

constexpr std::string_view kInstructionTypePrefix =
    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_";
constexpr std::string_view kNotIssuedReasonPrefix =
    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_";

// ELF64 constants (System V gABI).
constexpr uint32_t kShtSymtab = 2;
constexpr uint32_t kShtDynsym = 11;
constexpr uint64_t kShfExecinstr = 0x4;
constexpr uint8_t kSttFunc = 2;
constexpr size_t kElf64HeaderSize = 64;
constexpr size_t kElf64SectionHeaderSize = 64;
constexpr size_t kElf64SymbolSize = 24;

// Little-endian field read; the caller checks the bounds.
template <typename T>
T ReadLe(const uint8_t* data) {
    T value = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        value |= static_cast<T>(static_cast<T>(data[i]) << (8 * i));
    }
    return value;
}

struct Section {
    uint32_t name = 0;
    uint32_t type = 0;
    uint64_t flags = 0;
    uint64_t addr = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t link = 0;
    uint64_t entsize = 0;
};

bool InBounds(const uint64_t offset, const uint64_t length,
              const size_t size) {
    return offset <= size && length <= size - offset;
}

// The NUL-terminated string at offset in a string table the caller has
// bounds-checked; empty when it runs past the table.
std::string_view StringAt(const uint8_t* data, const Section& table,
                          const uint64_t offset) {
    if (offset >= table.size) return {};
    const char* text =
        reinterpret_cast<const char*>(data + table.offset + offset);
    const uint64_t max_length = table.size - offset;
    size_t length = 0;
    while (length < max_length && text[length] != '\0') ++length;
    return length == max_length ? std::string_view()
                                : std::string_view(text, length);
}

}  // namespace

std::optional<uint64_t> ResolveAmdPcSamplingInterval(
    const uint64_t min_interval, const uint64_t max_interval) {
    if (max_interval == 0 || min_interval > max_interval) return std::nullopt;
    if (kAmdPcSamplingIntervalCycles < min_interval) {
        uint64_t interval = 1;
        while (interval < min_interval) {
            if (interval > max_interval / 2) return std::nullopt;
            interval <<= 1;
        }
        return interval;
    }
    if (kAmdPcSamplingIntervalCycles > max_interval) {
        uint64_t interval = 1;
        while (interval <= max_interval / 2) interval <<= 1;
        if (interval < min_interval) return std::nullopt;
        return interval;
    }
    return kAmdPcSamplingIntervalCycles;
}

std::vector<std::string_view> SplitAmdEngineGroup(
    const std::string_view engines, const char separator) {
    std::vector<std::string_view> names;
    size_t start = 0;
    while (start <= engines.size()) {
        const size_t end = engines.find(separator, start);
        std::string_view name = engines.substr(
            start, end == std::string_view::npos ? std::string_view::npos
                                                 : end - start);
        while (!name.empty() &&
               (name.front() == ' ' || name.front() == '\t')) {
            name.remove_prefix(1);
        }
        while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) {
            name.remove_suffix(1);
        }
        if (!name.empty()) names.push_back(name);
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return names;
}

AmdPcSamplingGroup ClassifyAmdPcSamplingGroup(const std::string_view engines,
                                              const char separator) {
    const auto names = SplitAmdEngineGroup(engines, separator);
    if (std::find(names.begin(), names.end(), "PcSampling") == names.end()) {
        return AmdPcSamplingGroup::Absent;
    }
    const bool beside_trace_only =
        std::all_of(names.begin(), names.end(), [](const std::string_view n) {
            return n == "PcSampling" || n == "Trace";
        });
    return beside_trace_only || names.front() == "PcSampling"
               ? AmdPcSamplingGroup::Samples
               : AmdPcSamplingGroup::Dropped;
}

std::string AmdPcSamplingBucketName(const bool issued, const char* sdk_name,
                                    const uint32_t value) {
    std::string bucket = issued ? "amd_pcs_issued_" : "amd_pcs_not_issued_";
    std::string_view name = sdk_name ? sdk_name : "";
    const std::string_view prefix =
        issued ? kInstructionTypePrefix : kNotIssuedReasonPrefix;
    if (name.substr(0, prefix.size()) == prefix) {
        name.remove_prefix(prefix.size());
    }
    if (name.empty()) return bucket + "unknown_" + std::to_string(value);
    for (const char c : name) {
        if (c >= 'A' && c <= 'Z') {
            bucket.push_back(static_cast<char>(c - 'A' + 'a'));
        } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            bucket.push_back(c);
        } else {
            bucket.push_back('_');
        }
    }
    return bucket;
}

const char* AmdPcSamplingRefusalReason(const AmdPcSamplingRefusal refusal) {
    switch (refusal) {
        case AmdPcSamplingRefusal::NotOptedIn:
            return "amd_pc_sampling_not_opted_in";
        case AmdPcSamplingRefusal::MethodUnavailable:
            return "amd_pc_sampling_method_unavailable";
        case AmdPcSamplingRefusal::DriverUnsupported:
            return "amd_pc_sampling_driver_unsupported";
        case AmdPcSamplingRefusal::Unavailable:
            return "amd_pc_sampling_unavailable";
        case AmdPcSamplingRefusal::ConfigurationFailed:
            return "amd_pc_sampling_configuration_failed";
        case AmdPcSamplingRefusal::WindowOnly:
            return "amd_pc_sampling_window_only_unsupported";
        case AmdPcSamplingRefusal::GroupUnsupported:
            return "amd_pc_sampling_group_unsupported";
    }
    return "amd_pc_sampling_configuration_failed";
}

AmdPcSamplingChoice ChooseAmdPcSamplingConfiguration(
    const std::vector<AmdPcSamplingListing>& listed) {
    AmdPcSamplingChoice choice;
    for (const AmdPcSamplingListing& listing : listed) {
        if (!listing.stochastic_cycles) continue;
        if (const auto interval = ResolveAmdPcSamplingInterval(
                listing.min_interval, listing.max_interval)) {
            choice.interval = *interval;
        } else {
            choice.detail = "no power-of-2 interval from " +
                            std::to_string(listing.min_interval) + " to " +
                            std::to_string(listing.max_interval) + " cycles";
        }
        return choice;
    }
    // A session in progress is all the agent lists, its interval as both
    // bounds; a GPU that has no session lists ranges.
    if (listed.size() == 1 && listed.front().min_interval != 0 &&
        listed.front().min_interval == listed.front().max_interval) {
        choice.refusal = AmdPcSamplingRefusal::Unavailable;
        choice.detail = "another process samples this GPU by another method";
        return choice;
    }
    choice.refusal = AmdPcSamplingRefusal::MethodUnavailable;
    return choice;
}

AmdPcSamplingRefusal ClassifyAmdPcSamplingStatus(
    const AmdPcSamplingCall call, const std::string_view status_name) {
    // rocprofiler-sdk 1.1.0 answers both calls NOT_IMPLEMENTED while its
    // ROCPROFILER_PC_SAMPLING_BETA_ENABLED switch is off.
    if (status_name == "ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED") {
        return AmdPcSamplingRefusal::NotOptedIn;
    }
    // KFD older than 1.16, or a PC sampling interface too old for the
    // method on this target.
    if (status_name == "ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL") {
        return AmdPcSamplingRefusal::DriverUnsupported;
    }
    // The capability query's ioctl failed: an amdgpu without AMD's PC
    // sampling ioctl (mainline's KFD 1.16 passes the version check).
    if (call == AmdPcSamplingCall::Query &&
        status_name == "ROCPROFILER_STATUS_ERROR") {
        return AmdPcSamplingRefusal::DriverUnsupported;
    }
    if (status_name == "ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE") {
        return AmdPcSamplingRefusal::Unavailable;
    }
    return AmdPcSamplingRefusal::ConfigurationFailed;
}

AmdCodeObject ParseAmdCodeObject(const uint8_t* data, const size_t size) {
    AmdCodeObject code_object;
    if (data == nullptr || size < kElf64HeaderSize || data[0] != 0x7f ||
        data[1] != 'E' || data[2] != 'L' || data[3] != 'F' ||
        data[4] != 2 /* ELFCLASS64 */ || data[5] != 1 /* ELFDATA2LSB */) {
        return code_object;
    }
    const auto section_offset = ReadLe<uint64_t>(data + 0x28);
    const auto section_entry_size = ReadLe<uint16_t>(data + 0x3a);
    const auto section_count = ReadLe<uint16_t>(data + 0x3c);
    const auto section_names = ReadLe<uint16_t>(data + 0x3e);
    if (section_entry_size < kElf64SectionHeaderSize ||
        !InBounds(section_offset,
                  uint64_t{section_entry_size} * section_count, size)) {
        return code_object;
    }

    std::vector<Section> sections(section_count);
    for (size_t i = 0; i < section_count; ++i) {
        const uint8_t* header = data + section_offset + i * section_entry_size;
        Section& section = sections[i];
        section.name = ReadLe<uint32_t>(header);
        section.type = ReadLe<uint32_t>(header + 4);
        section.flags = ReadLe<uint64_t>(header + 8);
        section.addr = ReadLe<uint64_t>(header + 16);
        section.offset = ReadLe<uint64_t>(header + 24);
        section.size = ReadLe<uint64_t>(header + 32);
        section.link = ReadLe<uint32_t>(header + 40);
        section.entsize = ReadLe<uint64_t>(header + 56);
    }

    if (section_names < sections.size() &&
        InBounds(sections[section_names].offset, sections[section_names].size,
                 size)) {
        for (const Section& section : sections) {
            if (StringAt(data, sections[section_names], section.name) ==
                ".debug_line") {
                code_object.line_info = true;
                break;
            }
        }
    }

    // The full symbol table names local device functions too; a stripped
    // object still has the dynamic one.
    const Section* symbols = nullptr;
    for (const uint32_t type : {kShtSymtab, kShtDynsym}) {
        for (const Section& section : sections) {
            if (section.type == type) {
                symbols = &section;
                break;
            }
        }
        if (symbols) break;
    }
    if (symbols == nullptr || symbols->entsize < kElf64SymbolSize ||
        symbols->link >= sections.size() ||
        !InBounds(symbols->offset, symbols->size, size)) {
        return code_object;
    }
    const Section& strings = sections[symbols->link];
    if (!InBounds(strings.offset, strings.size, size)) return code_object;

    struct Candidate {
        AmdCodeObjectFunction function;
        uint16_t section = 0;
    };
    std::vector<Candidate> candidates;
    // Counted, not stepped: a huge sh_entsize can't wrap the offset.
    const uint64_t symbol_count = symbols->size / symbols->entsize;
    for (uint64_t i = 0; i < symbol_count; ++i) {
        const uint8_t* symbol = data + symbols->offset + i * symbols->entsize;
        const auto name_offset = ReadLe<uint32_t>(symbol);
        const uint8_t info = symbol[4];
        const auto section_index = ReadLe<uint16_t>(symbol + 6);
        const auto value = ReadLe<uint64_t>(symbol + 8);
        const auto symbol_size = ReadLe<uint64_t>(symbol + 16);
        if ((info & 0xf) != kSttFunc || section_index >= sections.size() ||
            (sections[section_index].flags & kShfExecinstr) == 0) {
            continue;
        }
        const std::string_view name = StringAt(data, strings, name_offset);
        if (name.empty()) continue;
        candidates.push_back(
            {{value, symbol_size, std::string(name)}, section_index});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return std::tie(a.function.vaddr, a.function.name) <
                         std::tie(b.function.vaddr, b.function.name);
              });
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                 [](const Candidate& a, const Candidate& b) {
                                     return a.function.vaddr ==
                                                b.function.vaddr &&
                                            a.function.name == b.function.name;
                                 }),
                     candidates.end());

    // A symbol without a size runs to the next function's start, or to the
    // end of its section. An alias of a sized symbol takes its size.
    for (size_t first = 0; first < candidates.size();) {
        const uint64_t vaddr = candidates[first].function.vaddr;
        size_t next = first;
        uint64_t sized = 0;
        while (next < candidates.size() &&
               candidates[next].function.vaddr == vaddr) {
            sized = std::max(sized, candidates[next].function.size);
            ++next;
        }
        for (size_t i = first; i < next; ++i) {
            AmdCodeObjectFunction& function = candidates[i].function;
            if (function.size != 0) continue;
            if (sized != 0) {
                function.size = sized;
                continue;
            }
            const Section& section = sections[candidates[i].section];
            uint64_t end =
                section.size > std::numeric_limits<uint64_t>::max() -
                                   section.addr
                    ? std::numeric_limits<uint64_t>::max()
                    : section.addr + section.size;
            if (next < candidates.size()) {
                end = std::min(end, candidates[next].function.vaddr);
            }
            if (vaddr >= section.addr && end > vaddr) {
                function.size = end - vaddr;
            }
        }
        first = next;
    }

    for (Candidate& candidate : candidates) {
        if (candidate.function.size == 0) continue;
        code_object.functions.push_back(std::move(candidate.function));
    }
    return code_object;
}

std::optional<AmdPcSampleLocation> LocateAmdPcSample(
    const std::vector<AmdCodeObjectFunction>& functions,
    const uint64_t code_object_offset) {
    auto next = std::upper_bound(
        functions.begin(), functions.end(), code_object_offset,
        [](const uint64_t offset, const AmdCodeObjectFunction& function) {
            return offset < function.vaddr;
        });
    if (next == functions.begin()) return std::nullopt;
    const AmdCodeObjectFunction& function = *std::prev(next);
    const uint64_t pc_offset = code_object_offset - function.vaddr;
    if (pc_offset >= function.size ||
        pc_offset > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return AmdPcSampleLocation{&function, static_cast<uint32_t>(pc_offset)};
}

std::vector<uint64_t> AmdPcSampleAggregator::waitingCodeObjects() const {
    std::unordered_set<uint64_t> crcs;
    for (const auto& [key, value] : rows_) {
        crcs.insert(code_objects_.at(key.code_object_id).crc);
    }
    return {crcs.begin(), crcs.end()};
}

void AmdPcSampleAggregator::addCodeObject(
    const uint64_t code_object_id, const uint64_t crc,
    std::shared_ptr<const std::vector<AmdCodeObjectFunction>> functions) {
    code_objects_[code_object_id] = CodeObject{crc, std::move(functions)};
}

void AmdPcSampleAggregator::addSample(const uint64_t code_object_id,
                                      const uint64_t code_object_offset,
                                      const bool issued, const uint32_t bucket,
                                      const uint32_t corr_id) {
    const auto code_object = code_objects_.find(code_object_id);
    if (code_object == code_objects_.end() ||
        !code_object->second.functions) {
        ++totals_.unmatched;
        return;
    }
    const auto& functions = *code_object->second.functions;
    const auto location = LocateAmdPcSample(functions, code_object_offset);
    if (!location) {
        ++totals_.unmatched;
        return;
    }
    RowKey key;
    key.code_object_id = code_object_id;
    key.function =
        static_cast<uint32_t>(location->function - functions.data());
    key.pc_offset = location->pc_offset;
    key.bucket = bucket;
    RowValue& row = rows_[key];
    ++row.samples;
    row.corr_id = corr_id;
    ++totals_.samples;
    ++(issued ? totals_.issued : totals_.not_issued);
}

}  // namespace gpufl::amd
