#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpufl/backends/amd/amd_capture_capabilities.hpp"
#include "gpufl/backends/amd/amd_pc_sampling_policy.hpp"
#include "gpufl/backends/amd/amd_profiling_policy.hpp"

namespace {

using gpufl::amd::AmdCodeObjectFunction;
using gpufl::amd::AmdPcSampleAggregator;
using gpufl::amd::AmdPcSamplingCall;
using gpufl::amd::AmdPcSamplingGroup;
using gpufl::amd::AmdPcSamplingListing;
using gpufl::amd::AmdPcSamplingRefusal;

// tests/backends/amd/data/pc_sampling_fixture.hip, built for gfx942 by
// ROCm 7.2.0's clang (the command is in the source). Two kernels and the
// non-inlined device function one of them calls. The .lines copy has DWARF
// line tables; pc_sampling_unsized.s is two kernels written as Tensile
// writes them, without symbol sizes.
std::vector<uint8_t> ReadFixture(
    const std::string& name = "pc_sampling_fixture.gfx942.co") {
    std::ifstream in(std::string(GPUFL_AMD_TEST_DATA_DIR) + "/" + name,
                     std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>());
}

uint64_t ReadLe(const std::vector<uint8_t>& bytes, const size_t at,
                const size_t width) {
    uint64_t value = 0;
    for (size_t i = width; i-- > 0;) value = (value << 8) | bytes[at + i];
    return value;
}

void WriteLe(std::vector<uint8_t>& bytes, const size_t at, uint64_t value,
             const size_t width) {
    for (size_t i = 0; i < width; ++i, value >>= 8) {
        bytes[at + i] = static_cast<uint8_t>(value);
    }
}

// The file offset of the first section header of an ELF64 type.
size_t SectionHeader(const std::vector<uint8_t>& elf, const uint32_t type) {
    const size_t table = ReadLe(elf, 0x28, 8);
    const size_t entry = ReadLe(elf, 0x3a, 2);
    const size_t count = ReadLe(elf, 0x3c, 2);
    for (size_t i = 0; i < count; ++i) {
        if (ReadLe(elf, table + i * entry + 4, 4) == type) {
            return table + i * entry;
        }
    }
    return 0;
}

constexpr uint64_t kVectorAdd = 0x1f00;
constexpr uint64_t kScaleAndBias = 0x1f8c;
constexpr uint64_t kScaled = 0x2000;

// Each function's ISA rows, as llvm-objdump -d labels them: the offsets
// from the function's first instruction, padding after s_endpgm left out.
const std::vector<uint32_t> kVectorAddRows = {
    0x0,  0x8,  0x10, 0x14, 0x1c, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x38, 0x40,
    0x44, 0x4c, 0x50, 0x58, 0x60, 0x68, 0x70, 0x78, 0x7c, 0x80, 0x88};
const std::vector<uint32_t> kScaleAndBiasRows = {0x0, 0x4, 0xc};
const std::vector<uint32_t> kScaledRows = {
    0x0,  0x8,  0x10, 0x14, 0x18, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x34, 0x3c,
    0x40, 0x48, 0x4c, 0x54, 0x5c, 0x60, 0x68, 0x70, 0x74, 0x7c, 0x80, 0x88};

std::shared_ptr<const std::vector<AmdCodeObjectFunction>> FixtureFunctions(
    const std::string& name = "pc_sampling_fixture.gfx942.co") {
    const auto bytes = ReadFixture(name);
    return std::make_shared<const std::vector<AmdCodeObjectFunction>>(
        gpufl::amd::ParseAmdCodeObject(bytes.data(), bytes.size()).functions);
}

gpufl::amd::AmdCaptureCapabilityInput PcSamplingInput(
    const gpufl::amd::AmdProfilingSupport& support) {
    gpufl::amd::AmdCaptureCapabilityInput input;
    input.plan = gpufl::amd::ResolveAmdProfilingPlan(
        gpufl::ProfilingEngine::PcSampling, support);
    input.trace_configured = true;
    return input;
}

const gpufl::CaptureCapability* PcSamplingRow(
    const gpufl::CaptureCapabilitiesEvent& event) {
    for (const auto& capability : event.capabilities) {
        if (capability.feature == "pc_sampling") return &capability;
    }
    return nullptr;
}

gpufl::CaptureCapability Refused(const AmdPcSamplingRefusal refusal,
                                 const std::string& detail = "") {
    gpufl::amd::AmdProfilingSupport support;
    support.pc_sampling_refusal = refusal;
    auto input = PcSamplingInput(support);
    input.pc_sampling_detail = detail;
    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    const auto* row = PcSamplingRow(event);
    EXPECT_NE(row, nullptr);
    return row ? *row : gpufl::CaptureCapability{};
}

gpufl::CaptureCapability Sampled(const uint64_t issued,
                                 const uint64_t not_issued,
                                 const uint64_t lost) {
    gpufl::amd::AmdProfilingSupport support;
    support.pc_sampling = true;
    auto input = PcSamplingInput(support);
    input.pc_sampling_interval = uint64_t{1} << 20;
    input.pc_samples = issued + not_issued;
    input.pc_samples_issued = issued;
    input.pc_samples_not_issued = not_issued;
    input.pc_samples_lost = lost;
    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    const auto* row = PcSamplingRow(event);
    EXPECT_NE(row, nullptr);
    return row ? *row : gpufl::CaptureCapability{};
}

}  // namespace

TEST(AmdPcSamplingInterval, StaysAtTwoToTheTwentiethWhenListed) {
    using gpufl::amd::ResolveAmdPcSamplingInterval;
    // AMD's driver lists 256 to 2^31 cycles for stochastic sampling.
    EXPECT_EQ(ResolveAmdPcSamplingInterval(256, uint64_t{1} << 31),
              uint64_t{1} << 20);
    // The MI300X VF in ROCm/rocm-systems#13505 listed up to 2^20.
    EXPECT_EQ(ResolveAmdPcSamplingInterval(256, uint64_t{1} << 20),
              uint64_t{1} << 20);
}

TEST(AmdPcSamplingInterval, ClampsToTheListedRangeAsAPowerOfTwo) {
    using gpufl::amd::ResolveAmdPcSamplingInterval;
    EXPECT_EQ(ResolveAmdPcSamplingInterval(256, 1 << 16), 1u << 16);
    EXPECT_EQ(ResolveAmdPcSamplingInterval(256, 100000), 65536u);
    EXPECT_EQ(ResolveAmdPcSamplingInterval(1 << 21, uint64_t{1} << 31),
              1u << 21);
    EXPECT_EQ(ResolveAmdPcSamplingInterval(3000000, uint64_t{1} << 31),
              1u << 22);
    // Another process's configuration lists as its single interval.
    EXPECT_EQ(ResolveAmdPcSamplingInterval(1 << 18, 1 << 18), 1u << 18);

    EXPECT_FALSE(ResolveAmdPcSamplingInterval((1 << 18) + 1, (1 << 18) + 5));
    EXPECT_FALSE(ResolveAmdPcSamplingInterval(4096, 256));
    EXPECT_FALSE(ResolveAmdPcSamplingInterval(0, 0));
}

// The names rocprofiler-sdk 1.1.0's name functions return (strings in
// /opt/rocm-7.2.0/lib/librocprofiler-sdk.so.1.1.0).
TEST(AmdPcSamplingBuckets, NameEveryNotIssuedReasonOfTheSdk) {
    const std::vector<std::pair<const char*, const char*>> reasons = {
        {"NONE", "amd_pcs_not_issued_none"},
        {"NO_INSTRUCTION_AVAILABLE",
         "amd_pcs_not_issued_no_instruction_available"},
        {"ALU_DEPENDENCY", "amd_pcs_not_issued_alu_dependency"},
        {"WAITCNT", "amd_pcs_not_issued_waitcnt"},
        {"INTERNAL_INSTRUCTION", "amd_pcs_not_issued_internal_instruction"},
        {"BARRIER_WAIT", "amd_pcs_not_issued_barrier_wait"},
        {"ARBITER_NOT_WIN", "amd_pcs_not_issued_arbiter_not_win"},
        {"ARBITER_WIN_EX_STALL", "amd_pcs_not_issued_arbiter_win_ex_stall"},
        {"OTHER_WAIT", "amd_pcs_not_issued_other_wait"},
        {"SLEEP_WAIT", "amd_pcs_not_issued_sleep_wait"},
    };
    uint32_t value = 0;
    for (const auto& [reason, bucket] : reasons) {
        const std::string sdk_name =
            std::string("ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_"
                        "REASON_") +
            reason;
        EXPECT_EQ(gpufl::amd::AmdPcSamplingBucketName(false, sdk_name.c_str(),
                                                      value++),
                  bucket);
    }
}

TEST(AmdPcSamplingBuckets, NameEveryIssuedInstructionTypeOfTheSdk) {
    const std::vector<std::pair<const char*, const char*>> types = {
        {"NONE", "amd_pcs_issued_none"},
        {"VALU", "amd_pcs_issued_valu"},
        {"MATRIX", "amd_pcs_issued_matrix"},
        {"SCALAR", "amd_pcs_issued_scalar"},
        {"TEX", "amd_pcs_issued_tex"},
        {"LDS", "amd_pcs_issued_lds"},
        {"LDS_DIRECT", "amd_pcs_issued_lds_direct"},
        {"FLAT", "amd_pcs_issued_flat"},
        {"EXPORT", "amd_pcs_issued_export"},
        {"MESSAGE", "amd_pcs_issued_message"},
        {"BARRIER", "amd_pcs_issued_barrier"},
        {"BRANCH_NOT_TAKEN", "amd_pcs_issued_branch_not_taken"},
        {"BRANCH_TAKEN", "amd_pcs_issued_branch_taken"},
        {"JUMP", "amd_pcs_issued_jump"},
        {"OTHER", "amd_pcs_issued_other"},
        {"NO_INST", "amd_pcs_issued_no_inst"},
        {"DUAL_VALU", "amd_pcs_issued_dual_valu"},
    };
    uint32_t value = 0;
    for (const auto& [type, bucket] : types) {
        const std::string sdk_name =
            std::string("ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_") + type;
        EXPECT_EQ(gpufl::amd::AmdPcSamplingBucketName(true, sdk_name.c_str(),
                                                      value++),
                  bucket);
    }
}

TEST(AmdPcSamplingBuckets, KeepAValueTheSdkCannotName) {
    EXPECT_EQ(gpufl::amd::AmdPcSamplingBucketName(false, nullptr, 12),
              "amd_pcs_not_issued_unknown_12");
    EXPECT_EQ(gpufl::amd::AmdPcSamplingBucketName(true, "", 31),
              "amd_pcs_issued_unknown_31");
    // A later SDK's name keeps its words even under another prefix.
    EXPECT_EQ(gpufl::amd::AmdPcSamplingBucketName(true, "Vector-ALU", 1),
              "amd_pcs_issued_vector_alu");
}

TEST(AmdPcSamplingStatus, ClassifiesEachRefusal) {
    using gpufl::amd::ClassifyAmdPcSamplingStatus;
    const auto query = AmdPcSamplingCall::Query;
    const auto configure = AmdPcSamplingCall::Configure;
    EXPECT_EQ(ClassifyAmdPcSamplingStatus(
                  query, "ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED"),
              AmdPcSamplingRefusal::NotOptedIn);
    EXPECT_EQ(ClassifyAmdPcSamplingStatus(
                  query, "ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL"),
              AmdPcSamplingRefusal::DriverUnsupported);
    // The capability ioctl failed: an in-box amdgpu without AMD's ioctl.
    EXPECT_EQ(ClassifyAmdPcSamplingStatus(query, "ROCPROFILER_STATUS_ERROR"),
              AmdPcSamplingRefusal::DriverUnsupported);
    EXPECT_EQ(ClassifyAmdPcSamplingStatus(
                  query, "ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE"),
              AmdPcSamplingRefusal::Unavailable);
    EXPECT_EQ(ClassifyAmdPcSamplingStatus(
                  configure, "ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE"),
              AmdPcSamplingRefusal::Unavailable);
    EXPECT_EQ(ClassifyAmdPcSamplingStatus(configure,
                                          "ROCPROFILER_STATUS_ERROR"),
              AmdPcSamplingRefusal::ConfigurationFailed);
    EXPECT_EQ(ClassifyAmdPcSamplingStatus(
                  configure, "ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT"),
              AmdPcSamplingRefusal::ConfigurationFailed);
}

TEST(AmdPcSamplingConfiguration, TakesTheStochasticSamplerInCycles) {
    using gpufl::amd::ChooseAmdPcSamplingConfiguration;
    // What AMD's driver lists for an MI300-series GPU nobody samples.
    const auto idle = ChooseAmdPcSamplingConfiguration(
        {{false, 1, 1000000}, {true, 256, uint64_t{1} << 31}});
    EXPECT_EQ(idle.interval, uint64_t{1} << 20);

    // Another process's stochastic session lists its interval as both
    // bounds; the run joins it.
    const auto joined =
        ChooseAmdPcSamplingConfiguration({{true, 1 << 18, 1 << 18}});
    EXPECT_EQ(joined.interval, 1u << 18);

    const auto odd = ChooseAmdPcSamplingConfiguration(
        {{true, (1 << 18) + 1, (1 << 18) + 5}});
    EXPECT_EQ(odd.interval, 0u);
    EXPECT_EQ(odd.refusal, AmdPcSamplingRefusal::ConfigurationFailed);
    EXPECT_EQ(odd.detail,
              "no power-of-2 interval from 262145 to 262149 cycles");
}

TEST(AmdPcSamplingConfiguration, TellsABusyGpuFromOneWithoutTheMethod) {
    using gpufl::amd::ChooseAmdPcSamplingConfiguration;
    // Host-trap only (gfx90a), or nothing at all.
    EXPECT_EQ(ChooseAmdPcSamplingConfiguration({{false, 1, 1000000}}).refusal,
              AmdPcSamplingRefusal::MethodUnavailable);
    EXPECT_EQ(ChooseAmdPcSamplingConfiguration({}).refusal,
              AmdPcSamplingRefusal::MethodUnavailable);

    // An MI325X another process samples by host-trap lists only that
    // session: the GPU is busy, not lacking stochastic sampling.
    const auto busy = ChooseAmdPcSamplingConfiguration({{false, 100, 100}});
    EXPECT_EQ(busy.interval, 0u);
    EXPECT_EQ(busy.refusal, AmdPcSamplingRefusal::Unavailable);
}

TEST(AmdPcSamplingGroup, SamplesOnlyWhereTheAmdBackendRunsIt) {
    using gpufl::amd::ClassifyAmdPcSamplingGroup;
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("PcSampling", '+'),
              AmdPcSamplingGroup::Samples);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("Trace+PcSampling", '+'),
              AmdPcSamplingGroup::Samples);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("PcSampling+Trace", '+'),
              AmdPcSamplingGroup::Samples);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("Trace,PcSampling", ','),
              AmdPcSamplingGroup::Samples);
    // Any other group keeps its first engine.
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("PcSampling+PmSampling", '+'),
              AmdPcSamplingGroup::Samples);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("PmSampling+PcSampling", '+'),
              AmdPcSamplingGroup::Dropped);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("Trace+PcSampling+PmSampling", '+'),
              AmdPcSamplingGroup::Dropped);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("Trace", '+'),
              AmdPcSamplingGroup::Absent);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("Deep", '+'),
              AmdPcSamplingGroup::Absent);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("PcSamplingX", '+'),
              AmdPcSamplingGroup::Absent);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("", ','), AmdPcSamplingGroup::Absent);
}

TEST(AmdPcSamplingGroup, TrimsSpacesAndTabsAlike) {
    using gpufl::amd::ClassifyAmdPcSamplingGroup;
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("Trace+ PcSampling", '+'),
              AmdPcSamplingGroup::Samples);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("Trace,\tPcSampling", ','),
              AmdPcSamplingGroup::Samples);
    EXPECT_EQ(ClassifyAmdPcSamplingGroup("PmSampling \t, PcSampling", ','),
              AmdPcSamplingGroup::Dropped);
    const std::vector<std::string_view> names =
        gpufl::amd::SplitAmdEngineGroup(" Trace +\tPcSampling\t+", '+');
    EXPECT_EQ(names, (std::vector<std::string_view>{"Trace", "PcSampling"}));
}

TEST(AmdPcSamplingCodeObject, ListsTheRecordedFunctions) {
    const auto functions = FixtureFunctions();
    ASSERT_EQ(functions->size(), 3u);
    EXPECT_EQ((*functions)[0].name, "_Z9vectorAddPKfS0_Pfi");
    EXPECT_EQ((*functions)[0].vaddr, kVectorAdd);
    EXPECT_EQ((*functions)[0].size, 140u);
    // A local device function: only the full symbol table names it.
    EXPECT_EQ((*functions)[1].name, "_Z12scaleAndBiasff");
    EXPECT_EQ((*functions)[1].vaddr, kScaleAndBias);
    EXPECT_EQ((*functions)[1].size, 16u);
    EXPECT_EQ((*functions)[2].name, "_Z6scaledPKfPffi");
    EXPECT_EQ((*functions)[2].vaddr, kScaled);
    EXPECT_EQ((*functions)[2].size, 140u);

    const std::vector<uint8_t> not_elf(64, 0);
    EXPECT_TRUE(
        gpufl::amd::ParseAmdCodeObject(not_elf.data(), not_elf.size())
            .functions.empty());
    const auto bytes = ReadFixture();
    EXPECT_TRUE(
        gpufl::amd::ParseAmdCodeObject(bytes.data(), 1024).functions.empty());
}

// Rows of a code object without DWARF lines never wait for its ISA rows.
TEST(AmdPcSamplingCodeObject, FindsTheLineTable) {
    const auto plain = ReadFixture();
    EXPECT_FALSE(
        gpufl::amd::ParseAmdCodeObject(plain.data(), plain.size()).line_info);

    const auto lines = ReadFixture("pc_sampling_fixture.lines.gfx942.co");
    const auto code_object =
        gpufl::amd::ParseAmdCodeObject(lines.data(), lines.size());
    EXPECT_TRUE(code_object.line_info);
    ASSERT_EQ(code_object.functions.size(), 3u);
    EXPECT_EQ(code_object.functions[0].name, "_Z9vectorAddPKfS0_Pfi");
    EXPECT_EQ(code_object.functions[0].vaddr, kVectorAdd);
    EXPECT_EQ(code_object.functions[2].name, "_Z6scaledPKfPffi");
    EXPECT_EQ(code_object.functions[2].vaddr, kScaled);
}

// rocBLAS's and hipBLASLt's Tensile kernels are function symbols of size
// 0. Each runs to the next function's start, or to the end of .text, so
// their samples are kept rather than lost.
TEST(AmdPcSamplingCodeObject, UnsizedKernelsRunToTheNextFunction) {
    const auto functions = FixtureFunctions("pc_sampling_unsized.gfx942.co");
    ASSERT_EQ(functions->size(), 2u);
    EXPECT_EQ((*functions)[0].name, "first_kernel");
    EXPECT_EQ((*functions)[0].vaddr, 0x1400u);
    EXPECT_EQ((*functions)[0].size, 0x100u);
    EXPECT_EQ((*functions)[1].name, "second_kernel");
    EXPECT_EQ((*functions)[1].vaddr, 0x1500u);
    EXPECT_EQ((*functions)[1].size, 0x8u);  // .text ends at 0x1508

    // s_cbranch_scc1, after a local label that doesn't end the kernel.
    const auto branch = gpufl::amd::LocateAmdPcSample(*functions, 0x140c);
    ASSERT_TRUE(branch.has_value());
    EXPECT_EQ(branch->function->name, "first_kernel");
    EXPECT_EQ(branch->pc_offset, 0xcu);
    const auto end = gpufl::amd::LocateAmdPcSample(*functions, 0x1504);
    ASSERT_TRUE(end.has_value());
    EXPECT_EQ(end->function->name, "second_kernel");
    EXPECT_EQ(end->pc_offset, 0x4u);
    EXPECT_FALSE(gpufl::amd::LocateAmdPcSample(*functions, 0x1508));
}

// A sh_entsize near 2^64 used to wrap the symbol offset and read the bytes
// before the table. The table then holds no whole entry.
TEST(AmdPcSamplingCodeObject, AHugeSymbolEntrySizeReadsNoSymbol) {
    auto elf = ReadFixture("pc_sampling_unsized.gfx942.co");
    const size_t symtab = SectionHeader(elf, 2 /* SHT_SYMTAB */);
    ASSERT_NE(symtab, 0u);
    const size_t table = ReadLe(elf, symtab + 24, 8);
    const size_t table_size = ReadLe(elf, symtab + 32, 8);
    // Plant a function symbol just before the table: a wrapped offset
    // would find it.
    size_t kernel = 0;
    for (size_t at = table; at + 24 <= table + table_size; at += 24) {
        if (elf[at + 4] == 0x12 && ReadLe(elf, at + 8, 8) == 0x1400) {
            kernel = at;
        }
    }
    ASSERT_NE(kernel, 0u);
    std::copy(elf.begin() + kernel, elf.begin() + kernel + 24,
              elf.begin() + table - 24);
    WriteLe(elf, table - 24 + 16, 4, 8);
    WriteLe(elf, symtab + 56, ~uint64_t{0} - 23, 8);

    EXPECT_TRUE(gpufl::amd::ParseAmdCodeObject(elf.data(), elf.size())
                    .functions.empty());
}

// rocprofiler-sdk names a kernel by kernel_address = kernel_object (the
// loaded .kd) + kernel_code_entry_byte_offset, and a sample by its offset
// from the code object's load_base. The fixture's first segment starts at
// virtual address 0, so the kernel lands where its symbol table entry says.
TEST(AmdPcSamplingCodeObject, KernelDescriptorsEnterAtTheirFunctions) {
    const auto bytes = ReadFixture();
    ASSERT_GE(bytes.size(), 0xe40u);
    const auto entry_offset = [&](const uint64_t descriptor) {
        uint64_t offset = 0;  // kernel_code_entry_byte_offset, little-endian
        for (int i = 7; i >= 0; --i) {
            offset = (offset << 8) | bytes[descriptor + 16 + i];
        }
        return offset;
    };
    constexpr uint64_t kLoadBase = 0x7f2a00000000;
    EXPECT_EQ(kLoadBase + 0xdc0 + entry_offset(0xdc0) - kLoadBase,
              kVectorAdd);
    EXPECT_EQ(kLoadBase + 0xe00 + entry_offset(0xe00) - kLoadBase, kScaled);
}

TEST(AmdPcSamplingCodeObject, SampleOffsetsLandOnTheIsaRows) {
    const auto functions = FixtureFunctions();
    const auto expect_rows = [&](const uint64_t base,
                                 const std::vector<uint32_t>& rows,
                                 const std::string& name) {
        for (const uint32_t row : rows) {
            const auto location =
                gpufl::amd::LocateAmdPcSample(*functions, base + row);
            ASSERT_TRUE(location.has_value()) << name << "+" << row;
            EXPECT_EQ(location->function->name, name);
            EXPECT_EQ(location->pc_offset, row);
        }
    };
    expect_rows(kVectorAdd, kVectorAddRows, "_Z9vectorAddPKfS0_Pfi");
    expect_rows(kScaleAndBias, kScaleAndBiasRows, "_Z12scaleAndBiasff");
    expect_rows(kScaled, kScaledRows, "_Z6scaledPKfPffi");

    // Padding after a function, and anything before the code.
    EXPECT_FALSE(gpufl::amd::LocateAmdPcSample(*functions, 0x1f9c));
    EXPECT_FALSE(gpufl::amd::LocateAmdPcSample(*functions, 0x208c));
    EXPECT_FALSE(gpufl::amd::LocateAmdPcSample(*functions, 0xdc0));
}

TEST(AmdPcSampleAggregator, AddsUpOneRowPerFunctionOffsetAndBucket) {
    AmdPcSampleAggregator aggregator;
    aggregator.addCodeObject(7, 0xc0ffee, FixtureFunctions());
    constexpr uint32_t kWaitcnt = 0;
    constexpr uint32_t kValu = 1;

    // vectorAdd's s_waitcnt vmcnt(0), waiting three times, issued once.
    for (uint32_t launch = 1; launch <= 3; ++launch) {
        aggregator.addSample(7, kVectorAdd + 0x78, false, kWaitcnt, launch);
    }
    aggregator.addSample(7, kVectorAdd + 0x78, true, kValu, 4);
    aggregator.addSample(7, kScaleAndBias + 0x4, true, kValu, 5);
    aggregator.addSample(7, 0x1f9c, false, kWaitcnt, 5);  // padding
    aggregator.addSample(9, kVectorAdd, true, kValu, 5);  // unknown object

    const auto& totals = aggregator.totals();
    EXPECT_EQ(totals.samples, 5u);
    EXPECT_EQ(totals.issued, 2u);
    EXPECT_EQ(totals.not_issued, 3u);
    EXPECT_EQ(totals.unmatched, 2u);

    // Rows wait for their code object's ISA line table.
    EXPECT_TRUE(aggregator.takeRows([](uint64_t) { return false; }).empty());
    EXPECT_EQ(aggregator.waitingCodeObjects(),
              (std::vector<uint64_t>{0xc0ffee}));
    const auto rows = aggregator.takeRows([](const uint64_t crc) {
        return crc == 0xc0ffee;
    });
    EXPECT_TRUE(aggregator.waitingCodeObjects().empty());
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0].function, "_Z12scaleAndBiasff");
    EXPECT_EQ(rows[0].pc_offset, 0x4u);
    EXPECT_EQ(rows[0].bucket, kValu);
    EXPECT_EQ(rows[0].samples, 1u);
    EXPECT_EQ(rows[1].function, "_Z9vectorAddPKfS0_Pfi");
    EXPECT_EQ(rows[1].pc_offset, 0x78u);
    EXPECT_EQ(rows[1].bucket, kWaitcnt);
    EXPECT_EQ(rows[1].samples, 3u);
    EXPECT_EQ(rows[1].corr_id, 3u);  // the latest launch
    EXPECT_EQ(rows[1].code_object_crc, 0xc0ffeeu);
    EXPECT_EQ(rows[2].bucket, kValu);
    EXPECT_EQ(rows[2].samples, 1u);

    // A take empties the aggregate; the totals stay.
    EXPECT_TRUE(aggregator.takeRows([](uint64_t) { return true; }).empty());
    EXPECT_EQ(aggregator.totals().samples, 5u);
}

TEST(AmdPcSamplingCapability, NotRequested) {
    gpufl::amd::AmdCaptureCapabilityInput input;
    input.plan.requested_engine = gpufl::ProfilingEngine::Trace;
    input.plan.selected_path = gpufl::amd::AmdProfilingPath::BufferTracing;
    input.trace_configured = true;
    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);
    const auto* row = PcSamplingRow(event);
    ASSERT_NE(row, nullptr);
    EXPECT_FALSE(row->requested);
    EXPECT_EQ(row->status, "not_requested");
    EXPECT_EQ(row->mode, "disabled");
    EXPECT_EQ(row->reason_code, "not_selected");
    EXPECT_EQ(row->message, "AMD PC sampling was not requested.");
}

TEST(AmdPcSamplingCapability, NamesEachRefusal) {
    const auto not_opted_in = Refused(AmdPcSamplingRefusal::NotOptedIn);
    EXPECT_TRUE(not_opted_in.requested);
    EXPECT_EQ(not_opted_in.status, "skipped");
    EXPECT_EQ(not_opted_in.mode, "disabled");
    EXPECT_EQ(not_opted_in.reason_code, "amd_pc_sampling_not_opted_in");
    EXPECT_NE(not_opted_in.message.find("beta feature of rocprofiler-sdk"),
              std::string::npos);

    const auto method = Refused(AmdPcSamplingRefusal::MethodUnavailable);
    EXPECT_EQ(method.status, "skipped");
    EXPECT_EQ(method.reason_code, "amd_pc_sampling_method_unavailable");
    EXPECT_NE(method.message.find("no stochastic PC sampling"),
              std::string::npos);

    const auto driver = Refused(AmdPcSamplingRefusal::DriverUnsupported);
    EXPECT_EQ(driver.status, "skipped");
    EXPECT_EQ(driver.reason_code, "amd_pc_sampling_driver_unsupported");
    EXPECT_NE(driver.message.find("amdgpu-dkms"), std::string::npos);

    const auto unavailable = Refused(AmdPcSamplingRefusal::Unavailable);
    EXPECT_EQ(unavailable.status, "skipped");
    EXPECT_EQ(unavailable.reason_code, "amd_pc_sampling_unavailable");

    const auto failed =
        Refused(AmdPcSamplingRefusal::ConfigurationFailed,
                "ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT");
    EXPECT_EQ(failed.status, "skipped");
    EXPECT_EQ(failed.reason_code, "amd_pc_sampling_configuration_failed");
    EXPECT_EQ(failed.message,
              "rocprofiler-sdk refused PC sampling: "
              "ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT.");

    const auto window = Refused(AmdPcSamplingRefusal::WindowOnly);
    EXPECT_EQ(window.status, "skipped");
    EXPECT_EQ(window.mode, "disabled");
    EXPECT_EQ(window.reason_code, "amd_pc_sampling_window_only_unsupported");
    EXPECT_NE(window.message.find("can't be limited to deep windows"),
              std::string::npos);
}

// PmSampling+PcSampling runs PmSampling on AMD: the PC sampling row says so
// rather than reading as never requested.
TEST(AmdPcSamplingCapability, NamesADroppedGroupMember) {
    gpufl::amd::AmdProfilingRequest request;
    request.engine = gpufl::ProfilingEngine::PmSampling;
    request.pc_sampling_dropped = true;
    gpufl::amd::AmdProfilingSupport support;
    support.device_counting = true;
    gpufl::amd::AmdCaptureCapabilityInput input;
    input.plan = gpufl::amd::ResolveAmdProfilingPlan(request, support);
    input.trace_configured = true;
    const auto event = gpufl::amd::BuildAmdCaptureCapabilitiesEvent(input);

    EXPECT_EQ(event.selected_engine, "amd.device_counting");
    const auto* row = PcSamplingRow(event);
    ASSERT_NE(row, nullptr);
    EXPECT_TRUE(row->requested);
    EXPECT_EQ(row->status, "skipped");
    EXPECT_EQ(row->mode, "disabled");
    EXPECT_EQ(row->reason_code, "amd_pc_sampling_group_unsupported");
    EXPECT_EQ(row->message,
              "On AMD GPUs, PC sampling runs alone or beside Trace, so this "
              "pass's group ran pm_sampling instead.");
}

TEST(AmdPcSamplingCapability, CountsWhatWasCollected) {
    const auto collected = Sampled(1200, 34567, 0);
    EXPECT_TRUE(collected.requested);
    EXPECT_EQ(collected.status, "collected");
    EXPECT_EQ(collected.mode, "rocprofiler_pc_sampling_stochastic");
    EXPECT_TRUE(collected.reason_code.empty());
    EXPECT_EQ(collected.message,
              "Stochastic PC sampling every 1,048,576 cycles: 35,767 samples, "
              "1,200 issued, 34,567 waiting.");

    const auto partial = Sampled(10, 20, 4);
    EXPECT_EQ(partial.status, "partial");
    EXPECT_EQ(partial.reason_code, "amd_pc_sampling_samples_dropped");
    EXPECT_EQ(partial.message,
              "Stochastic PC sampling every 1,048,576 cycles: 30 samples, 10 "
              "issued, 20 waiting. 4 samples were dropped or matched no "
              "kernel.");

    const auto empty = Sampled(0, 0, 0);
    EXPECT_EQ(empty.status, "enabled_no_data");
    EXPECT_EQ(empty.mode, "rocprofiler_pc_sampling_stochastic");
    EXPECT_EQ(empty.reason_code, "enabled_but_no_records");
    EXPECT_EQ(empty.message,
              "AMD PC sampling was enabled but recorded no samples this "
              "session.");

    // Every sample lost is not a quiet run.
    const auto lost = Sampled(0, 0, 4);
    EXPECT_EQ(lost.status, "partial");
    EXPECT_EQ(lost.reason_code, "amd_pc_sampling_samples_dropped");
    EXPECT_EQ(lost.message,
              "Stochastic PC sampling every 1,048,576 cycles kept no samples: "
              "4 were dropped or matched no kernel.");
}
