#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpufl/core/common.hpp"
#include "gpufl/core/dictionary_manager.hpp"
#include "gpufl/core/logger/log_sink.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/source_capture_policy.hpp"

namespace {

namespace fs = std::filesystem;
using gpufl::detail::SourceCaptureDisposition;

class SourceCapturePolicyTest : public testing::Test {
   protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() /
                ("gpufl_source_capture_" +
                 std::to_string(gpufl::detail::GetPid()));
        std::error_code ec;
        fs::remove_all(root_, ec);
        fs::create_directories(root_ / "src", ec);
        ASSERT_FALSE(ec) << ec.message();
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    fs::path write(const fs::path& relative, const std::string& content) {
        const fs::path path = root_ / relative;
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
        EXPECT_FALSE(ec) << ec.message();
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        EXPECT_TRUE(output.is_open());
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
        output.close();
        return path;
    }

    gpufl::SourceCaptureSettings settings() const {
        gpufl::SourceCaptureSettings value;
        value.approved_roots = {root_.string()};
        return value;
    }

    fs::path root_;
};

TEST_F(SourceCapturePolicyTest, CapturesApprovedCudaSourceWithLogicalPath) {
    const fs::path source = write("src/kernel.cu", "line one\r\nline two\n");
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, settings());

    const auto result = policy.capture(
        source.string(), 7, "profiler_source_correlation");

    EXPECT_EQ(result.record.disposition, SourceCaptureDisposition::Captured);
    EXPECT_EQ(result.record.source_file_id, 7u);
    EXPECT_EQ(result.record.logical_path, "src/kernel.cu");
    EXPECT_EQ(result.record.bytes, 19u);
    EXPECT_EQ(result.lines,
              (std::vector<std::string>{"line one", "line two"}));
    EXPECT_EQ(policy.manifest().captured_files, 1u);
    EXPECT_EQ(policy.manifest().captured_bytes, 19u);
}

TEST_F(SourceCapturePolicyTest, RejectsPathsOutsideTheApprovedRoot) {
    const fs::path outside = root_.parent_path() / "gpufl_outside_source.cu";
    {
        std::ofstream output(outside, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(output.is_open());
        output << "__global__ void outside() {}\n";
    }
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, settings());

    const auto result = policy.capture(outside.string(), 2, "debug_line");

    EXPECT_EQ(result.record.disposition,
              SourceCaptureDisposition::OutsideApprovedRoots);
    EXPECT_TRUE(result.lines.empty());
    EXPECT_EQ(result.record.logical_path,
              "unavailable/source-2/gpufl_outside_source.cu");
    std::error_code ec;
    fs::remove(outside, ec);
}

TEST_F(SourceCapturePolicyTest, CapturesRustSourceNamedByDeviceLineTable) {
    // cuda-oxide kernels: the cubin's line table names src/main.rs and the
    // profiler correlation hands that path here, like a .cu file.
    const fs::path source =
        write("src/main.rs", "#[kernel]\npub fn vecadd() {}\n");
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, settings());
    const auto result = policy.capture(
        source.string(), 1, "profiler_source_correlation");
    EXPECT_EQ(result.record.disposition, SourceCaptureDisposition::Captured);
    EXPECT_EQ(result.record.logical_path, "src/main.rs");
    EXPECT_EQ(result.lines,
              (std::vector<std::string>{"#[kernel]", "pub fn vecadd() {}"}));
    // Cargo manifests and lockfiles beside the sources are not source.
    const fs::path manifest = write("Cargo.toml", "[package]\n");
    EXPECT_EQ(policy.capture(manifest.string(), 2, "profiler_source_correlation")
                  .record.disposition,
              SourceCaptureDisposition::UnsupportedExtension);
}

TEST_F(SourceCapturePolicyTest, RejectsUnsupportedAndNonTextFiles) {
    const fs::path unsupported = write("src/notes.txt", "not source\n");
    const fs::path binary = write("src/binary.cu", std::string("a\0b", 3));
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, settings());

    EXPECT_EQ(policy.capture(unsupported.string(), 1, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::UnsupportedExtension);
    EXPECT_EQ(policy.capture(binary.string(), 2, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::NonTextContent);
}

TEST_F(SourceCapturePolicyTest, EnforcesFileTotalAndLineBudgets) {
    const fs::path first = write("src/first.cu", "a\nb\n");
    const fs::path total = write("src/total.cu", "c\nd\ne\n");
    const fs::path large = write("src/large.cu", "123456789");
    const fs::path long_line = write("src/line.cu", "12345\n");
    gpufl::detail::SourceCapturePolicy policy;

    auto file_config = settings();
    file_config.limits.max_bytes_per_file = 8;
    policy.configure(true, file_config);
    EXPECT_EQ(policy.capture(large.string(), 1, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::FileTooLarge);

    auto line_config = settings();
    line_config.limits.max_line_bytes = 4;
    policy.configure(true, line_config);
    EXPECT_EQ(policy.capture(long_line.string(), 2, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::LineTooLong);

    auto total_config = settings();
    total_config.limits.max_total_bytes = 5;
    policy.configure(true, total_config);
    EXPECT_EQ(policy.capture(total.string(), 3, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::TotalBudgetExceeded);

    auto count_config = settings();
    count_config.limits.max_files = 1;
    policy.configure(true, count_config);
    EXPECT_EQ(policy.capture(first.string(), 4, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::Captured);
    EXPECT_EQ(policy.capture(total.string(), 5, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::FileLimitExceeded);
}

TEST_F(SourceCapturePolicyTest, OptOutAndMissingRootFailClosed) {
    const fs::path source = write("src/kernel.cu", "kernel\n");
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(false, settings());
    EXPECT_EQ(policy.capture(source.string(), 1, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::Disabled);

    gpufl::SourceCaptureSettings no_roots;
    policy.configure(true, no_roots);
    EXPECT_EQ(policy.capture(source.string(), 2, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::NoApprovedRoot);
}

TEST_F(SourceCapturePolicyTest, RejectsDirectoryEvenWithSourceExtension) {
    const fs::path directory = root_ / "src/not-a-file.cu";
    std::error_code ec;
    fs::create_directory(directory, ec);
    ASSERT_FALSE(ec) << ec.message();
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, settings());

    EXPECT_EQ(policy.capture(directory.string(), 1, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::NotRegularFile);
}

TEST_F(SourceCapturePolicyTest, RejectsSymlinkThatEscapesTheApprovedRoot) {
    const fs::path outside = root_.parent_path() / "gpufl_symlink_target.cu";
    {
        std::ofstream output(outside, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(output.is_open());
        output << "__global__ void escaped() {}\n";
    }
    const fs::path link = root_ / "src/escaped.cu";
    std::error_code ec;
    fs::create_symlink(outside, link, ec);
    if (ec) {
        const std::string message = ec.message();
        std::error_code cleanup_ec;
        fs::remove(outside, cleanup_ec);
        GTEST_SKIP() << "Creating symlinks is unavailable: " << message;
    }
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, settings());

    EXPECT_EQ(policy.capture(link.string(), 1, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::SymlinkEscape);
    fs::remove(outside, ec);
}

TEST_F(SourceCapturePolicyTest, BoundsDetailedManifestEntries) {
    auto config = settings();
    config.limits.max_manifest_entries = 1;
    const fs::path first = write("src/first.txt", "one\n");
    const fs::path second = write("src/second.txt", "two\n");
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, config);

    policy.capture(first.string(), 1, "debug_line");
    policy.capture(second.string(), 2, "debug_line");

    EXPECT_EQ(policy.manifest().files.size(), 1u);
    EXPECT_EQ(policy.manifest().skipped_files, 2u);
    EXPECT_EQ(policy.manifest().omitted_manifest_entries, 1u);
}

TEST_F(SourceCapturePolicyTest, SkipsVendoredDirectoriesWithoutReadingThem) {
    const fs::path vendored =
        write("third_party/cutlass/gemm.cuh", "__device__ int cutlass;\n");
    const fs::path fetched =
        write("build/_deps/cub-src/cub.cuh", "__device__ int cub;\n");
    const fs::path venv = write(
        ".venv/lib/python3.12/site-packages/torch/include/aten.h", "aten\n");
    // Excluded before the extension check, so nothing about the file matters.
    const fs::path notes = write("vendor/notes.txt", "notes\n");
    // Only directories count: a file named like one is project code.
    const fs::path own = write("src/vendor.cu", "__global__ void own() {}\n");
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, settings());

    const auto result =
        policy.capture(vendored.string(), 1, "profiler_source_correlation");
    EXPECT_EQ(result.record.disposition,
              SourceCaptureDisposition::ExcludedThirdParty);
    EXPECT_EQ(result.record.logical_path, "third_party/cutlass/gemm.cuh");
    EXPECT_EQ(result.record.bytes, 0u);
    EXPECT_TRUE(result.lines.empty());
    EXPECT_STREQ(gpufl::detail::sourceCaptureDispositionName(
                     result.record.disposition),
                 "excluded_third_party");
    for (const fs::path& path : {fetched, venv, notes}) {
        EXPECT_EQ(policy.capture(path.string(), 2, "debug_line")
                      .record.disposition,
                  SourceCaptureDisposition::ExcludedThirdParty)
            << path;
    }
    EXPECT_EQ(policy.capture(own.string(), 3, "debug_line").record.disposition,
              SourceCaptureDisposition::Captured);
    EXPECT_EQ(policy.manifest().captured_files, 1u);
    EXPECT_EQ(policy.manifest().skipped_files, 4u);
}

TEST_F(SourceCapturePolicyTest, MatchesVendoredDirectoryNamesInAnyCase) {
    const fs::path upper = write("ThirdParty/Eigen/core.h", "eigen\n");
    const fs::path mixed = write("src/External/lib.cuh", "lib\n");
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, settings());

    EXPECT_EQ(policy.capture(upper.string(), 1, "debug_line").record.disposition,
              SourceCaptureDisposition::ExcludedThirdParty);
    EXPECT_EQ(policy.capture(mixed.string(), 2, "debug_line").record.disposition,
              SourceCaptureDisposition::ExcludedThirdParty);
}

TEST_F(SourceCapturePolicyTest, RootInsideAVendoredNameIsStillProjectCode) {
    // The project itself lives under a folder named external: only the
    // directories below the approved root are judged.
    const fs::path source =
        write("external/myproj/src/kernel.cu", "__global__ void k() {}\n");
    gpufl::SourceCaptureSettings config;
    config.approved_roots = {(root_ / "external" / "myproj").string()};
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, config);

    const auto result = policy.capture(source.string(), 1, "debug_line");
    EXPECT_EQ(result.record.disposition, SourceCaptureDisposition::Captured);
    EXPECT_EQ(result.record.logical_path, "src/kernel.cu");
}

TEST_F(SourceCapturePolicyTest, SkipsGitSubmodulePaths) {
    write(".gitmodules",
          "[submodule \"deps/cutlass\"]\n"
          "\tpath = deps/cutlass\n"
          "\turl = https://github.com/NVIDIA/cutlass.git\n"
          "[submodule \"escape\"]\n"
          "\tpath = ../outside\n");
    const fs::path submodule = write("deps/cutlass/include/gemm.h", "gemm\n");
    const fs::path sibling = write("deps/cutlass-ext/ext.cuh", "ext\n");
    const fs::path own = write("deps/own.cu", "own\n");
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, settings());

    EXPECT_EQ(policy.capture(submodule.string(), 1, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::ExcludedThirdParty);
    EXPECT_EQ(policy.capture(sibling.string(), 2, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::Captured);
    EXPECT_EQ(policy.capture(own.string(), 3, "debug_line").record.disposition,
              SourceCaptureDisposition::Captured);
}

TEST_F(SourceCapturePolicyTest, IncludeThirdPartyKeepsExplicitExclusions) {
    write(".gitmodules", "[submodule \"deps/cub\"]\n\tpath = deps/cub\n");
    const fs::path vendored = write("third_party/cutlass/gemm.cuh", "gemm\n");
    const fs::path submodule = write("deps/cub/cub.cuh", "cub\n");
    const fs::path generated = write("src/Generated/tables.cu", "tables\n");
    auto config = settings();
    config.include_third_party = true;
    config.excluded_dirs = {"generated"};
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, config);

    EXPECT_EQ(policy.capture(vendored.string(), 1, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::Captured);
    EXPECT_EQ(policy.capture(submodule.string(), 2, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::Captured);
    EXPECT_EQ(policy.capture(generated.string(), 3, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::ExcludedThirdParty);
}

TEST_F(SourceCapturePolicyTest, ExcludedDirectoryPathsSkipTheirSubtree) {
    const fs::path cached = write("build/cache/kernel.cu", "cached\n");
    const fs::path built = write("build/kernel.cu", "built\n");
    const fs::path tool = write("tools/gen/tool.cu", "tool\n");
    auto config = settings();
    config.excluded_dirs = {"build/cache", (root_ / "tools").string()};
    gpufl::detail::SourceCapturePolicy policy;
    policy.configure(true, config);

    EXPECT_EQ(policy.capture(cached.string(), 1, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::ExcludedThirdParty);
    EXPECT_EQ(policy.capture(built.string(), 2, "debug_line")
                  .record.disposition,
              SourceCaptureDisposition::Captured);
    EXPECT_EQ(policy.capture(tool.string(), 3, "debug_line").record.disposition,
              SourceCaptureDisposition::ExcludedThirdParty);
}

class RecordingSink final : public gpufl::ILogSink {
   public:
    explicit RecordingSink(std::shared_ptr<std::vector<std::string>> lines)
        : lines_(std::move(lines)) {}
    void write(gpufl::Channel, std::string_view json) override {
        lines_->emplace_back(json);
    }
    void close() override {}

   private:
    std::shared_ptr<std::vector<std::string>> lines_;
};

TEST_F(SourceCapturePolicyTest, DictionaryEmitsContentAndBoundedManifest) {
    const fs::path source = write("src/kernel.cu", "first\nsecond\n");
    gpufl::DictionaryManager dictionary;
    dictionary.configureSourceCapture(true, settings());
    EXPECT_EQ(dictionary.internSourceFile(source.string()), 1u);

    auto lines = std::make_shared<std::vector<std::string>>();
    gpufl::Logger logger;
    logger.addSink(std::make_unique<RecordingSink>(lines));
    dictionary.flushDictionary(logger, "session-1");
    dictionary.flushSourceContent(logger, "session-1");

    ASSERT_EQ(lines->size(), 3u);
    EXPECT_NE((*lines)[0].find(R"("type":"dictionary_update")"),
              std::string::npos);
    EXPECT_NE((*lines)[0].find(R"("source_file_dict":{"1":"src/kernel.cu"})"),
              std::string::npos);
    EXPECT_EQ((*lines)[0].find(root_.string()), std::string::npos);
    EXPECT_NE((*lines)[1].find(R"("type":"source_capture_manifest")"),
              std::string::npos);
    EXPECT_NE((*lines)[1].find(R"("logical_path":"src/kernel.cu")"),
              std::string::npos);
    EXPECT_NE((*lines)[1].find(R"("disposition":"captured")"),
              std::string::npos);
    EXPECT_EQ((*lines)[1].find(root_.string()), std::string::npos);
    EXPECT_NE((*lines)[2].find(R"("type":"source_file_content")"),
              std::string::npos);
    EXPECT_NE((*lines)[2].find(R"("lines":["first","second"])"),
              std::string::npos);
}

TEST_F(SourceCapturePolicyTest, ManifestNamesVendoredFilesWithoutContent) {
    const fs::path vendored =
        write("third_party/cutlass/gemm.cuh", "vendored-secret\n");
    gpufl::DictionaryManager dictionary;
    dictionary.configureSourceCapture(true, settings());
    EXPECT_EQ(dictionary.internSourceFile(vendored.string()), 1u);

    auto lines = std::make_shared<std::vector<std::string>>();
    gpufl::Logger logger;
    logger.addSink(std::make_unique<RecordingSink>(lines));
    dictionary.flushDictionary(logger, "session-1");
    dictionary.flushSourceContent(logger, "session-1");

    ASSERT_EQ(lines->size(), 2u);
    EXPECT_NE((*lines)[1].find(R"("type":"source_capture_manifest")"),
              std::string::npos);
    EXPECT_NE((*lines)[1].find(
                  R"("logical_path":"third_party/cutlass/gemm.cuh")"),
              std::string::npos);
    EXPECT_NE((*lines)[1].find(R"("disposition":"excluded_third_party")"),
              std::string::npos);
    for (const auto& line : *lines) {
        EXPECT_EQ(line.find("source_file_content"), std::string::npos);
        EXPECT_EQ(line.find("vendored-secret"), std::string::npos);
    }
}


#ifndef _WIN32
TEST_F(SourceCapturePolicyTest, DisassemblerDoesNotInheritInjectionSentinel) {
    struct SavedEnvironment {
        const char* key;
        bool present;
        std::string value;
        explicit SavedEnvironment(const char* name) : key(name),
            present(std::getenv(name) != nullptr),
            value(present ? std::getenv(name) : "") {}
        ~SavedEnvironment() {
            if (present) setenv(key, value.c_str(), 1);
            else unsetenv(key);
        }
    } inject("GPUFL_INJECT"), rocm("ROCM_PATH");
    const auto tool = write("llvm/bin/llvm-objdump",
        "#!/bin/sh\nprintf '%s' \"${GPUFL_INJECT-unset}\" > \"$ROCM_PATH/child-env\"\n");
    fs::permissions(tool, fs::perms::owner_all);
    ASSERT_EQ(setenv("GPUFL_INJECT", "1", 1), 0);
    ASSERT_EQ(setenv("ROCM_PATH", root_.c_str(), 1), 0);
    std::vector<uint8_t> elf(20, 0);
    elf[0] = 0x7f; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F'; elf[18] = 0xe0;
    gpufl::DictionaryManager dictionary;
    dictionary.enqueueDisassembly(1, elf.data(), elf.size());
    gpufl::Logger logger;
    dictionary.flushDisassembly(logger, "helper-env-test");
    std::ifstream evidence(root_ / "child-env");
    std::string value;
    evidence >> value;
    EXPECT_EQ(value, "unset");
    EXPECT_STREQ(std::getenv("GPUFL_INJECT"), "1");
}

// AMD PC samples take their source line from the line table the ISA rows
// were built from, and key their function by the same primary file.
TEST_F(SourceCapturePolicyTest, AmdIsaLineTableAnswersPcSampleLookups) {
    struct SavedEnvironment {
        const char* key;
        bool present;
        std::string value;
        explicit SavedEnvironment(const char* name) : key(name),
            present(std::getenv(name) != nullptr),
            value(present ? std::getenv(name) : "") {}
        ~SavedEnvironment() {
            if (present) setenv(key, value.c_str(), 1);
            else unsetenv(key);
        }
    } rocm("ROCM_PATH");
    const fs::path kernel = write("src/kernel.hip", "a\nb\nc\nd\ne\nf\n");
    const std::string header =
        "/opt/rocm/include/hip/amd_detail/amd_device_functions.h";
    // llvm-objdump -d -l output for one kernel; the last two instructions
    // come from an inlined HIP header.
    const std::string listing =
        "\nkernel.co:\tfile format elf64-amdgpu\n\n"
        "Disassembly of section .text:\n\n"
        "0000000000001700 <_Z6kernelPf>:\n"
        "; " + kernel.string() + ":5\n"
        "\ts_load_dwordx2 s[0:1], s[0:1], 0x0 // 000000001700: C0060000\n"
        "; " + kernel.string() + ":6\n"
        "\tv_mov_b32_e32 v0, 1.0 // 000000001708: 7E0002F2\n"
        "; " + header + ":10\n"
        "\tglobal_store_dword v1, v0, s[0:1] // 00000000170C: DC708000\n"
        "\ts_endpgm // 000000001714: BF810000\n";
    const auto tool = write("llvm/bin/llvm-objdump",
                            "#!/bin/sh\ncat <<'LISTING'\n" + listing +
                                "LISTING\n");
    fs::permissions(tool, fs::perms::owner_all);
    ASSERT_EQ(setenv("ROCM_PATH", root_.c_str(), 1), 0);
    std::vector<uint8_t> elf(20, 0);
    elf[0] = 0x7f; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F'; elf[18] = 0xe0;

    using Lookup = gpufl::DictionaryManager::IsaSourceLookup;
    gpufl::DictionaryManager dictionary;
    dictionary.configureSourceCapture(true, settings());
    dictionary.retainAmdIsaLines();
    gpufl::DictionaryManager::IsaSourceLine line;
    EXPECT_EQ(dictionary.lookupAmdIsaSource(42, "_Z6kernelPf", 8, &line),
              Lookup::NotParsed);

    dictionary.enqueueDisassembly(42, elf.data(), elf.size());
    gpufl::Logger logger;
    dictionary.flushDisassembly(logger, "amd-isa-lines");

    ASSERT_EQ(dictionary.lookupAmdIsaSource(42, "_Z6kernelPf", 0x8, &line),
              Lookup::Found);
    EXPECT_EQ(line.function_source, kernel.string());
    EXPECT_EQ(line.source_file, kernel.string());
    EXPECT_EQ(line.source_line, 6u);

    // An inlined header line belongs to the function keyed by its own file.
    // With no llvm-symbolizer to name its call, it keeps the header's line.
    ASSERT_EQ(dictionary.lookupAmdIsaSource(42, "_Z6kernelPf", 0xc, &line),
              Lookup::Found);
    EXPECT_EQ(line.function_source, kernel.string());
    EXPECT_EQ(line.source_file, header);
    EXPECT_EQ(line.source_line, 10u);

    // Not an instruction: the function's file, but no line.
    ASSERT_EQ(dictionary.lookupAmdIsaSource(42, "_Z6kernelPf", 0x4, &line),
              Lookup::Found);
    EXPECT_EQ(line.function_source, kernel.string());
    EXPECT_TRUE(line.source_file.empty());
    EXPECT_EQ(line.source_line, 0u);

    EXPECT_EQ(dictionary.lookupAmdIsaSource(42, "_Z5otherv", 0, &line),
              Lookup::NoLine);

    // The ISA rows interned the kernel's file under this logical name.
    EXPECT_EQ(dictionary.sourceFileName(
                  dictionary.internSourceFile(kernel.string())),
              "src/kernel.hip");

    // Only PC sampling asks for the tables; a trace keeps none.
    gpufl::DictionaryManager tracing;
    tracing.configureSourceCapture(true, settings());
    tracing.enqueueDisassembly(42, elf.data(), elf.size());
    tracing.flushDisassembly(logger, "amd-isa-lines-trace");
    EXPECT_EQ(tracing.lookupAmdIsaSource(42, "_Z6kernelPf", 8, &line),
              Lookup::NotParsed);
}

// AMD PC sampling asks for the line tables of code objects with DWARF
// lines; every collector flush disassembles those, and only those, so its
// rows need not wait for the run's end.
TEST_F(SourceCapturePolicyTest, AmdIsaRequestsDisassembleOnlyWhatWasAsked) {
    struct SavedEnvironment {
        const char* key;
        bool present;
        std::string value;
        explicit SavedEnvironment(const char* name) : key(name),
            present(std::getenv(name) != nullptr),
            value(present ? std::getenv(name) : "") {}
        ~SavedEnvironment() {
            if (present) setenv(key, value.c_str(), 1);
            else unsetenv(key);
        }
    } rocm("ROCM_PATH");
    const fs::path kernel = write("src/kernel.hip", "a\nb\nc\nd\ne\nf\n");
    const std::string listing =
        "\nkernel.co:\tfile format elf64-amdgpu\n\n"
        "Disassembly of section .text:\n\n"
        "0000000000001700 <_Z6kernelPf>:\n"
        "; " + kernel.string() + ":5\n"
        "\ts_load_dwordx2 s[0:1], s[0:1], 0x0 // 000000001700: C0060000\n"
        "\ts_endpgm // 000000001708: BF810000\n";
    const auto tool = write("llvm/bin/llvm-objdump",
                            "#!/bin/sh\ncat <<'LISTING'\n" + listing +
                                "LISTING\n");
    fs::permissions(tool, fs::perms::owner_all);
    ASSERT_EQ(setenv("ROCM_PATH", root_.c_str(), 1), 0);
    std::vector<uint8_t> elf(20, 0);
    elf[0] = 0x7f; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F'; elf[18] = 0xe0;

    using Lookup = gpufl::DictionaryManager::IsaSourceLookup;
    gpufl::DictionaryManager dictionary;
    dictionary.configureSourceCapture(true, settings());
    gpufl::Logger logger;
    const auto lookup = [&](const uint64_t crc) {
        return dictionary.lookupAmdIsaSource(crc, "_Z6kernelPf", 0, nullptr);
    };

    // Not retained: nothing is asked for.
    dictionary.requestAmdIsaLines(41);
    dictionary.retainAmdIsaLines();
    dictionary.enqueueDisassembly(41, elf.data(), elf.size());
    dictionary.flushRequestedAmdDisassembly(logger, "amd-isa-requests");
    EXPECT_EQ(lookup(41), Lookup::NotParsed);

    // Asked for before the backend enqueues it: the request waits.
    dictionary.requestAmdIsaLines(42);
    dictionary.flushRequestedAmdDisassembly(logger, "amd-isa-requests");
    EXPECT_EQ(lookup(42), Lookup::NotParsed);
    dictionary.enqueueDisassembly(42, elf.data(), elf.size());
    dictionary.enqueueDisassembly(43, elf.data(), elf.size());
    dictionary.flushRequestedAmdDisassembly(logger, "amd-isa-requests");
    EXPECT_EQ(lookup(42), Lookup::Found);
    EXPECT_EQ(lookup(43), Lookup::NotParsed);

    // The full flush still takes everything else.
    dictionary.flushDisassembly(logger, "amd-isa-requests");
    EXPECT_EQ(lookup(41), Lookup::Found);
    EXPECT_EQ(lookup(43), Lookup::Found);
}

// llvm-objdump -l gives code inlined from a toolchain header the header's
// line. One llvm-symbolizer run per code object finds each such
// instruction's call in the program, and the ISA rows and the PC samples'
// line table take that call's line. The tools' output is recorded from
// tests/backends/amd/data/inline_call_sites.gfx942.co.
TEST_F(SourceCapturePolicyTest, AmdInlinedHeaderCodeTakesItsCallsLine) {
    struct SavedEnvironment {
        const char* key;
        bool present;
        std::string value;
        explicit SavedEnvironment(const char* name) : key(name),
            present(std::getenv(name) != nullptr),
            value(present ? std::getenv(name) : "") {}
        ~SavedEnvironment() {
            if (present) setenv(key, value.c_str(), 1);
            else unsetenv(key);
        }
    } rocm("ROCM_PATH");
    const fs::path kernel = write("src/softmax.hip", "softmaxRows\n");
    const std::string hip =
        "/opt/rocm-7.2.0/lib/llvm/bin/../../../include/hip/amd_detail/";
    const std::string runtime = hip + "amd_hip_runtime.h";
    const std::string device = hip + "amd_device_functions.h";
    const std::string math =
        "/opt/rocm-7.2.0/lib/llvm/lib/clang/22/include/__clang_hip_math.h";
    const std::string scan =
        "/opt/rocm-7.2.0/include/rocprim/block/block_scan.hpp";
    // blockIdx.x, a line of the kernel's own, __syncthreads()'s wait and
    // barrier, and expf; then a function whose every frame is a header.
    const std::string listing =
        "\nkernel.co:\tfile format elf64-amdgpu\n\n"
        "Disassembly of section .text:\n\n"
        "0000000000001c00 <_Z11softmaxRowsPKfPfii>:\n"
        "; _Z11softmaxRowsPKfPfii():\n"
        "; " + runtime + ":259\n"
        "\ts_load_dwordx2 s[8:9], s[0:1], 0x10 // 000000001C00: C0060200\n"
        "; " + kernel.string() + ":17\n"
        "\ts_waitcnt lgkmcnt(0) // 000000001C08: BF8CC07F\n"
        "; " + device + ":691\n"
        "\ts_waitcnt lgkmcnt(0) // 000000001CC4: BF8CC07F\n"
        "; " + device + ":692\n"
        "\ts_barrier // 000000001CC8: BF8A0000\n"
        "; " + math + ":431\n"
        "\tv_exp_f32_e32 v7, v7 // 000000001DF0: 7E0E4107\n\n"
        "0000000000002400 <_Z8scanTilePf>:\n"
        "; " + scan + ":180\n"
        "\tv_add_f32_e32 v0, v0, v1 // 000000002400: 02000300\n";
    // llvm-symbolizer --inlines: each address's frames, innermost first.
    const std::string frames =
        "0x1c00\n" + runtime + ":259:58\n" + runtime + ":287:3\n" +
        kernel.string() + ":16:15\n\n"
        "0x1cc4\n" + device + ":691:5\n" + device + ":707:50\n" + device +
        ":714:3\n" + kernel.string() + ":25:5\n\n"
        "0x1cc8\n" + device + ":692:5\n" + device + ":707:50\n" + device +
        ":714:3\n" + kernel.string() + ":25:5\n\n"
        "0x1df0\n" + math + ":431:32\n" + kernel.string() + ":34:62\n\n"
        "0x2400\n" + scan + ":180:9\n"
        "/opt/rocm-7.2.0/include/rocprim/device/device_scan.hpp:95:5\n\n";
    const auto objdump = write("llvm/bin/llvm-objdump",
                               "#!/bin/sh\ncat <<'LISTING'\n" + listing +
                                   "LISTING\n");
    fs::permissions(objdump, fs::perms::owner_all);
    const auto symbolizer = write(
        "llvm/bin/llvm-symbolizer",
        "#!/bin/sh\n"
        "echo run >> \"$ROCM_PATH/symbolizer-runs\"\n"
        "printf '%s\\n' \"$@\" > \"$ROCM_PATH/symbolizer-args\"\n"
        "cat > \"$ROCM_PATH/symbolizer-stdin\"\n"
        "cat <<'FRAMES'\n" + frames + "FRAMES\n");
    fs::permissions(symbolizer, fs::perms::owner_all);
    ASSERT_EQ(setenv("ROCM_PATH", root_.c_str(), 1), 0);
    std::vector<uint8_t> elf(20, 0);
    elf[0] = 0x7f; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F'; elf[18] = 0xe0;

    gpufl::DictionaryManager dictionary;
    dictionary.configureSourceCapture(true, settings());
    dictionary.retainAmdIsaLines();
    auto lines = std::make_shared<std::vector<std::string>>();
    gpufl::Logger logger;
    logger.addSink(std::make_unique<RecordingSink>(lines));
    dictionary.enqueueDisassembly(42, elf.data(), elf.size());
    dictionary.flushDisassembly(logger, "amd-inline-call-sites");

    // One run for the code object, asked only about the header lines.
    const auto read_lines = [&](const char* name) {
        std::ifstream input(root_ / name);
        std::vector<std::string> values;
        for (std::string value; std::getline(input, value);) {
            values.push_back(value);
        }
        return values;
    };
    EXPECT_EQ(read_lines("symbolizer-runs").size(), 1u);
    const auto args = read_lines("symbolizer-args");
    EXPECT_NE(std::find(args.begin(), args.end(), "--inlines"), args.end());
    EXPECT_TRUE(std::any_of(args.begin(), args.end(), [](const auto& arg) {
        return arg.rfind("--obj=", 0) == 0;
    }));
    auto asked = read_lines("symbolizer-stdin");
    std::sort(asked.begin(), asked.end());
    EXPECT_EQ(asked, (std::vector<std::string>{"0x1c00", "0x1cc4", "0x1cc8",
                                               "0x1df0", "0x2400"}));

    using Lookup = gpufl::DictionaryManager::IsaSourceLookup;
    const auto lookup = [&](const char* function, const uint32_t pc) {
        gpufl::DictionaryManager::IsaSourceLine line;
        EXPECT_EQ(dictionary.lookupAmdIsaSource(42, function, pc, &line),
                  Lookup::Found);
        return std::make_pair(line.source_file, line.source_line);
    };
    const auto at = [](const std::string& file, const uint32_t line) {
        return std::make_pair(file, line);
    };
    const char* softmax = "_Z11softmaxRowsPKfPfii";
    EXPECT_EQ(lookup(softmax, 0x0), at(kernel.string(), 16));
    EXPECT_EQ(lookup(softmax, 0x8), at(kernel.string(), 17));
    EXPECT_EQ(lookup(softmax, 0xc4), at(kernel.string(), 25));
    EXPECT_EQ(lookup(softmax, 0xc8), at(kernel.string(), 25));
    EXPECT_EQ(lookup(softmax, 0x1f0), at(kernel.string(), 34));
    // No frame in the program: rocPRIM's line stays.
    EXPECT_EQ(lookup("_Z8scanTilePf", 0x0), at(scan, 180));

    // The ISA row of the s_barrier (pc 200) is on line 25, and no row
    // names the HIP or clang headers.
    std::string emitted;
    for (const auto& json : *lines) emitted += json + "\n";
    const uint32_t file = dictionary.internSourceFile(kernel.string());
    const uint32_t metric = dictionary.internMetric("isa_inst_present");
    EXPECT_NE(emitted.find(",200," + std::to_string(metric) + ",1,0,2,0," +
                           std::to_string(file) + ",25]"),
              std::string::npos);
    EXPECT_EQ(emitted.find("amd_hip_runtime.h"), std::string::npos);
    EXPECT_EQ(emitted.find("amd_device_functions.h"), std::string::npos);
    EXPECT_EQ(emitted.find("__clang_hip_math.h"), std::string::npos);
    EXPECT_NE(emitted.find("block_scan.hpp"), std::string::npos);
}

// The same through ROCm's own llvm-objdump and llvm-symbolizer, on the
// checked-in code object, where ROCm is installed.
TEST_F(SourceCapturePolicyTest, AmdInlinedHeaderCodeOfACompiledKernel) {
    const char* rocm = std::getenv("ROCM_PATH");
    const fs::path bin =
        fs::path(rocm && rocm[0] ? rocm : "/opt/rocm") / "llvm" / "bin";
    if (!fs::exists(bin / "llvm-objdump") ||
        !fs::exists(bin / "llvm-symbolizer")) {
        GTEST_SKIP() << "No llvm-objdump and llvm-symbolizer in " << bin;
    }
    std::ifstream input(
        std::string(GPUFL_AMD_TEST_DATA_DIR) + "/inline_call_sites.gfx942.co",
        std::ios::binary);
    const std::vector<uint8_t> code_object(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    ASSERT_FALSE(code_object.empty());

    gpufl::DictionaryManager dictionary;
    dictionary.retainAmdIsaLines();
    auto lines = std::make_shared<std::vector<std::string>>();
    gpufl::Logger logger;
    logger.addSink(std::make_unique<RecordingSink>(lines));
    dictionary.enqueueDisassembly(7, code_object.data(), code_object.size());
    dictionary.flushDisassembly(logger, "amd-inline-call-sites");

    // The fixture's lines; -ffile-prefix-map put it under /src.
    const std::string source =
        "/src/tests/backends/amd/data/inline_call_sites.hip";
    using Lookup = gpufl::DictionaryManager::IsaSourceLookup;
    const auto line_of = [&](const char* function, const uint32_t pc) {
        gpufl::DictionaryManager::IsaSourceLine line;
        EXPECT_EQ(dictionary.lookupAmdIsaSource(7, function, pc, &line),
                  Lookup::Found);
        EXPECT_EQ(line.source_file, source) << function << " +" << pc;
        EXPECT_EQ(line.function_source, source);
        return line.source_line;
    };
    const char* softmax = "_Z11softmaxRowsPKfPfii";
    // Each __syncthreads(): its s_waitcnt and s_barrier.
    EXPECT_EQ(line_of(softmax, 0xc4), 25u);
    EXPECT_EQ(line_of(softmax, 0xc8), 25u);
    EXPECT_EQ(line_of(softmax, 0x108), 28u);
    EXPECT_EQ(line_of(softmax, 0x168), 31u);
    EXPECT_EQ(line_of(softmax, 0x248), 36u);
    EXPECT_EQ(line_of(softmax, 0x268), 39u);
    // Each expf's v_exp_f32, and the wait on fmaxf's load.
    EXPECT_EQ(line_of(softmax, 0x1f0), 34u);
    EXPECT_EQ(line_of(softmax, 0x344), 44u);
    EXPECT_EQ(line_of(softmax, 0x9c), 23u);
    // blockIdx.x and blockDim.x.
    EXPECT_EQ(line_of(softmax, 0x0), 16u);
    EXPECT_EQ(line_of(softmax, 0xcc), 26u);
    // blockSum's own statements keep their lines, and its barrier goes to
    // its __syncthreads(), not to normalize's call of blockSum.
    const char* normalize = "_Z9normalizePf";
    EXPECT_EQ(line_of(normalize, 0x20), 51u);
    EXPECT_EQ(line_of(normalize, 0x2c), 52u);
    EXPECT_EQ(line_of(normalize, 0x30), 54u);

    for (const auto& json : *lines) {
        for (const char* header : {"amd_hip_runtime.h",
                                   "amd_device_functions.h",
                                   "__clang_hip_math.h"}) {
            EXPECT_EQ(json.find(header), std::string::npos)
                << json.substr(0, 200);
        }
    }
}
#endif

// Only the collector disassembles during a run; with no collector, a wait
// for line tables gives up at once instead of holding shutdown.
TEST(AmdIsaLineWait, ReturnsAtOnceWithoutACollector) {
    const auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(gpufl::Monitor::WaitForAmdIsaLines(
        {42}, std::chrono::milliseconds(2000)));
    EXPECT_LT(std::chrono::steady_clock::now() - start,
              std::chrono::milliseconds(500));
    EXPECT_TRUE(
        gpufl::Monitor::WaitForAmdIsaLines({}, std::chrono::milliseconds(0)));
}

}  // namespace
