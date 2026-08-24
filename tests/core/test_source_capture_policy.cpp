#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gpufl/core/common.hpp"
#include "gpufl/core/dictionary_manager.hpp"
#include "gpufl/core/logger/log_sink.hpp"
#include "gpufl/core/logger/logger.hpp"
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

}  // namespace
