// Time-based transport-window rotation (rotate_after_ms), driven by a FAKE
// monotonic clock - no sleeps. Windows are observed where consumers observe
// them: published `<channel>.<N>.log.gz` files in the session dir, plus
// FileLogSink::rotationStats() for WHICH trigger fired (files alone cannot
// say size-vs-time). Contract under test:
//   - a window rotates when the data in it spans >= rotate_after_ms
//   - an EMPTY window never rotates (idle channels publish no empty files)
//   - the window's age starts at its FIRST write, not at channel open
//   - when time and size are due at once, time is recorded (it was due
//     before this write's bytes existed)
//   - a clock that never advances never time-rotates
#include <gtest/gtest.h>

#include <zlib.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "gpufl/core/logger/file_compressor.hpp"
#include "gpufl/core/logger/file_log_sink.hpp"
#include "gpufl/core/logger/log_rotator.hpp"
#include "gpufl/core/logger/log_salvage.hpp"
#include "gpufl/core/logger/session_ownership.hpp"
#include "gpufl/core/logger/window_metadata.hpp"
#include "gpufl/core/logger/log_sink.hpp"
#include "gpufl/core/logger/logger.hpp"

namespace fs = std::filesystem;

namespace {

bool endsWithSuffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(),
                         suffix) == 0;
}

// Records the paths the rotator asks the compressor to write, and can be made
// to fail, so the export TRANSACTION can be pinned without needing a crash:
// production's GzipFileCompressor gives no way to observe which name a
// half-written gzip would have had.
class RecordingCompressor final : public gpufl::IFileCompressor {
   public:
    bool compress(const std::string& path) override {
        compress_calls.push_back(path);
        return succeed;
    }

    bool compressTo(const std::string& src, const std::string& dst) override {
        targets.push_back(dst);
        // A real compressor leaves bytes behind when it dies mid-write, so
        // write first and only then report the failure.
        std::ofstream out(dst, std::ios::binary | std::ios::trunc);
        out << (succeed ? "pretend-gzip-of:" : "half-written-gzip-of:") << src;
        out.close();
        return succeed;
    }

    std::vector<std::string> targets;
    std::vector<std::string> compress_calls;
    bool succeed = true;
};

class FileLogSinkRotationTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        base_ = fs::temp_directory_path() /
                (std::string("gpufl_rotation_test_") + info->name());
        fs::remove_all(base_);
        fs::create_directories(base_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(base_, ec);
    }

    gpufl::Logger::Options options(std::int64_t rotate_after_ms,
                                   std::size_t rotate_bytes = 0) {
        gpufl::Logger::Options o;
        o.base_path = base_.string();
        o.session_id = "s1";
        o.rotate_bytes = rotate_bytes;  // 0 = size trigger off
        o.rotate_after_ms = rotate_after_ms;
        o.now_ms = [this] { return fake_now_ms_; };
        return o;
    }

    fs::path sessionDir() const { return base_ / "s1"; }
    fs::path tmpDir() const { return sessionDir() / ".tmp"; }

    static void writeText(const fs::path& path, const std::string& text) {
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.good());
        out << text;
        out.close();
    }

    static std::string readText(const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    }

    static void writeEmptyFile(const fs::path& path) {
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.close();
    }

    // Decompresses a published window so a test can assert the EVENTS
    // survived, not merely that some file exists at the expected name.
    static std::string gunzipToString(const fs::path& path) {
        gzFile file = gzopen(path.string().c_str(), "rb");
        if (!file) return {};
        std::string out;
        char buffer[8192];
        int read = 0;
        while ((read = gzread(file, buffer, sizeof(buffer))) > 0) {
            out.append(buffer, static_cast<std::size_t>(read));
        }
        gzclose(file);
        return out;
    }

    gpufl::LogRotationOptions rotatorOptions() const {
        gpufl::LogRotationOptions r{};
        r.base_path = base_.string();
        r.session_id = "s1";
        r.channel_name = "device";
        r.compress_rotated = true;
        return r;
    }

    // Published windows for a channel = `<channel>.<N>.log.gz` files in the
    // session ROOT. The active file lives in `.tmp/` and never counts.
    std::size_t publishedWindows(const std::string& channel) const {
        const fs::path session_dir = base_ / "s1";
        if (!fs::exists(session_dir)) return 0;
        std::size_t n = 0;
        for (const auto& e : fs::directory_iterator(session_dir)) {
            if (!e.is_regular_file()) continue;
            const std::string name = e.path().filename().string();
            if (name.rfind(channel + ".", 0) == 0 &&
                name.size() >= 7 &&
                name.compare(name.size() - 7, 7, ".log.gz") == 0) {
                ++n;
            }
        }
        return n;
    }

    fs::path base_;
    std::int64_t fake_now_ms_ = 0;
};

class RawJsonEvent final : public gpufl::IJsonSerializable {
public:
    RawJsonEvent(const gpufl::Channel channel, std::string json)
        : channel_(channel), json_(std::move(json)) {}

    std::string buildJson() const override { return json_; }
    gpufl::Channel channel() const override { return channel_; }

private:
    gpufl::Channel channel_;
    std::string json_;
};

TEST_F(FileLogSinkRotationTest, TimeTriggerPublishesOnceWindowSpanExceeded) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    fake_now_ms_ = 0;
    sink.write(gpufl::Channel::Device, R"({"a":1})");  // window 1 starts
    fake_now_ms_ = 4999;
    sink.write(gpufl::Channel::Device, R"({"a":2})");  // span 4999 < 5000
    EXPECT_EQ(publishedWindows("device"), 0u);
    EXPECT_EQ(sink.rotationStats().by_time, 0u);

    fake_now_ms_ = 5000;  // span reaches the threshold BEFORE this write
    sink.write(gpufl::Channel::Device, R"({"a":3})");  // rotates, then writes
    sink.waitForPendingExports();
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_EQ(sink.rotationStats().by_time, 1u);
    EXPECT_EQ(sink.rotationStats().by_size, 0u);

    // The new window's age starts at ITS first write (5000), not at the
    // rotation or at channel open - no instant re-rotation.
    fake_now_ms_ = 9999;
    sink.write(gpufl::Channel::Device, R"({"a":4})");
    EXPECT_EQ(sink.rotationStats().by_time, 1u);
    fake_now_ms_ = 10000;
    sink.write(gpufl::Channel::Device, R"({"a":5})");
    sink.waitForPendingExports();
    EXPECT_EQ(sink.rotationStats().by_time, 2u);
    EXPECT_EQ(publishedWindows("device"), 2u);
    EXPECT_EQ(sink.rotationStats().published, 2u);
}

TEST_F(FileLogSinkRotationTest, EmptyWindowNeverRotatesNoMatterHowLate) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    // Channel idle far past the threshold: the first write must NOT rotate
    // (an empty window has no age) - it starts window 1 instead.
    fake_now_ms_ = 100000;
    sink.write(gpufl::Channel::Device, R"({"first":true})");
    EXPECT_EQ(publishedWindows("device"), 0u);
    EXPECT_EQ(sink.rotationStats().by_time, 0u);
}

TEST_F(FileLogSinkRotationTest, SizeTriggerStillRotatesAndRecordsSize) {
    // Time trigger OFF; size threshold small enough that the second line
    // would push past it.
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/0,
                                    /*rotate_bytes=*/64));

    const std::string line(40, 'x');
    sink.write(gpufl::Channel::Device, line);           // 41 bytes
    EXPECT_EQ(publishedWindows("device"), 0u);
    sink.write(gpufl::Channel::Device, line);           // 82 > 64 → rotate
    sink.waitForPendingExports();
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_EQ(sink.rotationStats().by_size, 1u);
    EXPECT_EQ(sink.rotationStats().by_time, 0u);
}

TEST_F(FileLogSinkRotationTest,
       ReportsAcceptedSerializedBytesForExactChannelFanout) {
    std::vector<std::uint64_t> reported;
    auto opt = options(/*rotate_after_ms=*/0);
    opt.on_serialized_bytes = [&reported](const std::uint64_t bytes) {
        reported.push_back(bytes);
    };

    gpufl::FileLogSink sink(opt);
    const std::string device_line = R"({"device":1})";
    const std::string shared_line = R"({"shared":true})";

    sink.write(gpufl::Channel::Device, device_line);
    sink.write(gpufl::Channel::All, shared_line);

    ASSERT_EQ(reported.size(), 2u);
    EXPECT_EQ(reported[0], device_line.size() + 1u);
    EXPECT_EQ(reported[1], 4u * (shared_line.size() + 1u));
}

TEST_F(FileLogSinkRotationTest,
       DoesNotReportSerializedBytesForSpoolRejectedWrites) {
    std::vector<std::uint64_t> reported;
    auto opt = options(/*rotate_after_ms=*/0);
    opt.max_spool_bytes = 1;
    opt.min_free_bytes = 0;
    opt.on_serialized_bytes = [&reported](const std::uint64_t bytes) {
        reported.push_back(bytes);
    };

    gpufl::FileLogSink sink(opt);
    sink.write(gpufl::Channel::Device, R"({"event":1})");

    EXPECT_TRUE(sink.rotationStats().spool_saturated);
    EXPECT_TRUE(reported.empty());
}

TEST_F(FileLogSinkRotationTest,
       LoggerBindsSerializedBytesCallbackBeforeFirstWrite) {
    gpufl::Logger logger;
    ASSERT_TRUE(logger.open(options(/*rotate_after_ms=*/0)));

    std::vector<std::uint64_t> reported;
    logger.setSerializedBytesCallbackBeforeFirstWrite(
        [&reported](const std::uint64_t bytes) {
            reported.push_back(bytes);
        });

    const std::string json = R"({"event":"bound"})";
    logger.write(RawJsonEvent(gpufl::Channel::Device, json));

    ASSERT_EQ(reported.size(), 1u);
    EXPECT_EQ(reported[0], json.size() + 1u);
}

TEST_F(FileLogSinkRotationTest,
       ShutdownWindowRecordsTimingWhenTimeRotationIsDisabled) {
    fake_now_ms_ = 100;
    {
        gpufl::FileLogSink sink(options(/*rotate_after_ms=*/0));
        sink.write(gpufl::Channel::Device, R"({"event":1})");
        fake_now_ms_ = 5100;
    }

    const fs::path metadata =
        gpufl::windowMetadataPath(sessionDir(), "device", 1);
    ASSERT_TRUE(fs::exists(metadata));
    std::ifstream input(metadata, std::ios::binary);
    const std::string json(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    EXPECT_NE(json.find(R"("opened_mono_ms":100)"), std::string::npos);
    EXPECT_NE(json.find(R"("closed_mono_ms":5100)"), std::string::npos);
}

TEST_F(FileLogSinkRotationTest,
       SizeRotatedWindowsEachRecordTheirOwnFirstWriteTimestamp) {
    fake_now_ms_ = 100;
    {
        gpufl::FileLogSink sink(options(/*rotate_after_ms=*/0,
                                        /*rotate_bytes=*/64));
        const std::string line(40, 'x');
        sink.write(gpufl::Channel::Device, line);
        fake_now_ms_ = 200;
        sink.write(gpufl::Channel::Device, line);
        sink.waitForPendingExports();
        fake_now_ms_ = 300;
    }

    const auto read_metadata = [](const fs::path& path) {
        std::ifstream input(path, std::ios::binary);
        return std::string(
            (std::istreambuf_iterator<char>(input)),
            std::istreambuf_iterator<char>());
    };
    const std::string first = read_metadata(
        gpufl::windowMetadataPath(sessionDir(), "device", 1));
    const std::string second = read_metadata(
        gpufl::windowMetadataPath(sessionDir(), "device", 2));
    ASSERT_FALSE(first.empty());
    ASSERT_FALSE(second.empty());
    EXPECT_NE(first.find(R"("opened_mono_ms":100)"), std::string::npos);
    EXPECT_NE(first.find(R"("closed_mono_ms":200)"), std::string::npos);
    EXPECT_NE(second.find(R"("opened_mono_ms":200)"), std::string::npos);
    EXPECT_NE(second.find(R"("closed_mono_ms":300)"), std::string::npos);
}

TEST_F(FileLogSinkRotationTest, TimeRecordedWhenBothTriggersDue) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000,
                                    /*rotate_bytes=*/64));

    const std::string line(40, 'x');
    fake_now_ms_ = 0;
    sink.write(gpufl::Channel::Device, line);
    fake_now_ms_ = 6000;  // time overdue AND next write exceeds 64 bytes
    sink.write(gpufl::Channel::Device, line);
    sink.waitForPendingExports();
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_EQ(sink.rotationStats().by_time, 1u);
    EXPECT_EQ(sink.rotationStats().by_size, 0u);
}

TEST_F(FileLogSinkRotationTest, FrozenClockNeverTimeRotates) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    fake_now_ms_ = 42;  // never advances
    for (int i = 0; i < 100; ++i) {
        sink.write(gpufl::Channel::Device, R"({"i":1})");
    }
    EXPECT_EQ(publishedWindows("device"), 0u);
    EXPECT_EQ(sink.rotationStats().by_time, 0u);
    EXPECT_EQ(sink.rotationStats().by_size, 0u);
}

TEST_F(FileLogSinkRotationTest, CloseWithoutWritesPublishesNothing) {
    {
        gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));
        fake_now_ms_ = 100000;  // channels stay empty the whole time
    }  // destructor closes
    EXPECT_EQ(publishedWindows("device"), 0u);
    EXPECT_EQ(publishedWindows("scope"), 0u);
    EXPECT_EQ(publishedWindows("system"), 0u);
    EXPECT_EQ(publishedWindows("sass"), 0u);
}

TEST_F(FileLogSinkRotationTest, CloseExportsTheFinalNonEmptyWindow) {
    {
        gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));
        sink.write(gpufl::Channel::Device, R"({"tail":true})");
    }  // close exports the active window
    EXPECT_EQ(publishedWindows("device"), 1u);
}

// THE deadline case: a channel writes once and goes quiet. The write-path
// trigger alone would hold that window in `.tmp` until the next write or
// shutdown - rotateDueWindows() (the collector beat) must publish it.
TEST_F(FileLogSinkRotationTest, DeadlineRotationPublishesWithoutFurtherWrites) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    fake_now_ms_ = 0;
    sink.write(gpufl::Channel::Device, R"({"only":true})");
    fake_now_ms_ = 10000;
    sink.rotateDueWindows();
    sink.waitForPendingExports();  // no write in between
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_EQ(sink.rotationStats().by_time, 1u);

    // The fresh window is empty - servicing again must not publish an
    // empty file or count another rotation.
    fake_now_ms_ = 100000;
    sink.rotateDueWindows();
    sink.waitForPendingExports();
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_EQ(sink.rotationStats().by_time, 1u);
}

TEST_F(FileLogSinkRotationTest, RotateDueWindowsSkipsEmptyAndFreshWindows) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    // Nothing ever written anywhere: the beat is a global no-op.
    fake_now_ms_ = 50000;
    sink.rotateDueWindows();
    sink.waitForPendingExports();
    EXPECT_EQ(sink.rotationStats().by_time, 0u);

    // A fresh window (age < deadline) is left alone.
    sink.write(gpufl::Channel::Device, R"({"fresh":true})");
    fake_now_ms_ = 54999;
    sink.rotateDueWindows();
    sink.waitForPendingExports();
    EXPECT_EQ(publishedWindows("device"), 0u);
    EXPECT_EQ(sink.rotationStats().by_time, 0u);
}

// Cutover blocked (the retire rename is denied) must NOT count as a
// rotation and must NOT reset the window age - the very next beat retries
// instead of waiting out a fresh rotate_after_ms.
TEST_F(FileLogSinkRotationTest, BlockedCutoverKeepsWindowAgeAndRetries) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    // Sabotage: a DIRECTORY at the retire target `.tmp/device.1.log`
    // makes the cutover rename fail. Index scans skip non-regular files,
    // so the name still resolves to index 1.
    fs::create_directories(tmpDir() / "device.1.log");

    fake_now_ms_ = 0;
    sink.write(gpufl::Channel::Device, R"({"blocked":true})");
    fake_now_ms_ = 6000;
    sink.rotateDueWindows();
    sink.waitForPendingExports();
    EXPECT_EQ(publishedWindows("device"), 0u);
    EXPECT_EQ(sink.rotationStats().cutover_blocked, 1u);
    EXPECT_EQ(sink.rotationStats().by_time, 0u);

    // Clear the blockage; retry WITHOUT advancing the clock. Only a
    // preserved window age lets this rotate immediately.
    fs::remove(tmpDir() / "device.1.log");
    sink.rotateDueWindows();
    sink.waitForPendingExports();
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_EQ(sink.rotationStats().by_time, 1u);
    EXPECT_EQ(sink.rotationStats().cutover_blocked, 1u);
}

// Publish blocked AFTER the window was cut over: it sits in `.tmp`
// staging for the salvage pass. The CUTOVER still counts (the boundary
// really happened and the data is immutable); the export counts staged,
// never published.
TEST_F(FileLogSinkRotationTest, StagedPublishCountsStagedNotPublished) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    // A DIRECTORY at the publish target blocks the final rename only.
    fs::create_directories(sessionDir() / "device.1.log.gz");

    fake_now_ms_ = 0;
    sink.write(gpufl::Channel::Device, R"({"staged":true})");
    fake_now_ms_ = 6000;
    sink.rotateDueWindows();
    sink.waitForPendingExports();  // ~700ms of backoff, on the worker
    EXPECT_EQ(publishedWindows("device"), 0u);
    EXPECT_EQ(sink.rotationStats().by_time, 1u);     // cutover happened
    EXPECT_EQ(sink.rotationStats().staged, 1u);
    EXPECT_EQ(sink.rotationStats().published, 0u);
    EXPECT_TRUE(fs::is_regular_file(tmpDir() / "device.1.log.gz"));

    // The active window restarted: the next window publishes as index 2
    // (staging keeps index 1 reserved - no overwrite).
    sink.write(gpufl::Channel::Device, R"({"next":true})");
    fake_now_ms_ = 12000;
    sink.rotateDueWindows();
    sink.waitForPendingExports();
    EXPECT_TRUE(fs::is_regular_file(sessionDir() / "device.2.log.gz"));
    EXPECT_EQ(sink.rotationStats().published, 1u);
}

// The whole point of the split: the thread that hits the boundary does a
// rename and nothing else. Compression and the publish backoff must NOT
// have run when rotateDueWindows() returns - otherwise the collector beat
// pays for gzip and stops draining the CUPTI ring.
TEST_F(FileLogSinkRotationTest, CutoverReturnsWhileWorkerExportIsBlocked) {
    std::promise<void> worker_entered_promise;
    auto worker_entered = worker_entered_promise.get_future();
    std::promise<void> release_worker_promise;
    const auto release_worker = release_worker_promise.get_future().share();
    bool entered = false;

    auto opt = options(/*rotate_after_ms=*/5000);
    opt.before_retired_export = [&] {
        if (!entered) {
            entered = true;
            worker_entered_promise.set_value();
        }
        release_worker.wait();
    };
    gpufl::FileLogSink sink(opt);

    fake_now_ms_ = 0;
    sink.write(gpufl::Channel::Device, R"({"a":1})");
    fake_now_ms_ = 6000;

    // Run the cutover on a future so the same test also catches a mutation
    // that performs export inline: the worker hook is reached, but the
    // cutover future cannot become ready until the hook is released.
    auto cutover = std::async(std::launch::async,
                              [&] { sink.rotateDueWindows(); });
    const auto entered_status =
        worker_entered.wait_for(std::chrono::seconds(2));
    if (entered_status != std::future_status::ready) {
        // Never strand the async future behind the test latch on a failure.
        release_worker_promise.set_value();
        cutover.wait();
        FAIL() << "retirement worker did not reach the export hook";
        return;
    }
    EXPECT_EQ(cutover.wait_for(std::chrono::milliseconds(500)),
              std::future_status::ready);

    // The worker is still blocked, but the immutable raw file exists and
    // the active channel accepts the next window.
    EXPECT_TRUE(fs::is_regular_file(tmpDir() / "device.1.log"));
    sink.write(gpufl::Channel::Device, R"({"next":true})");
    fake_now_ms_ = 12000;
    sink.rotateDueWindows();  // queue a second window behind the blocked one
    EXPECT_EQ(sink.rotationStats().by_time, 2u);
    EXPECT_EQ(sink.rotationStats().pending_exports, 2u);

    release_worker_promise.set_value();
    cutover.get();
    sink.waitForPendingExports();
    EXPECT_EQ(sink.rotationStats().published, 2u);
    EXPECT_EQ(sink.rotationStats().pending_exports, 0u);
    EXPECT_GE(sink.rotationStats().max_pending_exports, 2u);
    EXPECT_GT(sink.rotationStats().max_pending_export_bytes, 0u);
}

TEST_F(FileLogSinkRotationTest,
       SalvageDropsPartialGzipAndPublishesRawExactlyOnce) {
    fs::create_directories(tmpDir());
    writeText(tmpDir() / "device.1.log", R"({"window":1})");
    writeText(tmpDir() / "device.1.log.gz.part", "incomplete gzip bytes");

    const auto result = gpufl::salvageSessionTempDir(sessionDir());

    EXPECT_EQ(result.deferred, 0u);
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_TRUE(fs::is_regular_file(sessionDir() / "device.1.log.gz"));
    EXPECT_FALSE(fs::exists(sessionDir() / "device.2.log.gz"));
    EXPECT_FALSE(fs::exists(tmpDir()));
}

// The sidecar must name the window the agent will actually see. Salvage
// fingerprints a STAGED file and then renames it, so recording
// `payload.filename()` wrote the staging name ("device.log.gz") while the
// published window was "device.1.log.gz". The agent's isValidFor() compared
// the two, rejected the pair as a contract violation, wrote a durable loss
// marker, and refused to upload - observed on every final segment of a real
// segmented run.
TEST_F(FileLogSinkRotationTest,
       SalvagedWindowMetadataNamesThePublishedFileNotTheStagingFile) {
    fs::create_directories(tmpDir());
    const std::string payload = R"({"window":"active"})";
    writeText(tmpDir() / "device.log", payload);   // un-indexed active file

    const auto result = gpufl::salvageSessionTempDir(sessionDir());

    ASSERT_EQ(result.deferred, 0u);
    ASSERT_EQ(publishedWindows("device"), 1u);
    ASSERT_TRUE(fs::is_regular_file(sessionDir() / "device.1.log.gz"));

    const fs::path sidecar = sessionDir() / ".gpufl-window.device.1.json";
    ASSERT_TRUE(fs::is_regular_file(sidecar));
    const std::string json = readText(sidecar);
    EXPECT_NE(json.find("\"payload_file\":\"device.1.log.gz\""),
              std::string::npos)
        << "sidecar must name the published window; got: " << json;
    EXPECT_EQ(json.find("\"payload_file\":\"device.log.gz\""),
              std::string::npos)
        << "sidecar names the staging file, which the agent will reject";
}

// The other salvage publish path: an UN-INDEXED gzip in `.tmp`
// (`device.log.gz`), which salvage publishes under a freshly assigned index.
// The name split only appears when the staged basename differs from the
// target's - an already-indexed staged file has the same basename in both
// places, so it cannot expose this. The final segment of a real segmented run
// leaves exactly this shape behind.
TEST_F(FileLogSinkRotationTest,
       UnindexedStagedGzipSalvageMetadataNamesThePublishedFile) {
    fs::create_directories(tmpDir());
    const fs::path raw = tmpDir() / "device.log";
    const fs::path staged = tmpDir() / "device.log.gz";
    writeText(raw, R"({"window":"active","payload":"staged before publish"})");
    gpufl::GzipFileCompressor compressor;
    ASSERT_TRUE(compressor.compressTo(raw.string(), staged.string()));
    fs::remove(raw);   // the rotator had already consumed its raw source

    const auto result = gpufl::salvageSessionTempDir(sessionDir());

    ASSERT_EQ(result.deferred, 0u);
    ASSERT_TRUE(fs::is_regular_file(sessionDir() / "device.1.log.gz"));

    const fs::path sidecar = sessionDir() / ".gpufl-window.device.1.json";
    ASSERT_TRUE(fs::is_regular_file(sidecar));
    const std::string json = readText(sidecar);
    EXPECT_NE(json.find("\"payload_file\":\"device.1.log.gz\""),
              std::string::npos)
        << "sidecar must name the published window; got: " << json;
    EXPECT_EQ(json.find("\"payload_file\":\"device.log.gz\""),
              std::string::npos)
        << "sidecar names the staging file, which the agent will reject";
}

TEST_F(FileLogSinkRotationTest,
       PublishedWindowParserRejectsTheUnindexedActiveFilename) {
    std::string channel;
    std::size_t sequence = 99;

    EXPECT_FALSE(gpufl::parsePublishedWindowName(
        "device.log.gz", channel, sequence));
    EXPECT_TRUE(gpufl::parsePublishedWindowName(
        "device.7.log.gz", channel, sequence));
    EXPECT_EQ(channel, "device");
    EXPECT_EQ(sequence, 7u);
}

TEST_F(FileLogSinkRotationTest,
       SalvagePrefersCompletedGzipOverDuplicateRaw) {
    fs::create_directories(tmpDir());
    const fs::path raw = tmpDir() / "device.1.log";
    const fs::path gzip = tmpDir() / "device.1.log.gz";
    writeText(raw, R"({"window":1})");
    gpufl::GzipFileCompressor compressor;
    ASSERT_TRUE(compressor.compressTo(raw.string(), gzip.string()));

    const auto result = gpufl::salvageSessionTempDir(sessionDir());

    EXPECT_EQ(result.deferred, 0u);
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_TRUE(fs::is_regular_file(sessionDir() / "device.1.log.gz"));
    EXPECT_FALSE(fs::exists(sessionDir() / "device.2.log.gz"));
    EXPECT_FALSE(fs::exists(tmpDir()));
}

TEST_F(FileLogSinkRotationTest,
       SalvageRejectsCorruptGzipAndRecoversCompleteRaw) {
    fs::create_directories(tmpDir());
    const fs::path raw = tmpDir() / "device.1.log";
    const fs::path gzip = tmpDir() / "device.1.log.gz";
    writeText(raw, R"({"window":1,"payload":"enough bytes for a gzip"})");
    gpufl::GzipFileCompressor compressor;
    ASSERT_TRUE(compressor.compressTo(raw.string(), gzip.string()));
    const auto complete_size = fs::file_size(gzip);
    ASSERT_GT(complete_size, 8u);
    fs::resize_file(gzip, complete_size - 8);  // crash before gzip trailer

    const auto result = gpufl::salvageSessionTempDir(sessionDir());

    EXPECT_EQ(result.deferred, 0u);
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_TRUE(fs::is_regular_file(sessionDir() / "device.1.log.gz"));
    EXPECT_FALSE(fs::exists(sessionDir() / "device.2.log.gz"));
    EXPECT_FALSE(fs::exists(tmpDir()));
}

TEST_F(FileLogSinkRotationTest,
       SalvageDoesNotRepublishRawWhenSameIndexAlreadyPublished) {
    fs::create_directories(tmpDir());
    const fs::path raw = tmpDir() / "device.1.log";
    writeText(raw, R"({"window":1})");
    gpufl::GzipFileCompressor compressor;
    ASSERT_TRUE(compressor.compressTo(
        raw.string(), (sessionDir() / "device.1.log.gz").string()));

    const auto result = gpufl::salvageSessionTempDir(sessionDir());

    EXPECT_EQ(result.deferred, 0u);
    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_FALSE(fs::exists(sessionDir() / "device.2.log.gz"));
    EXPECT_FALSE(fs::exists(tmpDir()));
}

// zlib reports EOF-on-first-read as a CLEAN read, so a zero-length file used
// to validate as a gzip. A rename publishes a directory entry before the data
// is necessarily durable, so a power loss right after `.part` -> `.gz` can
// leave an empty `.gz` beside its still-complete raw source. Validating it
// deleted the raw as a duplicate and published the empty file as the window:
// a full window lost, silently, with `.tmp` swept clean afterwards.
TEST_F(FileLogSinkRotationTest, SalvageRefusesEmptyGzipAndRecoversTheRaw) {
    fs::create_directories(tmpDir());
    const std::string payload = R"({"window":1,"payload":"must survive"})";
    writeText(tmpDir() / "device.1.log", payload);
    writeEmptyFile(tmpDir() / "device.1.log.gz");
    ASSERT_EQ(fs::file_size(tmpDir() / "device.1.log.gz"), 0u);

    const auto result = gpufl::salvageSessionTempDir(sessionDir());

    EXPECT_EQ(result.deferred, 0u);
    EXPECT_EQ(publishedWindows("device"), 1u);
    // The events themselves must be what got published - not the empty file.
    EXPECT_EQ(gunzipToString(sessionDir() / "device.1.log.gz"), payload);
    EXPECT_FALSE(fs::exists(sessionDir() / "device.2.log.gz"));
    EXPECT_FALSE(fs::exists(tmpDir()));
}

// Same crash window, one step later: the raw was already removed when the
// power went. The events are genuinely gone, but the empty artifact must not
// pin `.tmp` - that directory is the "session still writing" signal, so a
// permanently deferred zero-byte file would leave the session looking
// unfinished to the uploader and the agent forever.
TEST_F(FileLogSinkRotationTest, SalvageDiscardsEmptyGzipWithNoSource) {
    fs::create_directories(tmpDir());
    writeEmptyFile(tmpDir() / "device.1.log.gz");

    const auto result = gpufl::salvageSessionTempDir(sessionDir());

    EXPECT_EQ(result.deferred, 0u);
    EXPECT_EQ(publishedWindows("device"), 0u);
    EXPECT_FALSE(fs::exists(tmpDir()));
    // Terminal, and it must be COUNTED: once the artifact is gone the
    // session looks clean everywhere downstream, so this number is the only
    // thing that can tell anyone the upload has a hole in it.
    EXPECT_EQ(result.lost_windows, 1);
}

// The same loss, reported through the sink that owns the session, because
// FileLogSink::close() is where an in-process run learns about it.
TEST_F(FileLogSinkRotationTest, CloseReportsUnrecoverableWindowsAsLost) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));
    sink.write(gpufl::Channel::Device, R"({"live":true})");
    // A window whose bytes never reached the disk, with its raw already
    // gone: nothing on disk can still yield those events.
    writeEmptyFile(tmpDir() / "device.7.log.gz");

    sink.close();

    EXPECT_EQ(sink.rotationStats().lost_windows, 1u);
}

// Window indices must come from the channel, never from a fresh directory
// scan. nextWindowIndex() scans `.tmp` and the session root, and the export
// worker moves files between exactly those two directories with no lock
// held, so a window in flight can be missed by BOTH scans - the index is
// handed out twice and fs::rename replaces the published window silently.
// Deleting a published window is a deterministic stand-in for that race:
// a filesystem-derived index would drop back to 1 and collide.
TEST_F(FileLogSinkRotationTest, WindowIndicesNeverGoBackwards) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    fake_now_ms_ = 0;
    sink.write(gpufl::Channel::Device, R"({"w":1})");
    fake_now_ms_ = 6000;
    sink.rotateDueWindows();
    sink.waitForPendingExports();
    ASSERT_TRUE(fs::exists(sessionDir() / "device.1.log.gz"));

    // The published window leaves the directory (pruned, uploaded and
    // swept, or an operator moved it) - the allocator must not care.
    fs::remove(sessionDir() / "device.1.log.gz");

    sink.write(gpufl::Channel::Device, R"({"w":2})");
    fake_now_ms_ = 12000;
    sink.rotateDueWindows();
    sink.waitForPendingExports();

    EXPECT_TRUE(fs::exists(sessionDir() / "device.2.log.gz"));
    EXPECT_FALSE(fs::exists(sessionDir() / "device.1.log.gz"));
    EXPECT_EQ(sink.rotationStats().published, 2u);
}

// Shutdown owns the same monotonic index sequence as mid-run rotation. It
// must not re-scan a directory whose acknowledged windows may have gone.
TEST_F(FileLogSinkRotationTest, FinalWindowIndexNeverGoesBackwards) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    fake_now_ms_ = 0;
    sink.write(gpufl::Channel::Device, R"({"w":1})");
    fake_now_ms_ = 6000;
    sink.rotateDueWindows();
    sink.waitForPendingExports();
    ASSERT_TRUE(fs::exists(sessionDir() / "device.1.log.gz"));

    fs::remove(sessionDir() / "device.1.log.gz");
    sink.write(gpufl::Channel::Device, R"({"w":2})");
    sink.close();

    EXPECT_TRUE(fs::exists(sessionDir() / "device.2.log.gz"));
    EXPECT_FALSE(fs::exists(sessionDir() / "device.1.log.gz"));
}

// Defence in depth for the same hazard: even if an index were reused, the
// export must refuse to rename over an existing window rather than destroy
// it. The bytes stay in `.tmp` for salvage to reconcile.
TEST_F(FileLogSinkRotationTest, ExportRefusesToOverwriteAPublishedWindow) {
    RecordingCompressor compressor;
    gpufl::LogFileRotator rotator(rotatorOptions(), &compressor);
    writeText(sessionDir() / "device.1.log.gz", "the already-published window");
    writeText(tmpDir() / "device.1.log", R"({"w":"second"})");

    const auto result = rotator.exportRetiredWindow(1);

    EXPECT_EQ(result,
              gpufl::LogFileRotator::ExportWindowResult::StagedForSalvage);
    // The original window is untouched.
    EXPECT_EQ(fs::file_size(sessionDir() / "device.1.log.gz"),
              std::string("the already-published window").size());
}

// Refusing the overwrite is only half the contract: a later salvage pass must
// not call the staged file a retry duplicate merely because the index matches.
TEST_F(FileLogSinkRotationTest,
       CollisionSurvivesSalvageWhenPayloadsAreDifferent) {
    fs::create_directories(tmpDir());
    const fs::path published_raw = sessionDir() / "published.raw";
    const fs::path staged_raw = tmpDir() / "device.1.log";
    const fs::path published = sessionDir() / "device.1.log.gz";
    writeText(published_raw, R"({"window":"first"})");
    writeText(staged_raw, R"({"window":"second"})");

    gpufl::GzipFileCompressor compressor;
    ASSERT_TRUE(
        compressor.compressTo(published_raw.string(), published.string()));
    fs::remove(published_raw);

    gpufl::LogFileRotator rotator(rotatorOptions(), &compressor);
    ASSERT_EQ(rotator.exportRetiredWindow(1),
              gpufl::LogFileRotator::ExportWindowResult::StagedForSalvage);
    ASSERT_TRUE(fs::exists(tmpDir() / "device.1.log.gz"));

    const auto salvage = gpufl::salvageSessionTempDir(sessionDir());

    EXPECT_GT(salvage.deferred, 0);
    EXPECT_EQ(gunzipToString(published), R"({"window":"first"})");
    EXPECT_EQ(gunzipToString(tmpDir() / "device.1.log.gz"),
              R"({"window":"second"})");
}

// And the cutover half: retiring onto an index that already has a retired
// window would clobber a window waiting to be exported.
TEST_F(FileLogSinkRotationTest, CutoverRefusesToOverwriteARetiredWindow) {
    RecordingCompressor compressor;
    gpufl::LogFileRotator rotator(rotatorOptions(), &compressor);
    writeText(tmpDir() / "device.1.log", "a window awaiting export");
    writeText(tmpDir() / "device.log", "the active window");

    EXPECT_EQ(rotator.retireActiveWindow(1),
              gpufl::LogFileRotator::RetireResult::Blocked);
    EXPECT_EQ(gunzipToString(tmpDir() / "device.1.log"),
              "a window awaiting export");
    EXPECT_TRUE(fs::exists(tmpDir() / "device.log"));
}

// removeOrTruncateFile deliberately leaves a zero-byte husk when it cannot
// unlink. Treating that husk as "the complete raw source" authorises
// deleting a corrupt-but-partly-readable gzip - the window's last copy.
TEST_F(FileLogSinkRotationTest, SalvagePreservesCorruptGzipWhenRawIsEmpty) {
    fs::create_directories(tmpDir());
    const fs::path raw = tmpDir() / "device.1.log";
    const fs::path gzip = tmpDir() / "device.1.log.gz";
    writeText(raw, R"({"window":1,"payload":"enough bytes for a gzip"})");
    gpufl::GzipFileCompressor compressor;
    ASSERT_TRUE(compressor.compressTo(raw.string(), gzip.string()));
    const auto complete_size = fs::file_size(gzip);
    ASSERT_GT(complete_size, 8u);
    fs::resize_file(gzip, complete_size - 8);   // crash before the trailer
    fs::resize_file(raw, 0);                    // truncate-fallback husk

    const auto result = gpufl::salvageSessionTempDir(sessionDir());

    // The only remaining bytes must survive for manual recovery, and the
    // session must stay visibly unfinished rather than silently complete.
    EXPECT_GT(result.deferred, 0);
    EXPECT_EQ(result.lost_windows, 0);
    EXPECT_TRUE(fs::exists(gzip));
    EXPECT_EQ(fs::file_size(gzip), complete_size - 8);
    EXPECT_EQ(publishedWindows("device"), 0u);
}

TEST_F(FileLogSinkRotationTest, TerminalLossMarkerSurvivesLaterSalvagePasses) {
    fs::create_directories(tmpDir());
    writeEmptyFile(tmpDir() / "device.9.log.gz");

    const auto first = gpufl::salvageSessionTempDir(sessionDir());
    ASSERT_EQ(first.lost_windows, 1);
    ASSERT_EQ(gpufl::transportLossMarkerCount(sessionDir()), 1u);
    ASSERT_FALSE(fs::exists(tmpDir()));

    const auto later = gpufl::salvageSessionTempDir(sessionDir());
    EXPECT_EQ(later.lost_windows, 1);
    EXPECT_EQ(gpufl::transportLossMarkerCount(sessionDir()), 1u);
}

// The export transaction itself, pinned at the rotator. Compressing straight
// to `<window>.log.gz` would make a half-written file indistinguishable from
// a finished window after a crash, and no higher-level test notices: the
// salvage fixtures construct their `.part` files by hand, so they stay green
// either way.
TEST_F(FileLogSinkRotationTest, ExportOnlyEverCompressesToAPartFile) {
    RecordingCompressor compressor;
    gpufl::LogFileRotator rotator(rotatorOptions(), &compressor);
    writeText(tmpDir() / "device.1.log", R"({"window":1})");

    const auto result = rotator.exportRetiredWindow(1);

    EXPECT_EQ(result, gpufl::LogFileRotator::ExportWindowResult::Published);
    ASSERT_EQ(compressor.targets.size(), 1u);
    EXPECT_TRUE(endsWithSuffix(compressor.targets[0], ".log.gz.part"))
        << "compressor wrote directly to " << compressor.targets[0];
    EXPECT_TRUE(fs::exists(sessionDir() / "device.1.log.gz"));
    EXPECT_FALSE(fs::exists(tmpDir() / "device.1.log"));
    EXPECT_FALSE(fs::exists(tmpDir() / "device.1.log.gz.part"));
}

// A compressor that dies mid-write must leave the raw source authoritative
// and nothing under the completed name - otherwise salvage would trust the
// stump as a finished window.
TEST_F(FileLogSinkRotationTest, FailedCompressionLeavesNoCompletedGzip) {
    RecordingCompressor compressor;
    compressor.succeed = false;
    gpufl::LogFileRotator rotator(rotatorOptions(), &compressor);
    const std::string payload = R"({"window":1})";
    writeText(tmpDir() / "device.1.log", payload);

    const auto result = rotator.exportRetiredWindow(1);

    EXPECT_EQ(result,
              gpufl::LogFileRotator::ExportWindowResult::DeferredInActive);
    EXPECT_TRUE(fs::exists(tmpDir() / "device.1.log"));
    EXPECT_FALSE(fs::exists(tmpDir() / "device.1.log.gz"));
    EXPECT_FALSE(fs::exists(tmpDir() / "device.1.log.gz.part"));
    EXPECT_EQ(publishedWindows("device"), 0u);
}

// Publishing while the raw source survives is what produces DUPLICATE rows:
// salvage would later hand the leftover raw a fresh index and publish the
// same events a second time. The export must stage instead.
TEST_F(FileLogSinkRotationTest, ExportRefusesToPublishWhileTheRawSurvives) {
    RecordingCompressor compressor;
    gpufl::LogFileRotator rotator(rotatorOptions(), &compressor);
    // A non-empty DIRECTORY at the raw window's path makes both the remove
    // and the truncate fallback fail, portably.
    fs::create_directories(tmpDir() / "device.1.log" / "holder");
    writeText(tmpDir() / "device.1.log" / "holder" / "pin", "x");

    const auto result = rotator.exportRetiredWindow(1);

    EXPECT_EQ(result,
              gpufl::LogFileRotator::ExportWindowResult::StagedForSalvage);
    EXPECT_EQ(publishedWindows("device"), 0u);
    EXPECT_TRUE(fs::exists(tmpDir() / "device.1.log.gz"));
}

// A cut-over window still publishes even if close() arrives while the
// export is queued: close drains the worker before sweeping `.tmp`.
TEST_F(FileLogSinkRotationTest, CloseDrainsPendingExports) {
    {
        gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));
        fake_now_ms_ = 0;
        sink.write(gpufl::Channel::Device, R"({"early":true})");
        fake_now_ms_ = 6000;
        sink.rotateDueWindows();   // no waitForPendingExports here
        sink.write(gpufl::Channel::Device, R"({"late":true})");
    }  // close(): drain worker, then export the final active window
    EXPECT_EQ(publishedWindows("device"), 2u);
    EXPECT_FALSE(fs::exists(tmpDir()));
}

// The writer cannot know whether the backend durably accepted a window.
// It never deletes published data; the agent removes ACKed payloads and
// leaves metadata tombstones.
TEST_F(FileLogSinkRotationTest,
       ClientNeverPrunesPublishedWindowsBeforeAgentAck) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));

    for (int i = 1; i <= 3; ++i) {
        sink.write(gpufl::Channel::Device, R"({"w":1})");
        fake_now_ms_ += 5000;
        sink.rotateDueWindows();
        sink.waitForPendingExports();
    }
    EXPECT_EQ(publishedWindows("device"), 3u);
    EXPECT_EQ(sink.rotationStats().by_time, 3u);
    EXPECT_TRUE(fs::exists(sessionDir() / "device.1.log.gz"));
    EXPECT_TRUE(fs::exists(sessionDir() / "device.2.log.gz"));
    EXPECT_TRUE(fs::exists(sessionDir() / "device.3.log.gz"));
}

TEST_F(FileLogSinkRotationTest,
       SpoolBudgetStopsNewWritesAndPersistsTerminalLoss) {
    auto opt = options(/*rotate_after_ms=*/0);
    opt.flush_always = true;
    opt.max_spool_bytes = 32;
    opt.min_free_bytes = 0;
    gpufl::FileLogSink sink(opt);

    sink.write(gpufl::Channel::Device, std::string(64, 'x'));
    sink.rotateDueWindows();  // collector beat performs the disk check

    const auto saturated = sink.rotationStats();
    ASSERT_TRUE(saturated.spool_saturated);
    EXPECT_GE(saturated.spool_bytes_at_saturation, 32u);
    EXPECT_EQ(gpufl::transportLossMarkerCount(sessionDir()), 1u);

    const auto bytes_before =
        fs::file_size(tmpDir() / "device.log");
    sink.write(gpufl::Channel::Device, R"({"must":"drop"})");
    EXPECT_EQ(fs::file_size(tmpDir() / "device.log"), bytes_before);
    EXPECT_EQ(sink.rotationStats().dropped_events, 2u)
        << "the write that crosses the cap and every later write are both "
           "rejected";
    EXPECT_GT(sink.rotationStats().dropped_bytes, 0u);

    sink.close();
    EXPECT_EQ(gpufl::transportLossMarkerCount(sessionDir()), 1u)
        << "clean shutdown must not erase the durable incomplete-session "
           "signal";
}

TEST_F(FileLogSinkRotationTest,
       FilesystemReserveIsCheckedBeforeTheFirstEvent) {
    std::error_code ec;
    const auto available = fs::space(base_, ec).available;
    ASSERT_FALSE(ec);

    auto opt = options(/*rotate_after_ms=*/0);
    opt.max_spool_bytes = 0;
    opt.min_free_bytes = available;
    gpufl::FileLogSink sink(opt);

    ASSERT_TRUE(sink.rotationStats().spool_saturated);
    sink.write(gpufl::Channel::Device, std::string(1024, 'x'));
    EXPECT_EQ(sink.rotationStats().dropped_events, 1u);
    EXPECT_EQ(fs::file_size(tmpDir() / "device.log"), 0u);
    EXPECT_EQ(gpufl::transportLossMarkerCount(sessionDir()), 1u);
}

TEST_F(FileLogSinkRotationTest,
       ProjectedWriteCannotCrossTheRealFilesystemReserve) {
    constexpr std::uint64_t kMargin = 8ull * 1024 * 1024;
    constexpr std::uint64_t kWrite = 16ull * 1024 * 1024;
    std::error_code ec;
    const auto available = fs::space(base_, ec).available;
    ASSERT_FALSE(ec);
    if (available <= kMargin * 2) {
        GTEST_SKIP() << "test filesystem has too little headroom";
    }

    auto opt = options(/*rotate_after_ms=*/0);
    opt.max_spool_bytes = 0;
    opt.min_free_bytes = available - kMargin;
    gpufl::FileLogSink sink(opt);
    ASSERT_FALSE(sink.rotationStats().spool_saturated)
        << "external disk use consumed the test margin during setup";

    sink.write(
        gpufl::Channel::Device,
        std::string(static_cast<std::size_t>(kWrite), 'x'));

    EXPECT_TRUE(sink.rotationStats().spool_saturated);
    EXPECT_EQ(sink.rotationStats().dropped_events, 1u);
    EXPECT_EQ(fs::file_size(tmpDir() / "device.log"), 0u)
        << "the write that would cross the reserve must be rejected before "
           "touching the active window";
    EXPECT_EQ(gpufl::transportLossMarkerCount(sessionDir()), 1u);
}

TEST_F(FileLogSinkRotationTest, ZeroSpoolLimitsExplicitlyDisableTheGuard) {
    auto opt = options(/*rotate_after_ms=*/0);
    opt.flush_always = true;
    opt.max_spool_bytes = 0;
    opt.min_free_bytes = 0;
    gpufl::FileLogSink sink(opt);

    sink.write(gpufl::Channel::Device, std::string(64, 'x'));
    sink.rotateDueWindows();

    EXPECT_FALSE(sink.rotationStats().spool_saturated);
    EXPECT_EQ(gpufl::transportLossMarkerCount(sessionDir()), 0u);
}

TEST_F(FileLogSinkRotationTest, SessionOwnershipIsExclusiveAndCrashReleased) {
    std::string first_error;
    auto first =
        gpufl::SessionOwnershipLock::tryAcquire(sessionDir(), &first_error);
    ASSERT_NE(first, nullptr) << first_error;

    std::string second_error;
    auto second =
        gpufl::SessionOwnershipLock::tryAcquire(sessionDir(), &second_error);
    EXPECT_EQ(second, nullptr);
    EXPECT_NE(second_error.find("owned by another live process"),
              std::string::npos);

    first.reset();
    auto after_release =
        gpufl::SessionOwnershipLock::tryAcquire(sessionDir(), &second_error);
    EXPECT_NE(after_release, nullptr) << second_error;
}

TEST_F(FileLogSinkRotationTest, SalvageNeverTouchesALiveOwnedSession) {
    fs::create_directories(sessionDir() / ".tmp");
    writeText(sessionDir() / ".tmp" / "device.1.log",
              "payload-from-live-writer\n");

    auto owner = gpufl::SessionOwnershipLock::tryAcquire(sessionDir());
    ASSERT_NE(owner, nullptr);

    const auto while_live = gpufl::salvageSessionTempDir(sessionDir());
    EXPECT_EQ(while_live.active_sessions_skipped, 1);
    EXPECT_EQ(while_live.salvaged, 0);
    EXPECT_TRUE(fs::exists(sessionDir() / ".tmp" / "device.1.log"));
    EXPECT_FALSE(fs::exists(sessionDir() / "device.1.log.gz"));

    owner.reset();
    const auto after_exit = gpufl::salvageSessionTempDir(sessionDir());
    EXPECT_EQ(after_exit.active_sessions_skipped, 0);
    EXPECT_EQ(after_exit.salvaged, 1);
    EXPECT_TRUE(fs::exists(sessionDir() / "device.1.log.gz"));
}

TEST_F(FileLogSinkRotationTest, PublishedWindowHasImmutableIdentityMetadata) {
    {
        gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));
        fake_now_ms_ = 100;
        sink.write(gpufl::Channel::Device, R"({"event":1})");
        fake_now_ms_ = 5100;
        sink.rotateDueWindows();
        sink.waitForPendingExports();
    }

    const fs::path metadata =
        gpufl::windowMetadataPath(sessionDir(), "device", 1);
    ASSERT_TRUE(fs::exists(metadata));
    std::ifstream input(metadata, std::ios::binary);
    const std::string json(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    EXPECT_NE(json.find(R"("type":"transport_window")"),
              std::string::npos);
    EXPECT_NE(json.find(R"("window_sequence":1)"),
              std::string::npos);
    EXPECT_NE(json.find(R"("opened_mono_ms":100)"),
              std::string::npos);
    EXPECT_NE(json.find(R"("closed_mono_ms":5100)"),
              std::string::npos);
    EXPECT_NE(json.find(R"("payload_crc32":)"), std::string::npos);
}

TEST_F(FileLogSinkRotationTest,
       MetadataFailureKeepsPayloadStagedAndInvisibleToTheAgent) {
    // A directory at the immutable sidecar name is a deterministic,
    // cross-platform publication failure. It must not be mistaken for an
    // already-published metadata file or silently downgrade this window to
    // the legacy, non-idempotent upload path.
    const fs::path metadata =
        gpufl::windowMetadataPath(sessionDir(), "device", 1);
    ASSERT_TRUE(fs::create_directories(metadata));

    {
        gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));
        fake_now_ms_ = 100;
        sink.write(gpufl::Channel::Device, R"({"event":1})");
        fake_now_ms_ = 5100;
        sink.rotateDueWindows();
        sink.waitForPendingExports();
    }

    EXPECT_FALSE(fs::exists(sessionDir() / "device.1.log.gz"))
        << "a current-client payload must never become visible without its "
           "identity sidecar";
    EXPECT_TRUE(fs::exists(tmpDir() / "device.1.log.gz"))
        << "the complete payload is recoverable once metadata publication "
           "can succeed";
    EXPECT_EQ(gpufl::transportLossMarkerCount(sessionDir()), 0u)
        << "metadata publication failure is deferred, not data loss";
}

TEST_F(FileLogSinkRotationTest,
       FailedPayloadPublishStillLeavesIdentityAheadOfTheStagedWindow) {
    // Occupy the final payload name so the no-replace publish fails after
    // metadata creation. This pins the ordering contract without racing a
    // polling thread: moving metadata after publish makes this test fail.
    std::promise<void> worker_entered_promise;
    auto worker_entered = worker_entered_promise.get_future();
    std::promise<void> release_worker_promise;
    auto release_worker = release_worker_promise.get_future().share();
    auto opt = options(/*rotate_after_ms=*/5000);
    opt.before_retired_export = [&] {
        worker_entered_promise.set_value();
        release_worker.wait();
    };

    {
        gpufl::FileLogSink sink(std::move(opt));
        fake_now_ms_ = 100;
        sink.write(gpufl::Channel::Device, R"({"event":"new"})");
        fake_now_ms_ = 5100;
        sink.rotateDueWindows();
        ASSERT_EQ(worker_entered.wait_for(std::chrono::seconds(2)),
                  std::future_status::ready);
        writeText(sessionDir() / "device.1.log.gz", "older-window");
        release_worker_promise.set_value();
        sink.waitForPendingExports();
    }

    EXPECT_TRUE(fs::exists(
        gpufl::windowMetadataPath(sessionDir(), "device", 1)));
    EXPECT_TRUE(fs::exists(tmpDir() / "device.1.log.gz"));
}

TEST_F(FileLogSinkRotationTest,
       ExistingMetadataWithDifferentPayloadRefusesSequenceReuse) {
    const fs::path staged = tmpDir() / "device.1.log";
    writeText(staged, "first-payload");
    ASSERT_TRUE(gpufl::ensureWindowMetadata(
        sessionDir(), "s1", "device", 1, staged));

    writeText(staged, "different-payload");
    gpufl::LogFileRotator rotator(rotatorOptions(), nullptr);
    EXPECT_EQ(
        rotator.exportRetiredWindow(1),
        gpufl::LogFileRotator::ExportWindowResult::StagedForSalvage);
    EXPECT_FALSE(fs::exists(sessionDir() / "device.1.log"))
        << "an immutable identity must never be rebound to different bytes";
    EXPECT_TRUE(fs::exists(staged));
}

TEST_F(FileLogSinkRotationTest,
       MetadataTombstonePreventsSequenceReuseAfterPayloadDeletion) {
    {
        gpufl::FileLogSink sink(options(/*rotate_after_ms=*/5000));
        sink.write(gpufl::Channel::Device, R"({"event":1})");
    }
    ASSERT_TRUE(fs::exists(
        gpufl::windowMetadataPath(sessionDir(), "device", 1)));
    ASSERT_TRUE(fs::remove(sessionDir() / "device.1.log.gz"));

    gpufl::GzipFileCompressor compressor;
    gpufl::LogFileRotator rotator(rotatorOptions(), &compressor);
    EXPECT_EQ(rotator.nextWindowIndex(), 2u)
        << "ACK cleanup may delete the payload, but its metadata tombstone "
           "must keep the sequence consumed.";
}

// A SHORT session - one write per channel, no rotation, clean close - must
// leave a FINISHED session: every channel's single window published in the
// session root with its metadata sidecar, and `.tmp` gone. This is the
// `gpufl trace` zero/short-kernel shape (a sub-second host-only target): on
// Windows it regressed to salvaged=0/deferred=4 at close, with all four
// windows stranded in `.tmp` and the lock left behind, so the session was
// never uploadable.
TEST_F(FileLogSinkRotationTest,
       CleanCloseOfShortSessionPublishesEveryChannelWindow) {
    gpufl::FileLogSink sink(options(/*rotate_after_ms=*/0));

    sink.write(gpufl::Channel::Device, R"({"type":"job_start"})");
    sink.write(gpufl::Channel::Scope, R"({"type":"nvtx_marker_event"})");
    sink.write(gpufl::Channel::System, R"({"type":"system_start"})");
    sink.write(gpufl::Channel::Sass, R"({"type":"dictionary_update"})");
    sink.close();

    EXPECT_EQ(publishedWindows("device"), 1u);
    EXPECT_EQ(publishedWindows("scope"), 1u);
    EXPECT_EQ(publishedWindows("system"), 1u);
    EXPECT_EQ(publishedWindows("sass"), 1u);
    // The sidecar is the immutable identity the agent uploads against.
    EXPECT_TRUE(fs::exists(sessionDir() / ".gpufl-window.device.1.json"));
    // A finished session leaves no staging directory behind.
    EXPECT_FALSE(fs::exists(tmpDir()));
}

// The launcher's post-run sweep must finish a session another process left
// stranded: valid staged gzips in `.tmp`, NO metadata sidecars yet, and a
// stale `.gpufl-session.lock` file whose OS lock died with its process. This
// is byte-for-byte the state a short `gpufl trace` run left behind on
// Windows, where two consecutive sweeps salvaged nothing.
TEST_F(FileLogSinkRotationTest, SweepSalvagesAStrandedSessionWithStaleLock) {
    const fs::path session = sessionDir();
    const fs::path tmp = tmpDir();
    fs::create_directories(tmp);
    writeText(session / ".gpufl-session.lock", "");

    const std::vector<std::string> channels = {"device", "scope", "system",
                                               "sass"};
    for (const auto& channel : channels) {
        const fs::path gz = tmp / (channel + ".1.log.gz");
        gzFile file = gzopen(gz.string().c_str(), "wb");
        ASSERT_NE(file, nullptr);
        const std::string line = R"({"type":"job_start"})" "\n";
        ASSERT_GT(gzwrite(file, line.data(),
                          static_cast<unsigned>(line.size())), 0);
        ASSERT_EQ(gzclose(file), Z_OK);
    }

    const auto result = gpufl::salvageSessionTempDir(session);

    EXPECT_EQ(result.active_sessions_skipped, 0);
    EXPECT_EQ(result.deferred, 0);
    EXPECT_EQ(result.salvaged, 4);
    for (const auto& channel : channels) {
        EXPECT_TRUE(fs::exists(session / (channel + ".1.log.gz")))
            << channel << " window was not published";
        EXPECT_TRUE(fs::exists(gpufl::windowMetadataPath(session, channel, 1)))
            << channel << " sidecar missing";
    }
    EXPECT_FALSE(fs::exists(tmp));
}

// Windows MAX_PATH regression: in a deep output directory every PUBLISHED
// name fit under 260 chars but the metadata STAGING name - published name
// plus ".part." plus a full 36-char UUID - did not. std::filesystem uses
// extended-length paths transparently, the CRT behind std::ofstream does
// not, so exactly that one file failed (errno=2) and every window of every
// channel was stranded in `.tmp` with nothing loud in the log. The staging
// suffix is now truncated; this test pins the boundary: a sidecar path
// around 230 chars must publish even though the OLD staging name (+42)
// would have crossed 260.
TEST_F(FileLogSinkRotationTest, DeepSessionDirectoryStillPublishesMetadata) {
    const std::string sidecar_name = ".gpufl-window.device.1.json";
    const std::size_t target_sidecar_len = 230;
    const std::size_t base_len = base_.string().size();
    // Session path must reach target_sidecar_len minus separator and name.
    if (base_len + 40 > target_sidecar_len - sidecar_name.size() - 1) {
        GTEST_SKIP() << "temp base path too long to stage this layout";
    }
    const std::size_t pad =
        target_sidecar_len - sidecar_name.size() - 1 - base_len - 1;
    const fs::path session = base_ / std::string(pad, 'd');
    const fs::path payload_dir = session / ".tmp";
    fs::create_directories(payload_dir);
    ASSERT_EQ(gpufl::windowMetadataPath(session, "device", 1)
                  .string()
                  .size(),
              target_sidecar_len);

    const fs::path payload = payload_dir / "device.1.log.gz";
    writeText(payload, "fingerprint-me");

    EXPECT_TRUE(gpufl::ensureWindowMetadata(session, "s1", "device", 1,
                                            payload, "device.1.log.gz"));
    EXPECT_TRUE(fs::exists(gpufl::windowMetadataPath(session, "device", 1)));
}

// Wiring: the collector beat calls Logger::rotateDueWindows(), which must
// reach every sink exactly once. (Monitor's 250 ms beat → Logger is closed
// by the 3090 sparse-channel run.)
TEST(LoggerRotateDueWindowsTest, ForwardsToEverySinkExactlyOnce) {
    class CountingSink : public gpufl::ILogSink {
       public:
        void write(gpufl::Channel, std::string_view) override {}
        void close() override {}
        void rotateDueWindows() override { ++calls; }
        int calls = 0;
    };

    gpufl::Logger logger;
    auto first = std::make_unique<CountingSink>();
    auto second = std::make_unique<CountingSink>();
    CountingSink* first_raw = first.get();
    CountingSink* second_raw = second.get();
    logger.addSink(std::move(first));
    logger.addSink(std::move(second));

    logger.rotateDueWindows();
    EXPECT_EQ(first_raw->calls, 1);
    EXPECT_EQ(second_raw->calls, 1);

    logger.rotateDueWindows();
    EXPECT_EQ(first_raw->calls, 2);
    EXPECT_EQ(second_raw->calls, 2);
}

}  // namespace
