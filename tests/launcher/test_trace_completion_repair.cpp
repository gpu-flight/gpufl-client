// Completion repair after the direct child exits. A session whose ownership
// lock is still held belongs to a process that outlived the target (a worker
// it started) and must not be marked finished; once released, it is.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "gpufl/core/logger/session_ownership.hpp"
#include "trace_command_common.hpp"

namespace fs = std::filesystem;

namespace {

class TraceCompletionRepairTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        root_ = fs::temp_directory_path() /
                (std::string("gpufl_completion_repair_") + info->name());
        fs::remove_all(root_);
        fs::create_directories(root_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    fs::path makeSession(const std::string& session_id) const {
        const fs::path dir = root_ / session_id;
        fs::create_directories(dir);
        std::ofstream(dir / "system.1.log")
            << R"({"type":"job_start","pid":7,"app":"target","ts_ns":100})"
            << '\n';
        return dir;
    }

    fs::path root_;
};

bool hasShutdown(const fs::path& session_dir) {
    for (const auto& entry : fs::directory_iterator(session_dir)) {
        if (entry.path().extension() != ".log") continue;
        std::ifstream in(entry.path());
        std::string line;
        while (std::getline(in, line)) {
            if (line.find("\"type\":\"shutdown\"") != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

TEST_F(TraceCompletionRepairTest, LeavesASessionALiveProcessStillOwns) {
    const fs::path live = makeSession("live-session");
    const fs::path ended = makeSession("ended-session");
    auto owner = gpufl::SessionOwnershipLock::tryAcquire(live);
    ASSERT_NE(owner, nullptr);

    gpufl::launcher::SyntheticShutdownContext context;
    context.window_stopped = true;
    const auto repair =
        gpufl::launcher::ensureTraceCompletionMarkers(root_, 0, context);

    EXPECT_EQ(repair.synthesized, 1);
    ASSERT_EQ(repair.live_sessions.size(), 1u);
    EXPECT_EQ(repair.live_sessions[0], "live-session");
    EXPECT_FALSE(hasShutdown(live));
    EXPECT_TRUE(hasShutdown(ended));

    owner.reset();
    const auto after_release =
        gpufl::launcher::ensureTraceCompletionMarkers(root_, 0, context);

    EXPECT_EQ(after_release.synthesized, 1);
    EXPECT_TRUE(after_release.live_sessions.empty());
    EXPECT_TRUE(hasShutdown(live));
}

}  // namespace
