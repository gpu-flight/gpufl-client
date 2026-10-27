// The bounded-window stop reaches the target's whole process group: a worker
// the target started must not outlive the window. Linux only (/proc).
#include <gtest/gtest.h>

#if defined(__linux__)

#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "trace_command.hpp"

namespace fs = std::filesystem;

namespace {

// Gone, or a zombie nobody reaped yet: either way it no longer runs.
bool processEnded(const pid_t pid) {
    std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
    std::string line;
    if (!std::getline(stat, line)) return true;
    const auto comm_end = line.rfind(')');
    if (comm_end == std::string::npos || comm_end + 2 >= line.size()) return true;
    const char state = line[comm_end + 2];
    return state == 'Z' || state == 'X';
}

TEST(TraceProcessGroupTest, DeadlineStopReachesProcessesTheTargetStarted) {
    const fs::path pid_file =
        fs::temp_directory_path() /
        ("gpufl_group_stop_" + std::to_string(::getpid()));
    fs::remove(pid_file);

    gpufl::launcher::RunOptions opts;
    opts.run_ms = 500;
    opts.stop_grace_ms = 2000;
    // sh starts a worker and waits on it. Stopping sh alone leaves the worker.
    const auto result = gpufl::launcher::runPosixTraceProcess(
        {"/bin/sh", "-c",
         "sleep 60 & echo $! > '" + pid_file.string() + "'; wait"},
        opts);

    EXPECT_FALSE(result.launcher_error) << result.error;
    EXPECT_TRUE(result.window_stopped);
    pid_t worker = 0;
    std::ifstream(pid_file) >> worker;
    fs::remove(pid_file);
    ASSERT_GT(worker, 0) << "the target never reported its worker";

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!processEnded(worker) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const bool ended = processEnded(worker);
    if (!ended) ::kill(worker, SIGKILL);
    EXPECT_TRUE(ended) << "worker " << worker << " outlived the window stop";
}

}  // namespace

#endif  // __linux__
