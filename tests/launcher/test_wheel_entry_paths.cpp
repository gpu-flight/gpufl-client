// The `gpufl` a Linux wheel puts on PATH finds the launcher pip installed in
// site-packages. Each case lays out one pip install scheme in a scratch
// directory, the way pip would, and checks which launcher the entry in that
// scheme's scripts directory resolves to.
#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "wheel_entry_paths.hpp"

namespace fs = std::filesystem;

namespace {

class WheelEntryPathsTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        root_ = fs::temp_directory_path() /
                (std::string("gpufl_wheel_entry_") +
                 std::to_string(::getpid()) + "_" + info->name());
        fs::remove_all(root_);
        fs::create_directories(root_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    // A launcher file at root_/rel, executable unless told otherwise.
    fs::path placeLauncher(const fs::path& rel, bool executable = true) const {
        const fs::path path = root_ / rel;
        fs::create_directories(path.parent_path());
        std::ofstream(path) << "#!/bin/sh\n";
        fs::permissions(path,
                        executable ? fs::perms::owner_all
                                   : fs::perms::owner_read |
                                         fs::perms::owner_write,
                        fs::perm_options::replace);
        return path;
    }

    // What the entry installed in root_/scripts would run.
    fs::path resolve(const fs::path& scripts,
                     const std::string& python_dir) const {
        return gpufl::launcher::findWheelLauncher(
            gpufl::launcher::wheelLauncherCandidates(root_ / scripts,
                                                     python_dir));
    }

    fs::path root_;
};

TEST_F(WheelEntryPathsTest, VenvCondaUserAndPrefixInstalls) {
    // <base>/bin and <base>/lib/pythonX.Y/site-packages.
    const fs::path launcher =
        placeLauncher("venv/lib/python3.12/site-packages/gpufl/bin/gpufl");
    EXPECT_EQ(resolve("venv/bin", "python3.12"), launcher);
}

TEST_F(WheelEntryPathsTest, SystemInterpreterWithPlatlibInLib64) {
    // Fedora/RHEL root pip: /usr/local/bin and
    // /usr/local/lib64/python3.12/site-packages.
    const fs::path launcher = placeLauncher(
        "usr/local/lib64/python3.12/site-packages/gpufl/bin/gpufl");
    EXPECT_EQ(resolve("usr/local/bin", "python3.12"), launcher);
}

TEST_F(WheelEntryPathsTest, DebianSystemPipDistPackages) {
    // /usr/local/bin and /usr/local/lib/python3.12/dist-packages.
    const fs::path launcher = placeLauncher(
        "usr/local/lib/python3.12/dist-packages/gpufl/bin/gpufl");
    EXPECT_EQ(resolve("usr/local/bin", "python3.12"), launcher);
}

TEST_F(WheelEntryPathsTest, PipTargetDirectory) {
    // pip install --target DIR: DIR/bin and DIR/gpufl.
    const fs::path launcher = placeLauncher("target/gpufl/bin/gpufl");
    EXPECT_EQ(resolve("target/bin", "python3.12"), launcher);
}

TEST_F(WheelEntryPathsTest, FreeThreadedInterpreterDirectory) {
    const fs::path launcher =
        placeLauncher("venv/lib/python3.13t/site-packages/gpufl/bin/gpufl");
    EXPECT_EQ(resolve("venv/bin", "python3.13t"), launcher);
    EXPECT_TRUE(resolve("venv/bin", "python3.13").empty());
}

TEST_F(WheelEntryPathsTest, NeverRunsAnotherInterpretersLauncher) {
    // ~/.local can hold gpufl for several Python versions; bin/gpufl is
    // whichever installed last, and it must run its own version's launcher.
    placeLauncher("local/lib/python3.12/site-packages/gpufl/bin/gpufl");
    const fs::path own =
        placeLauncher("local/lib/python3.13/site-packages/gpufl/bin/gpufl");
    EXPECT_EQ(resolve("local/bin", "python3.13"), own);
    fs::remove(own);
    EXPECT_TRUE(resolve("local/bin", "python3.13").empty());
}

TEST_F(WheelEntryPathsTest, SkipsAFileThatIsNotExecutable) {
    placeLauncher("venv/lib/python3.12/site-packages/gpufl/bin/gpufl",
                  /*executable=*/false);
    EXPECT_TRUE(resolve("venv/bin", "python3.12").empty());
}

TEST_F(WheelEntryPathsTest, SkipsADirectoryNamedLikeTheLauncher) {
    fs::create_directories(root_ /
                           "venv/lib/python3.12/site-packages/gpufl/bin/gpufl");
    EXPECT_TRUE(resolve("venv/bin", "python3.12").empty());
}

TEST_F(WheelEntryPathsTest, CandidatesAreRelativeToTheScriptsDirectory) {
    // The error message lists these, so they must be the real paths tried.
    const auto candidates = gpufl::launcher::wheelLauncherCandidates(
        "/opt/env/bin", "python3.12");
    ASSERT_FALSE(candidates.empty());
    EXPECT_EQ(candidates.front(),
              fs::path("/opt/env/lib/python3.12/site-packages/gpufl/bin/gpufl"));
    for (const fs::path& candidate : candidates) {
        EXPECT_EQ(candidate.string().rfind("/opt/env/", 0), 0u) << candidate;
        EXPECT_EQ(candidate.filename(), "gpufl") << candidate;
    }
}

}  // namespace
