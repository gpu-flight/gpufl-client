#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

#include "gpufl/gpufl.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/logger/logger.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/session_bootstrap.hpp"

namespace {

TEST(SessionBootstrapTest, OpensInitialContextAndRemembersReportSource) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() /
        ("gpufl_session_bootstrap_" +
         std::to_string(gpufl::detail::GetPid()));
    std::error_code ec;
    fs::remove_all(root, ec);

    gpufl::Runtime runtime;
    runtime.app_name = "bootstrap-test";
    runtime.session_id = "session-bootstrap";
    runtime.run_id = "run-bootstrap";
    runtime.logger = std::make_shared<gpufl::Logger>();

    gpufl::InitOptions options;
    options.log_path = root.string();
    gpufl::detail::InitialSessionLoggingState state;
    ASSERT_TRUE(gpufl::detail::openInitialSessionLogging(
        runtime, options, /*segmented=*/false, state));

    EXPECT_EQ(state.log_path, root.string());
    EXPECT_EQ(state.options.base_path, root.string());
    EXPECT_EQ(state.options.session_id, "session-bootstrap");
    ASSERT_TRUE(runtime.hasSegmentContext());
    {
        const auto context = runtime.acquireSegmentContext("bootstrap-test");
        ASSERT_TRUE(context);
        EXPECT_EQ(context->session_id, "session-bootstrap");
        EXPECT_EQ(context->segment_index, 0u);
    }

    const auto report_source = gpufl::detail::lastSessionReportSource();
    EXPECT_EQ(report_source.log_path, root.string());
    EXPECT_EQ(report_source.session_id, "session-bootstrap");

    runtime.sealActiveSegmentContext();
    runtime.logger->close();
    fs::remove_all(root, ec);
}

}  // namespace
