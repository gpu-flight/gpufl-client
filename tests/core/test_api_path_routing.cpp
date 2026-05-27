// Tests for normalizeApiPath() — the pure helper that canonicalizes
// the api_path field on InitOptions / UploadOptions.
//
// Previously this file also covered HttpLogSink's URL-routing and
// version-header behavior. HttpLogSink was removed in favor of
// gpufl::uploadLogs() (post-shutdown deferred upload) — the equivalent
// end-to-end URL-routing coverage now lives in tests/upload/
// test_upload_logs.cpp (added with the deferred-upload work).

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "gpufl/core/version.hpp"

TEST(NormalizeApiPath, EmptyFallsBackToDefault) {
    EXPECT_EQ(gpufl::normalizeApiPath(""), gpufl::kDefaultApiPath);
}

TEST(NormalizeApiPath, BareRootFallsBackToDefault) {
    EXPECT_EQ(gpufl::normalizeApiPath("/"), gpufl::kDefaultApiPath);
}

TEST(NormalizeApiPath, MissingLeadingSlashIsPrepended) {
    EXPECT_EQ(gpufl::normalizeApiPath("api/v1"), "/api/v1");
    EXPECT_EQ(gpufl::normalizeApiPath("custom/v2"), "/custom/v2");
}

TEST(NormalizeApiPath, TrailingSlashesAreStripped) {
    EXPECT_EQ(gpufl::normalizeApiPath("/api/v1/"), "/api/v1");
    EXPECT_EQ(gpufl::normalizeApiPath("/api/v1///"), "/api/v1");
}

TEST(NormalizeApiPath, AlreadyCanonicalIsUnchanged) {
    EXPECT_EQ(gpufl::normalizeApiPath("/api/v1"), "/api/v1");
    EXPECT_EQ(gpufl::normalizeApiPath("/profiler/api/v1"), "/profiler/api/v1");
}

TEST(NormalizeApiPath, ResultAlwaysHasLeadingSlashNoTrailing) {
    for (const auto& in : std::vector<std::string>{
            "api", "/api", "api/", "/api/", "x/y/z", "/x/y/z/"}) {
        const auto out = gpufl::normalizeApiPath(in);
        ASSERT_FALSE(out.empty());
        EXPECT_EQ(out.front(), '/') << "input='" << in << "' out='" << out << "'";
        if (out.size() > 1) {
            EXPECT_NE(out.back(), '/') << "input='" << in << "' out='" << out << "'";
        }
    }
}
