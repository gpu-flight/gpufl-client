#pragma once

#include "cli_parse.hpp"

namespace gpufl::launcher {

// Runs a `gpufl upload` invocation: resolves creds (flag value, else the
// GPUFL_BACKEND_URL / GPUFL_API_KEY / GPUFL_API_PATH env vars), builds a
// gpufl::UploadOptions, calls gpufl::uploadLogs(), prints a summary, and
// returns the CLI exit code:
//   0 = success (every event uploaded, no warnings)
//   1 = partial success (uploaded, but with warnings)
//   2 = failure (missing creds, auth error, timeout, bad dir, …)
// These match the exit-code contract of the retired Python `gpufl.cli`
// uploader so existing shell pipelines keep working.
int runUpload(const UploadArgs& args);

}  // namespace gpufl::launcher
