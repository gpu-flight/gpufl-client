#pragma once

#include "cli_parse.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace gpufl::launcher {

namespace fs = std::filesystem;

struct TraceProcessResult {
    int rc = 1;
    bool signaled = false;
    int signal = 0;
    bool launcher_error = false;
    std::string error;
};

class TracePlatform {
   public:
    virtual ~TracePlatform() = default;

    virtual const char* platformName() const = 0;
    virtual const char* injectLibraryName() const = 0;

    virtual fs::path selfExe() const = 0;
    virtual std::vector<fs::path> injectLibCandidates(const fs::path& exe) const = 0;
    virtual fs::path defaultOutputDir(const std::string& tag) const = 0;
    virtual std::string defaultAppName(const std::string& command0) const = 0;

    virtual bool setEnv(const char* key, const std::string& value,
                        std::string& error) const = 0;
    virtual bool prepareInjectionEnv(const fs::path& inject_lib,
                                     std::string& error) const = 0;
    virtual TraceProcessResult runProcess(
            const std::vector<std::string>& command) const = 0;
};

int runTraceCommon(const TraceArgs& args, const TracePlatform& platform);

}  // namespace gpufl::launcher
