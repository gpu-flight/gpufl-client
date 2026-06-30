#pragma once

#include <string>
#include <vector>

namespace gpufl::launcher {

struct AgentOptions {
    std::string source_folders;
    std::string log_types;
    std::string cursor_file;
    std::string backend_url;
    std::string api_key;
    std::string api_version = "v1";
    std::string agent_jar;
    // Scope a launcher-spawned --upload agent to THIS run: upload only sessions that
    // appear after it starts, ignoring old sessions in a reused --output dir.
    bool scope_to_new_sessions = false;
    // One-shot upload: exit immediately if no session is found at startup instead of
    // waiting for one to appear (`gpufl upload`, not trace/monitor).
    bool exit_if_empty = false;
};

struct AgentLaunchPlan {
    std::vector<std::string> command;
    std::string description;
};

class AgentProcess {
public:
    AgentProcess() = default;
    AgentProcess(const AgentProcess&) = delete;
    AgentProcess& operator=(const AgentProcess&) = delete;
    ~AgentProcess();

    bool start(const std::vector<std::string>& command, std::string& error);
    void stop();
    bool waitForExit(int timeoutMs);
    bool isRunning() const { return running_; }

private:
#ifdef _WIN32
    void* process_ = nullptr;
    void* thread_ = nullptr;
#else
    int pid_ = -1;
#endif
    bool running_ = false;
};

std::string envOrEmpty(const char* name);
std::string resolveOption(const std::string& flag, const char* env_name);

bool configureAgentEnvironment(const AgentOptions& opts, std::string& error);
bool buildAgentLaunchPlan(const AgentOptions& opts, AgentLaunchPlan& plan,
                          std::string& error);

}  // namespace gpufl::launcher
