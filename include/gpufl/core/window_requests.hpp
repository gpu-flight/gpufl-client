#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <set>
#include <string>

#include "gpufl/core/deep_window.hpp"

namespace gpufl::detail {

/**
 * @brief Bounds on the windows other processes may ask this one for.
 *
 * The coordinator already allows one window at a time and a cooldown. These
 * add what a live server needs on top: a cap on one window's length, and an
 * hourly cap on windows and on capture time, so a client that asks too often
 * cannot keep the deep engines armed. Defaults follow the LLM serving plan
 * (8.3): 1 s windows until longer ones qualify, six an hour, ten seconds of
 * capture an hour, a minute between windows.
 */
struct WindowRequestLimits {
    int64_t max_window_ms           = 1000;
    int64_t max_windows_per_hour    = 6;
    int64_t max_capture_ms_per_hour = 10000;
    int64_t cooldown_ms             = 60000;
};

/**
 * @brief The launch-local channel through which another process of the same
 *        launch asks this one for a bounded deep window.
 *
 * The serving signals that decide a window is worth taking live in the API
 * server (the vLLM adapter), and the GPU work runs in EngineCore, a different
 * process. Counters cannot cross processes, so no rule can see both. Instead
 * the adapter drops a request file in
 * `<GPUFL_LOG_DIR>/.gpufl-window-requests/<launch_id>/`, and every GPU process
 * of the launch acts on it once. This is a broadcast, not a claim: at tensor
 * parallel > 1 each worker needs its own window for the same moment.
 *
 * A process without a CUDA context never acts, because the coordinator answers
 * `PreparationPending` until it has one, so the API server leaves requests
 * alone without being told it is not a GPU process.
 *
 * Each process answers in `<request_id>.<pid>.result.json`: `opened` (with the
 * window's token), `busy`, `cooldown`, `budget`, `expired`,
 * `engine_unavailable` or `invalid`. What the window then captured is in the
 * window record, whose trigger carries the request id.
 *
 * Polled from the collector beat; not thread-safe on its own.
 */
class WindowRequestChannel {
   public:
    /// The coordinator calls this channel needs. Tests supply fakes.
    struct Hooks {
        OpenRequestResult (*request_open)(void* ctx, const DeepWindowSpec&) = nullptr;
        uint64_t (*last_opened_token)(void* ctx) = nullptr;
        uint64_t (*pending_open_token)(void* ctx) = nullptr;
        void* ctx = nullptr;
    };

    /** @brief Hooks wired to the real DeepWindow coordinator. */
    static Hooks liveHooks();

    /** @brief The request directory of one launch under a log directory. */
    static std::filesystem::path DirectoryFor(const std::string& log_dir,
                                              const std::string& launch_id);

    WindowRequestChannel(std::filesystem::path dir, int64_t pid,
                         WindowRequestLimits limits, Hooks hooks);

    /** @brief Read new requests and settle queued ones. One collector beat. */
    void poll(int64_t now_ns);

    /**
     * @brief Install the process-wide channel from the environment.
     *
     * Needs `GPUFL_LAUNCH_ID` and a log directory; `GPUFL_WINDOW_REQUESTS=0`
     * turns it off. A malformed limit disables the channel with an error rather
     * than quietly using a default the user never asked for.
     */
    static void InstallFromEnv(const std::string& log_dir);

    /** @brief Poll the installed channel, if any. Call on the collector beat. */
    static void Service();

    static bool Installed();
    static void ResetForTesting();

   private:
    struct Queued {
        uint64_t token = 0;
        int64_t  window_ms = 0;
    };

    void settleQueued(int64_t now_ns);
    void consider(const std::filesystem::path& file, const std::string& request_id,
                  int64_t now_ns);
    bool withinBudget(int64_t window_ms, int64_t now_ns);
    void writeResult(const std::string& request_id, const std::string& status,
                     uint64_t token, int64_t window_ms, int64_t now_ns) const;

    std::filesystem::path dir_;
    int64_t pid_;
    WindowRequestLimits limits_;
    Hooks hooks_;

    /// Requests this process has answered; a request is acted on once.
    std::set<std::string> answered_;
    /// Requests the coordinator accepted and has not opened yet.
    std::map<std::string, Queued> queued_;
    /// When each counted window opened and how long it may run, for the hourly caps.
    std::deque<std::pair<int64_t, int64_t>> opened_;
};

}  // namespace gpufl::detail
