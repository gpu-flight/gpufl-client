#pragma once

#include <memory>

namespace gpufl {
struct InitOptions;
struct Runtime;
struct SegmentContext;

namespace detail {

/**
 * Startup transaction for one public gpufl::init() invocation.
 *
 * It owns only state that exists while a runtime is being constructed. The
 * process-wide InitOptions remain in g_opts because normal runtime APIs read
 * them after startup; Runtime takes ownership only once logging is open and
 * its first immutable SegmentContext has been published.
 */
class ClientStartup {
public:
    explicit ClientStartup(InitOptions& active_options);
    ~ClientStartup();

    // Starts through continuous-sampling activation. The public init() wrapper
    // remains responsible for the NVTX guard and deep-window installation that
    // must happen after CUPTI has finished wiring its injection table.
    bool start();

private:
    bool resolveConfiguration();
    bool createRuntime();
    void launchVersionProbe() const;
    bool startMonitor();
    bool activateRuntime();
    void configureCollectors(Runtime& runtime) const;
    void emitInitialEvent(Runtime& runtime, const SegmentContext& segment);
    bool startSegmentRuntime(Runtime& runtime);
    void configureSampler(Runtime& runtime) const;
    void startContinuousSampling(Runtime& runtime,
                                 const SegmentContext& segment) const;

    InitOptions& options_;
    std::unique_ptr<Runtime> pending_runtime_;
    bool segmented_ = false;

    // Declared in the implementation because these domain types are only
    // meaningful during startup. Keeping them there prevents this coordinator
    // header from becoming another aggregate dependency hub.
    class State;
    std::unique_ptr<State> state_;
};

}  // namespace detail
}  // namespace gpufl
