#pragma once

#include <memory>
#include <optional>
#include <iterator>
#include <vector>

#include "gpufl/backends/nvidia/engine/profiling_engine.hpp"
#include "gpufl/core/debug_logger.hpp"

namespace gpufl {

/**
 * @brief Generalized composite: runs an ARBITRARY set of profiling sub-engines
 * in one process.
 *
 * Generalizes the old fixed `PcSamplingWithSassEngine`. Two uses:
 *   (1) measure which engine combinations actually coexist (the compatibility
 *       matrix), and
 *   (2) back the redefined `Deep` = the maximal validated-compatible set.
 *
 * Unlike the old composite it does **no** SASS-vs-PC arbitration: it simply
 * starts every sub-engine and lets each arm-or-decline, so a combo's real
 * behavior (coexist / one-declines / deadlock) is directly observable from the
 * per-engine capability the backend emits.
 *
 * "Trace" (CUPTI activity records — kernel / memcpy / sync) is NOT a sub-engine
 * here: it's the activity-record layer that `CuptiBackend` enables when the combo
 * includes Trace (`collectsKernelEvents()`). This composite holds only the
 * API-driven engines (PcSampling / SassMetrics / PmSampling / RangeProfiler).
 *
 * Lifecycle ordering: sub-engines are driven in a FIXED forward order for
 * start, stop, and shutdown alike (the same convention the old composite used —
 * NOT construct/reverse-destruct). The teardown-safety rule (a Profiler-API
 * engine — SASS / Range — must be disabled BEFORE the PC-Sampling API) is
 * satisfied by the backend supplying the list already ordered with PcSampling
 * LAST. Forward-order stop then disables SASS/Range before PC.
 */
class CompositeEngine final : public IProfilingEngine {
   public:
    explicit CompositeEngine(std::vector<std::unique_ptr<IProfilingEngine>> engines)
        : engines_(std::move(engines)) {}
    ~CompositeEngine() override = default;

    const char* name() const override { return "Composite"; }

    bool initialize(const MonitorOptions& opts, const EngineContext& ctx) override {
        for (auto& e : engines_) {
            if (!e) continue;
            const bool ok = e->initialize(opts, ctx);
            GFL_LOG_DEBUG("[Composite] initialize ", e->name(), ok ? " ok" : " failed");
        }
        return true;
    }

    void start() override {
        for (auto& e : engines_) if (e) e->start();
    }
    void stop() override {
        for (auto& e : engines_) if (e) e->stop();
    }
    void shutdown() override {
        for (auto& e : engines_) if (e) e->shutdown();
    }

    void onScopeStart(const char* n) override {
        for (auto& e : engines_) if (e) e->onScopeStart(n);
    }
    void onScopeStop(const char* n) override {
        for (auto& e : engines_) if (e) e->onScopeStop(n);
    }
    void onPerfScopeStart(const char* n) override {
        for (auto& e : engines_) if (e) e->onPerfScopeStart(n);
    }
    void onPerfScopeStop(const char* n) override {
        for (auto& e : engines_) if (e) e->onPerfScopeStop(n);
    }
    void drainData() override {
        for (auto& e : engines_) if (e) e->drainData();
    }
    void flushBeforeCudaTeardown(const char* reason) override {
        for (auto& e : engines_) if (e) e->flushBeforeCudaTeardown(reason);
    }
    void onLaunchTick() override {
        for (auto& e : engines_) if (e) e->onLaunchTick();
    }

    std::optional<PerfMetricEvent> takeLastPerfEvent() override {
        for (auto& e : engines_)
            if (e) { if (auto ev = e->takeLastPerfEvent()) return ev; }
        return std::nullopt;
    }
    std::vector<KernelPerfMetricEvent> takeKernelPerfEvents() override {
        std::vector<KernelPerfMetricEvent> out;
        for (auto& e : engines_) {
            if (!e) continue;
            auto events = e->takeKernelPerfEvents();
            out.insert(out.end(),
                       std::make_move_iterator(events.begin()),
                       std::make_move_iterator(events.end()));
        }
        return out;
    }
    bool hasInsufficientPrivileges() const override {
        for (auto& e : engines_) if (e && e->hasInsufficientPrivileges()) return true;
        return false;
    }
    bool stallReasonsUnavailable() const override {
        for (auto& e : engines_) if (e && e->stallReasonsUnavailable()) return true;
        return false;
    }
    bool isOperational() const override {
        for (auto& e : engines_) if (e && e->isOperational()) return true;
        return false;
    }
    bool producedData() const override {
        for (auto& e : engines_) if (e && e->producedData()) return true;
        return false;
    }

    /** The sub-engines, for per-engine capability reporting (which armed / produced). */
    const std::vector<std::unique_ptr<IProfilingEngine>>& engines() const { return engines_; }

   private:
    std::vector<std::unique_ptr<IProfilingEngine>> engines_;
};

}  // namespace gpufl
