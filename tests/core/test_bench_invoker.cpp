// BenchInvoker / GFL_BENCH tests (1.0.3+).
//
// BenchInvoker is the template-lambda helper behind the GFL_BENCH macro.
// It runs the lambda body `meta.warmup` times BEFORE opening a
// ScopedMonitor scope and `meta.repeat` times INSIDE it.
//
// These tests run without calling gpufl::init() — the ScopedMonitor ctor
// short-circuits in init_() when there's no Runtime, so the scope side
// is a no-op and we can verify pure BenchInvoker logic (body invocation
// count + ordering) without touching CUDA, file I/O, or the collector
// thread.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gpufl/gpufl.hpp"

using gpufl::ScopeMeta;
using gpufl::detail::BenchInvoker;

// ── Count-based tests ─────────────────────────────────────────────────────

TEST(BenchInvoker, RunsBodyWarmupPlusRepeatTimes) {
    int runs = 0;
    BenchInvoker{"scope", ScopeMeta{}.setRepeat(10).setWarmup(3)} += [&]() {
        ++runs;
    };
    EXPECT_EQ(runs, 13);  // 3 warmup + 10 measured
}

TEST(BenchInvoker, DefaultMetaRunsBodyZeroTimes) {
    // With repeat=0 and warmup=0, the body should never execute. The
    // scope itself still opens/closes (empty BEGIN/END pair).
    int runs = 0;
    BenchInvoker{"empty", ScopeMeta{}} += [&]() { ++runs; };
    EXPECT_EQ(runs, 0);
}

TEST(BenchInvoker, RepeatOnly) {
    int runs = 0;
    BenchInvoker{"repeat_only", ScopeMeta{}.setRepeat(7)} += [&]() {
        ++runs;
    };
    EXPECT_EQ(runs, 7);
}

TEST(BenchInvoker, WarmupOnly) {
    // Warmup-only is a legal but unusual configuration — the scope
    // opens and closes immediately (no measured iterations). Useful
    // for documenting "I ran N warmup launches outside any timed scope."
    int runs = 0;
    BenchInvoker{"warmup_only", ScopeMeta{}.setWarmup(5)} += [&]() {
        ++runs;
    };
    EXPECT_EQ(runs, 5);
}

// ── Ordering: warmup runs BEFORE scope opens ──────────────────────────────
//
// We can't directly observe ScopedMonitor's open/close from here without
// initializing the full runtime + a logger spy. Instead we use a tracker
// that snapshots `runs` at every invocation and asserts the sequence is
// `[1, 2, ..., warmup+repeat]` with no out-of-order calls — which is the
// only property the macro contract makes.
//
// The "warmup happens before the scope" invariant is enforced by
// inspection of BenchInvoker::operator+= (warmup for-loop, then
// ScopedMonitor ctor, then repeat for-loop) and end-to-end by the
// example's NDJSON log when run on real hardware.

TEST(BenchInvoker, BodyInvocationsAreSequential) {
    std::vector<int> trace;
    BenchInvoker{"seq", ScopeMeta{}.setRepeat(4).setWarmup(2)} += [&]() {
        trace.push_back(static_cast<int>(trace.size()) + 1);
    };
    ASSERT_EQ(trace.size(), 6u);
    for (size_t i = 0; i < trace.size(); ++i) {
        EXPECT_EQ(trace[i], static_cast<int>(i + 1));
    }
}

// ── Lambda capture semantics ──────────────────────────────────────────────

TEST(BenchInvoker, LambdaCapturesEnclosingState) {
    // The macro uses `[&]` capture, so the lambda mutates enclosing
    // locals. Verify multiple captured variables flow through.
    int counter = 0;
    int accumulator = 0;
    BenchInvoker{"capture", ScopeMeta{}.setRepeat(5)} += [&]() {
        ++counter;
        accumulator += counter;  // 1 + 2 + 3 + 4 + 5 = 15
    };
    EXPECT_EQ(counter, 5);
    EXPECT_EQ(accumulator, 15);
}

// ── Macro expansion test ──────────────────────────────────────────────────
//
// Verifies the GFL_BENCH macro itself (not just BenchInvoker) compiles
// and behaves the same. Catches regressions in the macro definition —
// e.g. the trailing `+= [&]()` pattern, the `__VA_ARGS__` forwarding,
// or the ScopeMeta{...} wrapping.

TEST(GflBenchMacro, RunsBodyWarmupPlusRepeatTimes) {
    int runs = 0;
    GFL_BENCH("macro_test",
              gpufl::ScopeMeta{}.setRepeat(8).setWarmup(2)) {
        ++runs;
    };
    EXPECT_EQ(runs, 10);
}

TEST(GflBenchMacro, EmptyMetaIsNoOpBody) {
    int runs = 0;
    GFL_BENCH("macro_empty", gpufl::ScopeMeta{}) { ++runs; };
    EXPECT_EQ(runs, 0);
}

// ── Tag field flows through ───────────────────────────────────────────────

TEST(ScopeMeta, BuilderChainSetsAllFields) {
    auto meta = ScopeMeta{}
                    .setTag("ml")
                    .setRepeat(42)
                    .setWarmup(7);
    EXPECT_EQ(meta.tag, "ml");
    EXPECT_EQ(meta.repeat, 42u);
    EXPECT_EQ(meta.warmup, 7u);
}

TEST(ScopeMeta, DefaultsAreZeroAndEmpty) {
    ScopeMeta meta;
    EXPECT_TRUE(meta.tag.empty());
    EXPECT_EQ(meta.repeat, 0u);
    EXPECT_EQ(meta.warmup, 0u);
}
