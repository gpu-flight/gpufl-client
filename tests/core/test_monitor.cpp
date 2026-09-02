#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "common/test_utils.hpp"
#include "gpufl/core/monitor.hpp"
#include "gpufl/core/monitor_batch_manager.hpp"

class MonitorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Monitor tests might need CUDA if they initialize the CuptiBackend
        // But we can also test them in a way that is safe.
        // If we want to test the full lifecycle, we should probably check for
        // CUDA.
    }

    void TearDown() override { gpufl::Monitor::Shutdown(); }
};

TEST_F(MonitorTest, Lifecycle) {
    gpufl::MonitorOptions opts;
    opts.enable_debug_output = true;

    // Initialize
    gpufl::Monitor::Initialize(opts);

    // Start
    gpufl::Monitor::Start();

    // Stop
    gpufl::Monitor::Stop();

    // Shutdown
    gpufl::Monitor::Shutdown();
}

TEST_F(MonitorTest, RangePushPop) {
    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    gpufl::Monitor::Start();

    // Test ranges - these should push events to the ring buffer
    gpufl::Monitor::PushRange("outer_range");
    gpufl::Monitor::PushRange("inner_range");
    gpufl::Monitor::PopRange();
    gpufl::Monitor::PopRange();

    // Give the collector thread a moment to process (though we don't strictly
    // check the output here)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    gpufl::Monitor::Stop();
    gpufl::Monitor::Shutdown();
}

TEST_F(MonitorTest, ProfilerScopes) {
    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    gpufl::Monitor::Start();

    // Test profiler scopes
    gpufl::Monitor::BeginProfilerScope("prof_scope");
    gpufl::Monitor::EndProfilerScope("prof_scope");

    gpufl::Monitor::Stop();
    gpufl::Monitor::Shutdown();
}

TEST_F(MonitorTest, MultipleInitialize) {
    gpufl::MonitorOptions opts;
    gpufl::Monitor::Initialize(opts);
    gpufl::Monitor::Initialize(opts);  // Should be safe

    gpufl::Monitor::Shutdown();
    gpufl::Monitor::Shutdown();  // Should be safe
}

TEST_F(MonitorTest, InitializeClearsThePreviousSessionsSynthesisPolicy) {
    gpufl::SetSuppressOrphanSyntheticKernels(true);

    gpufl::MonitorOptions opts;
    opts.profiling_engine = gpufl::ProfilingEngine::Monitor;
    opts.backend_kind = gpufl::MonitorBackendKind::None;
    gpufl::Monitor::Initialize(opts);

    EXPECT_FALSE(gpufl::SuppressOrphanSyntheticKernelsForTesting())
        << "a prior session's suppression state must not leak into a new one";
}

TEST_F(MonitorTest, TraceEnablesTrustworthyOrphanPolicyBeforeCollection) {
    SKIP_IF_NO_CUDA();

    gpufl::MonitorOptions opts;
    opts.profiling_engine = gpufl::ProfilingEngine::Trace;
    gpufl::Monitor::Initialize(opts);
    gpufl::Monitor::Start();

    EXPECT_TRUE(gpufl::SuppressOrphanSyntheticKernelsForTesting())
        << "Trace must drop unmatched launches instead of inventing GPU time";
}

// ── scope name stack ────────────────────────────────────────────────────────
//
// The active scope name is what stamps every profile and PM sample. It has to
// behave as a stack: a deep window opens inside the process scope and must
// hand the name back when it closes, or every sample taken after the window
// keeps the window's name.

namespace {

gpufl::ScopeBatchRow ScopeRow(uint64_t instance_id, uint32_t name_id,
                              uint8_t event_type, int depth = 0) {
    gpufl::ScopeBatchRow row;
    row.ts_ns = 1000 + static_cast<int64_t>(instance_id);
    row.scope_instance_id = instance_id;
    row.name_id = name_id;
    row.event_type = event_type;
    row.depth = depth;
    return row;
}

}  // namespace

TEST(ScopeNameStackTest, NestedCloseRestoresTheEnclosingName) {
    gpufl::detail::MonitorBatchManager m;
    const uint32_t outer = m.internScopeName("process:app");
    const uint32_t inner = m.internScopeName("deep_window");

    m.pushTrackedScopeRow(ScopeRow(1, outer, 0, 0));
    EXPECT_EQ(m.activeScopeNameId(), outer);

    m.pushTrackedScopeRow(ScopeRow(2, inner, 0, 1));
    EXPECT_EQ(m.activeScopeNameId(), inner);

    // The regression this guards: without a stack the name stayed on the
    // window and every later sample was attributed to it.
    m.pushTrackedScopeRow(ScopeRow(2, inner, 1));
    EXPECT_EQ(m.activeScopeNameId(), outer);

    m.pushTrackedScopeRow(ScopeRow(1, outer, 1));
    EXPECT_EQ(m.activeScopeNameId(), 0u);
}

TEST(ScopeNameStackTest, OutOfOrderCloseRemovesOnlyItsOwnScope) {
    // A deep window closes from the collector thread, so it can close while an
    // application scope opened after it is still open. Closing must not pop
    // whatever happens to be on top.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t process = m.internScopeName("process:app");
    const uint32_t window = m.internScopeName("deep_window");
    const uint32_t user = m.internScopeName("user_scope");

    m.pushTrackedScopeRow(ScopeRow(1, process, 0, 0));
    m.pushTrackedScopeRow(ScopeRow(2, window, 0, 1));
    m.pushTrackedScopeRow(ScopeRow(3, user, 0, 2));

    m.pushTrackedScopeRow(ScopeRow(2, window, 1));
    EXPECT_EQ(m.activeScopeNameId(), user) << "closing the window must leave the user scope active";

    m.pushTrackedScopeRow(ScopeRow(3, user, 1));
    EXPECT_EQ(m.activeScopeNameId(), process);
}

TEST(ScopeNameStackTest, UnmatchedCloseLeavesTheStackAlone) {
    gpufl::detail::MonitorBatchManager m;
    const uint32_t outer = m.internScopeName("process:app");

    m.pushTrackedScopeRow(ScopeRow(1, outer, 0, 0));
    m.pushTrackedScopeRow(ScopeRow(99, outer, 1));

    EXPECT_EQ(m.activeScopeNameId(), outer);
    EXPECT_EQ(m.openScopeDepth(), 1);
}

TEST(ScopeNameStackTest, DepthReportsWhereANewScopeWouldNest) {
    gpufl::detail::MonitorBatchManager m;
    const uint32_t outer = m.internScopeName("process:app");
    const uint32_t inner = m.internScopeName("deep_window");

    EXPECT_EQ(m.openScopeDepth(), 0);
    m.pushTrackedScopeRow(ScopeRow(1, outer, 0, 0));
    EXPECT_EQ(m.openScopeDepth(), 1);
    m.pushTrackedScopeRow(ScopeRow(2, inner, 0, 1));
    EXPECT_EQ(m.openScopeDepth(), 2);
    m.pushTrackedScopeRow(ScopeRow(2, inner, 1));
    EXPECT_EQ(m.openScopeDepth(), 1);
}

// ── PM sample scope attribution ─────────────────────────────────────────────
//
// Attribution used to rescan every completed scope of the run for every PM
// sample, over a list that was never trimmed. These cover the replacement: a
// retention watermark driven by decode progress, and a per-drain sort-and-sweep
// that must agree with the original resolver exactly.

namespace {

gpufl::PmSampleBatchRow PmSample(int64_t ts_ns) {
    gpufl::PmSampleBatchRow row;
    row.ts_ns = ts_ns;
    return row;
}

gpufl::ScopeBatchRow ScopeEdge(uint64_t instance_id, uint32_t name_id, int64_t ts_ns,
                               uint8_t event_type, int depth) {
    gpufl::ScopeBatchRow row;
    row.ts_ns = ts_ns;
    row.scope_instance_id = instance_id;
    row.name_id = name_id;
    row.event_type = event_type;
    row.depth = depth;
    return row;
}

// Open then close a scope over [start_ns, end_ns] at the given depth.
void RecordScope(gpufl::detail::MonitorBatchManager& m, uint64_t instance_id,
                 uint32_t name_id, int64_t start_ns, int64_t end_ns, int depth) {
    m.pushTrackedScopeRow(ScopeEdge(instance_id, name_id, start_ns, 0, depth));
    m.pushTrackedScopeRow(ScopeEdge(instance_id, name_id, end_ns, 1, depth));
}

}  // namespace

TEST(ScopeAttributionTest, BatchSweepMatchesThePerSampleResolver) {
    // Both must pick the same scope for every timestamp. The sweep is only an
    // optimisation, so any disagreement is a regression in attribution.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t outer = m.internScopeName("process:app");
    const uint32_t mid = m.internScopeName("epoch");
    const uint32_t inner = m.internScopeName("step");

    // Nested, overlapping, and sharing boundaries. Closed out of order on
    // purpose: real closes are timestamped before they take the lock.
    RecordScope(m, 3, inner, 300, 400, 2);
    RecordScope(m, 2, mid, 200, 500, 1);
    RecordScope(m, 1, outer, 100, 900, 0);
    RecordScope(m, 4, inner, 500, 500, 2);   // zero-width, boundary shared with mid
    // Overlapping siblings at the SAME depth. Without these the depth
    // comparison alone decides every sample and the start_ns tie-break is
    // never exercised - two threads each running their own scope is exactly
    // how this arises.
    RecordScope(m, 5, mid, 600, 800, 1);
    RecordScope(m, 6, inner, 700, 850, 1);

    std::vector<int64_t> timestamps;
    for (int64_t ts = 50; ts <= 950; ts += 7) timestamps.push_back(ts);
    for (int64_t ts : {100LL, 200LL, 300LL, 400LL, 500LL, 900LL}) timestamps.push_back(ts);

    std::vector<gpufl::PmSampleBatchRow> rows;
    for (int64_t ts : timestamps) rows.push_back(PmSample(ts));
    m.resolveScopeIdsForTesting(rows, /*fallback_id=*/0);

    for (const auto& row : rows) {
        const uint32_t reference = m.resolveScopeIdForTesting(row.ts_ns);
        EXPECT_EQ(row.scope_name_id, reference)
            << "sweep and per-sample resolver disagree at ts=" << row.ts_ns;
    }
}

TEST(ScopeAttributionTest, SampleInsideAStillOpenScopeIsAttributedToIt) {
    // PM drains mid-run, so the scope covering a sample is routinely still
    // open. Before this, such samples fell back to whatever was on top at drain
    // time - which is not the same question.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t outer = m.internScopeName("process:app");
    const uint32_t open = m.internScopeName("deep_window");

    RecordScope(m, 1, outer, 100, 900, 0);
    m.pushTrackedScopeRow(ScopeEdge(2, open, 400, 0, 1));   // never closed

    std::vector<gpufl::PmSampleBatchRow> rows{PmSample(300), PmSample(500)};
    m.resolveScopeIdsForTesting(rows, /*fallback_id=*/0);

    EXPECT_EQ(rows[0].scope_name_id, outer) << "before the open scope started";
    EXPECT_EQ(rows[1].scope_name_id, open) << "inside the still-open scope";
}

TEST(ScopeAttributionTest, PendingCloseCapsAnOpenScopeAtItsCapturedTimestamp) {
    // A close captures its timestamp before waiting for the scope-state lock.
    // The snapshot must observe that pending timestamp instead of extending the
    // still-open map entry through the entire PM batch.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t name = m.internScopeName("closing");
    m.pushTrackedScopeRow(ScopeEdge(1, name, 100, 0, 0));
    m.markScopeClosePending(1, 200);

    std::vector<gpufl::PmSampleBatchRow> rows{PmSample(150), PmSample(250)};
    m.resolveScopeIdsForTesting(rows, /*fallback_id=*/0);

    EXPECT_EQ(rows[0].scope_name_id, name);
    EXPECT_EQ(rows[1].scope_name_id, 0u)
        << "a pending close must prevent provisional extension past its timestamp";
}

TEST(ScopeAttributionTest, RetentionDropsOnlyWhatTheWatermarkReleases) {
    gpufl::detail::MonitorBatchManager m;
    const uint32_t name = m.internScopeName("step");
    RecordScope(m, 1, name, 100, 200, 0);
    RecordScope(m, 2, name, 300, 400, 0);
    RecordScope(m, 3, name, 500, 600, 0);
    EXPECT_EQ(m.retainedCompletedScopesForTesting(), 3u);

    // Nothing published yet: a run without PM sampling must never lose scopes.
    std::vector<gpufl::PmSampleBatchRow> probe{PmSample(150)};
    m.resolveScopeIdsForTesting(probe, 0);
    EXPECT_EQ(m.retainedCompletedScopesForTesting(), 3u);

    m.publishScopeRetentionWatermark(450);
    m.resolveScopeIdsForTesting(probe, 0);
    EXPECT_EQ(m.retainedCompletedScopesForTesting(), 1u)
        << "only the scope ending after the watermark survives";
}

TEST(ScopeAttributionTest, WatermarkNeverMovesBackwards) {
    // A failed decode or an overflow must not un-retire scopes already
    // released, so a lower value is ignored rather than applied.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t name = m.internScopeName("step");
    RecordScope(m, 1, name, 100, 200, 0);
    RecordScope(m, 2, name, 300, 400, 0);

    m.publishScopeRetentionWatermark(350);
    m.publishScopeRetentionWatermark(50);    // ignored

    std::vector<gpufl::PmSampleBatchRow> probe{PmSample(380)};
    m.resolveScopeIdsForTesting(probe, 0);
    EXPECT_EQ(m.retainedCompletedScopesForTesting(), 1u);
}

TEST(ScopeAttributionTest, DelayedDrainKeepsScopesThatWallClockWouldHaveDropped) {
    // The watermark is event time, not wall clock. A collector stalled for
    // seconds must still attribute the samples it eventually decodes.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t name = m.internScopeName("step");
    RecordScope(m, 1, name, 1'000'000'000, 1'100'000'000, 0);

    // Simulate a long stall: no watermark is published because nothing decoded.
    std::vector<gpufl::PmSampleBatchRow> rows{PmSample(1'050'000'000)};
    m.resolveScopeIdsForTesting(rows, /*fallback_id=*/7);

    EXPECT_EQ(rows[0].scope_name_id, name)
        << "a scope that ended 2s ago is still needed by an undecoded sample";
}

TEST(ScopeAttributionTest, EqualDepthAndStartResolveIdenticallyInBothPaths) {
    // std::sort is not stable and open scopes come out of an unordered_map, so
    // without a tertiary key two scopes sharing depth AND start could resolve
    // differently between the batch sweep and the reference resolver.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t a = m.internScopeName("thread_a");
    const uint32_t b = m.internScopeName("thread_b");

    // Same depth and start, DIFFERENT ends. If the ranking ever reports the two
    // as equivalent, the sweep's ordered set keeps only one of them - and a
    // sample after the shorter one ends then resolves to nothing instead of to
    // the scope still covering it. Equal ends would hide that entirely.
    RecordScope(m, 10, a, 100, 200, 1);
    RecordScope(m, 11, b, 100, 400, 1);

    std::vector<gpufl::PmSampleBatchRow> rows{PmSample(150), PmSample(250), PmSample(350)};
    m.resolveScopeIdsForTesting(rows, /*fallback_id=*/0);
    for (const auto& row : rows) {
        EXPECT_EQ(row.scope_name_id, m.resolveScopeIdForTesting(row.ts_ns));
    }
}

TEST(ScopeAttributionTest, UncoveredSampleIsUnattributedNotGivenTheLiveScope) {
    // The old fallback handed an unmatched sample whatever scope was open at
    // DECODE time. That is the temporal error this resolver exists to remove:
    // a sample can sit in the buffer while scopes come and go.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t earlier = m.internScopeName("earlier");
    const uint32_t live = m.internScopeName("live_now");

    RecordScope(m, 1, earlier, 100, 200, 0);
    // A different scope is open by the time the batch is decoded.
    m.pushTrackedScopeRow(ScopeEdge(2, live, 900, 0, 0));

    std::vector<gpufl::PmSampleBatchRow> rows{PmSample(500)};   // covered by neither
    m.resolveScopeIdsForTesting(rows, /*fallback_id=*/0);

    EXPECT_EQ(rows[0].scope_name_id, 0u)
        << "an uncovered sample must stay unattributed, not inherit the live scope";
}

TEST(ScopeAttributionTest, ManyOverlappingScopesStillResolveCorrectly) {
    // Cross-thread workloads keep a large active set. The sweep has to stay
    // correct as that grows, not just when a couple of scopes nest.
    gpufl::detail::MonitorBatchManager m;

    // A distinct name per scope. With one shared name the assertion could not
    // tell a correct pick from a wrong one - every answer would compare equal.
    constexpr int kScopes = 400;
    std::vector<uint32_t> names;
    names.reserve(kScopes);
    for (int i = 0; i < kScopes; ++i) {
        names.push_back(m.internScopeName("worker_" + std::to_string(i)));
    }
    for (int i = 0; i < kScopes; ++i) {
        RecordScope(m, static_cast<uint64_t>(i + 1), names[i],
                    /*start*/ i, /*end*/ 10'000 + i, /*depth*/ i % 4);
    }

    std::vector<gpufl::PmSampleBatchRow> rows;
    for (int64_t ts = 0; ts < 10'000; ts += 337) rows.push_back(PmSample(ts));
    m.resolveScopeIdsForTesting(rows, /*fallback_id=*/0);

    for (const auto& row : rows) {
        EXPECT_EQ(row.scope_name_id, m.resolveScopeIdForTesting(row.ts_ns))
            << "large active set diverges at ts=" << row.ts_ns;
    }
}

TEST(ScopeAttributionTest, CapAppliesWithoutAnyPmSamples) {
    // The retention watermark is the real bound, but only PM sampling ever
    // publishes one. A Trace-only run closes scopes and never decodes a sample,
    // so if the cap only ran on the PM path this deque would grow for the life
    // of the process - unbounded, and with nothing recorded to say so.
    gpufl::detail::MonitorBatchManager m;
    const uint32_t name = m.internScopeName("step");

    constexpr int kScopes = 70'000;   // over kMaxCompletedScopes
    for (int i = 0; i < kScopes; ++i) {
        RecordScope(m, static_cast<uint64_t>(i + 1), name, i, i + 1, 0);
    }

    EXPECT_LE(m.retainedCompletedScopesForTesting(), 65536u);
    EXPECT_EQ(m.scopeAttributionTruncated(), 0u)
        << "evicting unused Trace-only history is not PM attribution loss";
    EXPECT_EQ(m.pmSampleRowsSeen(), 0u)
        << "evicting unused Trace-only history is not PM attribution loss";
}

TEST(ScopeAttributionTest, CapCountsEvictionWhilePmSamplingIsActive) {
    gpufl::detail::MonitorBatchManager m;
    const uint32_t name = m.internScopeName("step");
    m.beginPmScopeAttribution(100);

    constexpr int kScopes = 70'000;
    for (int i = 0; i < kScopes; ++i) {
        RecordScope(m, static_cast<uint64_t>(i + 1), name, i, i + 1, 0);
    }

    EXPECT_GT(m.scopeAttributionTruncated(), 0u)
        << "eviction while PM can have buffered samples is an attribution risk";
}

TEST(ScopeAttributionTest, BoundarySnapshotsBothSidesOfEveryOpenScope) {
    gpufl::detail::MonitorBatchManager manager;
    manager.reset();

    gpufl::ScopeBatchRow outer = ScopeEdge(11, 101, 1000, 0, 0);
    outer.repeat = 8;
    outer.warmup = 2;
    outer.original_start_ns = 1000;
    manager.pushTrackedScopeRow(outer);
    gpufl::ScopeBatchRow inner = ScopeEdge(12, 102, 2000, 0, 1);
    inner.original_start_ns = 2000;
    manager.pushTrackedScopeRow(inner);

    const auto [closes, opens] =
        manager.snapshotScopeContinuations(5000);
    ASSERT_EQ(closes.size(), 2u);
    ASSERT_EQ(opens.size(), 2u);
    EXPECT_EQ(closes[0].event_type, 3);
    EXPECT_EQ(opens[0].event_type, 2);
    EXPECT_EQ(closes[0].ts_ns, 5000);
    EXPECT_EQ(opens[0].ts_ns, 5000);
    EXPECT_EQ(closes[0].scope_instance_id, 11u);
    EXPECT_EQ(opens[0].scope_instance_id, 11u);
    EXPECT_EQ(opens[0].original_start_ns, 1000);
    EXPECT_EQ(opens[0].repeat, 8u);
    EXPECT_EQ(opens[0].warmup, 2u);
    EXPECT_EQ(closes[1].scope_instance_id, 12u);

    // Snapshotting is non-destructive: the real end still closes the same
    // logical scope after any number of segment boundaries.
    manager.pushTrackedScopeRow(ScopeEdge(12, 102, 6000, 1, 1));
    const auto [later_closes, later_opens] =
        manager.snapshotScopeContinuations(7000);
    ASSERT_EQ(later_closes.size(), 1u);
    ASSERT_EQ(later_opens.size(), 1u);
    EXPECT_EQ(later_opens[0].scope_instance_id, 11u);
    EXPECT_EQ(later_opens[0].original_start_ns, 1000);
}

TEST(ScopeAttributionTest, OldTraceHistoryEvictedDuringPmIsNotPartialAttribution) {
    gpufl::detail::MonitorBatchManager m;
    const uint32_t name = m.internScopeName("trace_step");

    constexpr int kScopes = 70'000;
    for (int i = 0; i < kScopes; ++i) {
        RecordScope(m, static_cast<uint64_t>(i + 1), name, i, i + 1, 0);
    }
    m.beginPmScopeAttribution(1'000'000);

    // Force more evictions after PM starts. Every evicted entry still predates
    // the PM boundary, so none can be needed by a PM sample.
    for (int i = 0; i < 100; ++i) {
        const int64_t ts = 1'000'000 + i;
        RecordScope(m, static_cast<uint64_t>(kScopes + i + 1), name, ts, ts + 1, 0);
    }

    EXPECT_EQ(m.scopeAttributionTruncated(), 0u);
}

TEST(MemoryAllocationBatchTest, CountsAcceptedRowsAndResets) {
    gpufl::detail::MonitorBatchManager manager;
    manager.reset();

    EXPECT_EQ(manager.memoryAllocRowsSeen(), 0u);
    gpufl::MemoryAllocEventBatchRow row{};
    row.start_ns = 100;
    (void)manager.pushMemoryAlloc(row);
    EXPECT_EQ(manager.memoryAllocRowsSeen(), 1u);

    row.start_ns = 200;
    (void)manager.pushMemoryAlloc(row);
    EXPECT_EQ(manager.memoryAllocRowsSeen(), 2u);

    manager.reset();
    EXPECT_EQ(manager.memoryAllocRowsSeen(), 0u);
}
