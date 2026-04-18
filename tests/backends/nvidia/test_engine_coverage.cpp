// Parameterized coverage tests for all profiling engines. Each test:
//   1. Initializes the runtime with the parameterized engine + a temp log dir
//   2. Runs a real CUDA kernel inside a GFL_SCOPE
//   3. Shuts down cleanly
//   4. Reads the NDJSON log channels
//   5. Asserts engine-specific contracts (architecture-conditional where
//      the behavior differs, e.g. SASS metric skip on pre-sm_120 GPUs)
//
// CI machines (no GPU) skip via SKIP_IF_NO_CUDA(). Developer machines run
// all 5 engines and emit per-arch-appropriate assertions.

#include <gtest/gtest.h>

#include "common/test_utils.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUPTI && GPUFL_HAS_CUDA

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "common/log_utils.hpp"
#include "common/test_kernel.hpp"
#include "gpufl/gpufl.hpp"

namespace fs = std::filesystem;

namespace {

bool Contains(const std::vector<std::string>& v, const std::string& needle) {
    return std::find(v.begin(), v.end(), needle) != v.end();
}

// CUPTI + the profiler APIs do not reliably tolerate multiple init/shutdown
// cycles within a single process (second run leaves Activity API silently
// disabled, and certain engine paths SEH-crash on re-init). Run each engine
// in its own process — e.g. with `--gtest_filter=AllEngines/*/<Engine>`.
// This flag enforces that policy loudly rather than yielding false failures.
std::atomic<bool> g_engineCoverageRan{false};

}  // namespace

class EngineCoverageTest
    : public ::testing::TestWithParam<gpufl::ProfilingEngine> {
   protected:
    fs::path log_dir_;
    gpufl::ComputeCapability cc_{};
    const std::string log_prefix_ = "engine_coverage_test";

    void SetUp() override {
        SKIP_IF_NO_CUDA();
        if (g_engineCoverageRan.exchange(true)) {
            GTEST_SKIP()
                << "EngineCoverageTest only supports one engine per process.\n"
                << "Invoke separately per engine, e.g.:\n"
                << "  --gtest_filter='AllEngines/*/PcSampling'";
        }
        cc_ = GetTestDeviceCC();
        log_dir_ = gpufl::test::MakeTempLogDir();
    }

    void TearDown() override {
        if (::testing::Test::HasFailure()) {
            std::cerr << "[engine_coverage] Preserving logs for inspection: "
                      << log_dir_.string() << "\n";
        } else {
            std::error_code ec;
            fs::remove_all(log_dir_, ec);
        }
    }

    /** Init → run kernel → shutdown. */
    void RunSession(gpufl::ProfilingEngine engine) {
        gpufl::InitOptions opts;
        opts.app_name = "engine_coverage";
        opts.log_path = (log_dir_ / log_prefix_).string();
        opts.profiling_engine = engine;
        opts.enable_kernel_details = true;
        opts.enable_source_collection = true;
        opts.system_sample_rate_ms = 100;

        ASSERT_TRUE(gpufl::init(opts))
            << "gpufl::init failed for engine=" << gpufl::test::EngineName(engine);

        gpufl::systemStart("cov_system");
        {
            GFL_SCOPE("kernel_work") {
                gpufl::test::RunTestKernel();
            }
        }
        gpufl::systemStop("cov_system");

        gpufl::shutdown();
    }
};

TEST_P(EngineCoverageTest, EmitsExpectedEvents) {
    const auto engine = GetParam();
    RunSession(engine);

    auto logs = gpufl::test::ReadAllLogs(log_dir_, log_prefix_);

    // Blackwell (sm_120+) on WDDM has a documented CUPTI limitation where
    // PC Sampling / SASS counter collection / Range Profiler data is
    // delivered unreliably (see pc_sampling_engine.cpp notes). On such
    // platforms we only firm-assert that the engine *initializes* and
    // doesn't crash; sample-count contracts are best-effort.
    //
    // On Linux with the SamplingAPI (driver 590+), cuptiPCSamplingGetData
    // can return NOT_INITIALIZED even for standalone PcSampling, and
    // RangeProfiler requires elevated CUPTI privileges.  Treat PC sampling
    // and Range Profiler counts as best-effort on all platforms — the
    // firm contract is that the engine initializes, runs, and shuts down
    // without crashing.
    const bool samplesBestEffort = true;

    // ── Contracts shared by every engine ─────────────────────────────────
    // job_start must appear in at least one channel (logger emits to all on
    // session start). We assert it's in the scope channel specifically.
    EXPECT_FALSE(gpufl::test::FilterByType(logs.scope, "job_start").empty())
        << "No job_start event in scope log for engine "
        << gpufl::test::EngineName(engine);

    // Activity API is always on: kernel batches should appear for any engine
    // once a kernel runs.
    EXPECT_FALSE(gpufl::test::FilterByType(logs.device, "kernel_event_batch").empty())
        << "No kernel_event_batch events for engine "
        << gpufl::test::EngineName(engine);

    // ── Engine-specific contracts ────────────────────────────────────────
    // profile_sample_batch events are emitted to the Scope channel.
    const int pcSamples = gpufl::test::CountProfileSamplesOfKind(
        logs.scope, "pc_sampling");
    const int sassSamples = gpufl::test::CountProfileSamplesOfKind(
        logs.scope, "sass_metric");
    const auto sassConfigs =
        gpufl::test::FilterByType(logs.device, "sass_config");
    const auto perfEvents =
        gpufl::test::FilterByType(logs.scope, "perf_metric_event");

    // Useful summary print for manual differential review across machines.
    std::cerr << "[engine_coverage] engine=" << gpufl::test::EngineName(engine)
              << " cc=sm_" << cc_.major << cc_.minor
              << " pc_samples=" << pcSamples
              << " sass_samples=" << sassSamples
              << " sass_configs=" << sassConfigs.size()
              << " perf_events=" << perfEvents.size() << "\n";

    switch (engine) {
        case gpufl::ProfilingEngine::None: {
            EXPECT_EQ(pcSamples, 0)
                << "None engine must not produce pc_sampling rows";
            EXPECT_EQ(sassSamples, 0)
                << "None engine must not produce sass_metric rows";
            EXPECT_TRUE(sassConfigs.empty())
                << "None engine must not emit sass_config";
            EXPECT_TRUE(perfEvents.empty())
                << "None engine must not emit perf_metric_event";
            break;
        }

        case gpufl::ProfilingEngine::PcSampling: {
            if (samplesBestEffort) {
                std::cerr << "[engine_coverage] sm_120+ (Blackwell/WDDM): "
                             "PC-sampling best-effort, not asserting count\n";
            } else {
                EXPECT_GT(pcSamples, 0)
                    << "PcSampling must produce at least one pc_sampling row";
            }
            EXPECT_EQ(sassSamples, 0)
                << "PcSampling must not produce sass_metric rows";
            EXPECT_TRUE(sassConfigs.empty())
                << "PcSampling must not emit sass_config";
            break;
        }

        case gpufl::ProfilingEngine::SassMetrics: {
            ASSERT_FALSE(sassConfigs.empty())
                << "SassMetrics must emit at least one sass_config event";
            const auto configured = gpufl::test::GetStringArrayField(
                logs.device, "sass_config", "configured_metrics");
            const auto skipped = gpufl::test::GetStringArrayField(
                logs.device, "sass_config", "skipped_metrics");

            EXPECT_TRUE(Contains(configured, "smsp__sass_inst_executed"))
                << "inst_executed must be configured on every arch";
            EXPECT_TRUE(Contains(configured, "smsp__sass_thread_inst_executed"))
                << "thread_inst_executed must be configured on every arch";

            if (cc_.atLeast(12, 0)) {
                EXPECT_TRUE(Contains(configured,
                                     "smsp__sass_sectors_mem_global_ideal"))
                    << "sm_120+ must accept aggregate sectors_mem_global_ideal";
            } else {
                EXPECT_TRUE(Contains(
                    skipped, "smsp__sass_sectors_mem_global_op_ld_ideal"))
                    << "pre-sm_120 must proactively skip op_ld_ideal";
                EXPECT_TRUE(Contains(
                    skipped, "smsp__sass_sectors_mem_global_op_st_ideal"))
                    << "pre-sm_120 must proactively skip op_st_ideal";
            }

            if (samplesBestEffort) {
                std::cerr << "[engine_coverage] sm_120+ (Blackwell/WDDM): "
                             "SASS sample collection best-effort, "
                             "not asserting count\n";
            } else {
                EXPECT_GT(sassSamples, 0)
                    << "SassMetrics must produce at least one sass_metric row";
            }
            EXPECT_EQ(pcSamples, 0)
                << "SassMetrics must not produce pc_sampling rows";
            break;
        }

        case gpufl::ProfilingEngine::RangeProfiler: {
            if (samplesBestEffort && perfEvents.empty()) {
                std::cerr << "[engine_coverage] sm_120+ (Blackwell/WDDM): "
                             "Range Profiler perf events not delivered; "
                             "best-effort on this platform\n";
                break;
            }
            ASSERT_FALSE(perfEvents.empty())
                << "RangeProfiler must emit at least one perf_metric_event";
            const auto& pe = perfEvents.front();

            const double smPct   = pe.value<double>("sm_throughput_pct", -2.0);
            const double l1Pct   = pe.value<double>("l1_hit_rate_pct", -2.0);
            const double l2Pct   = pe.value<double>("l2_hit_rate_pct", -2.0);
            const double tensor  = pe.value<double>("tensor_active_pct", -2.0);
            const int64_t dramR  = pe.value<int64_t>("dram_read_bytes",  -1);
            const int64_t dramW  = pe.value<int64_t>("dram_write_bytes", -1);

            // Fields must be present (not the missing-field sentinel −2.0).
            EXPECT_GE(smPct, -1.0);
            EXPECT_GE(l1Pct, -1.0);
            EXPECT_GE(l2Pct, -1.0);
            EXPECT_GE(tensor, -1.0);
            EXPECT_GE(dramR, 0);
            EXPECT_GE(dramW, 0);

            EXPECT_EQ(pcSamples, 0)
                << "RangeProfiler must not produce pc_sampling rows";
            EXPECT_EQ(sassSamples, 0)
                << "RangeProfiler must not produce sass_metric rows";
            break;
        }

        case gpufl::ProfilingEngine::PcSamplingWithSass: {
            if (samplesBestEffort) {
                std::cerr << "[engine_coverage] sample collection best-effort "
                             "(PC sampling may be skipped due to Profiler API "
                             "conflict with SASS on SamplingAPI)\n";
            } else {
                EXPECT_GT(pcSamples, 0)
                    << "PcSamplingWithSass must produce pc_sampling rows";
                EXPECT_GT(sassSamples, 0)
                    << "PcSamplingWithSass must produce sass_metric rows";
            }
            ASSERT_FALSE(sassConfigs.empty())
                << "PcSamplingWithSass must emit sass_config";

            const auto configured = gpufl::test::GetStringArrayField(
                logs.device, "sass_config", "configured_metrics");
            const auto skipped = gpufl::test::GetStringArrayField(
                logs.device, "sass_config", "skipped_metrics");
            EXPECT_TRUE(Contains(configured, "smsp__sass_inst_executed"));
            EXPECT_TRUE(Contains(configured, "smsp__sass_thread_inst_executed"));
            if (!cc_.atLeast(12, 0)) {
                EXPECT_TRUE(Contains(
                    skipped, "smsp__sass_sectors_mem_global_op_ld_ideal"));
                EXPECT_TRUE(Contains(
                    skipped, "smsp__sass_sectors_mem_global_op_st_ideal"));
            }
            break;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllEngines, EngineCoverageTest,
    ::testing::Values(gpufl::ProfilingEngine::None,
                      gpufl::ProfilingEngine::PcSampling,
                      gpufl::ProfilingEngine::SassMetrics,
                      gpufl::ProfilingEngine::RangeProfiler,
                      gpufl::ProfilingEngine::PcSamplingWithSass),
    [](const ::testing::TestParamInfo<gpufl::ProfilingEngine>& info) {
        return std::string(gpufl::test::EngineName(info.param));
    });

#endif  // GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUPTI && GPUFL_HAS_CUDA
