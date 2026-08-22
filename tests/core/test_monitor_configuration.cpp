#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "gpufl/core/env_vars.hpp"
#include "gpufl/core/monitor_configuration.hpp"

namespace {

void setEnv(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    ::setenv(name, value, /*overwrite=*/1);
#endif
}

void unsetEnv(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    ::unsetenv(name);
#endif
}

class MonitorConfigurationTest : public testing::Test {
protected:
    void SetUp() override {
        saveAndUnset_(gpufl::env::kProfilingEngine, profiling_engine_);
        saveAndUnset_(gpufl::env::kPcSamplingPeriod, pc_sampling_period_);
        saveAndUnset_(gpufl::env::kDeepArm, deep_arm_);
    }

    void TearDown() override {
        restore_(gpufl::env::kProfilingEngine, profiling_engine_);
        restore_(gpufl::env::kPcSamplingPeriod, pc_sampling_period_);
        restore_(gpufl::env::kDeepArm, deep_arm_);
    }

private:
    static void saveAndUnset_(const char* name, std::optional<std::string>& out) {
        if (const char* value = std::getenv(name)) out = value;
        unsetEnv(name);
    }

    static void restore_(const char* name,
                         const std::optional<std::string>& value) {
        if (value) setEnv(name, value->c_str());
        else unsetEnv(name);
    }

    std::optional<std::string> profiling_engine_;
    std::optional<std::string> pc_sampling_period_;
    std::optional<std::string> deep_arm_;
};

TEST_F(MonitorConfigurationTest, CopiesEveryInitOptionWithNoEnvironmentOverride) {
    gpufl::InitOptions options;
    options.enable_debug_output = true;
    options.enable_stack_trace = true;
    options.enable_source_collection = false;
    options.source_capture.approved_roots = {"project-a", "project-b"};
    options.source_capture.limits.max_files = 12;
    options.enable_external_correlation = false;
    options.enable_synchronization = false;
    options.enable_memory_tracking = false;
    options.enable_cuda_graphs_tracking = true;
    options.kernel_sample_rate_ms = 37;
    options.pm_sampling_interval_us = 777;
    options.pm_sampling_max_samples = 1234;
    options.pm_sampling_preset = "compute";
    options.pm_sampling_metrics = {"sm__cycles_elapsed", "dram__bytes"};
    options.pm_sampling_scope_only = false;
    options.profiling_engine = gpufl::ProfilingEngine::PmSampling;
    options.backend = gpufl::BackendKind::Amd;

    const auto actual = gpufl::detail::buildMonitorOptions(options);

    EXPECT_TRUE(actual.enable_debug_output);
    EXPECT_TRUE(actual.enable_stack_trace);
    EXPECT_FALSE(actual.enable_source_collection);
    EXPECT_EQ(actual.source_capture.approved_roots,
              (std::vector<std::string>{"project-a", "project-b"}));
    EXPECT_EQ(actual.source_capture.limits.max_files, 12u);
    EXPECT_FALSE(actual.enable_external_correlation);
    EXPECT_FALSE(actual.enable_synchronization);
    EXPECT_FALSE(actual.enable_memory_tracking);
    EXPECT_TRUE(actual.enable_cuda_graphs_tracking);
    EXPECT_EQ(actual.kernel_sample_rate_ms, 37);
    EXPECT_EQ(actual.pm_sampling_interval_us, 777u);
    EXPECT_EQ(actual.pm_sampling_max_samples, 1234u);
    EXPECT_EQ(actual.pm_sampling_preset, "compute");
    EXPECT_EQ(actual.pm_sampling_metrics,
              (std::vector<std::string>{"sm__cycles_elapsed", "dram__bytes"}));
    EXPECT_FALSE(actual.pm_sampling_scope_only);
    EXPECT_EQ(actual.profiling_engine, gpufl::ProfilingEngine::PmSampling);
    EXPECT_EQ(actual.backend_kind, gpufl::MonitorBackendKind::Amd);
}

TEST_F(MonitorConfigurationTest, EnvironmentOverridesTakePrecedence) {
    setEnv(gpufl::env::kProfilingEngine, "RangeProfilerKernelReplay");
    setEnv(gpufl::env::kPcSamplingPeriod, "17");
    setEnv(gpufl::env::kDeepArm, "window");

    gpufl::InitOptions options;
    options.profiling_engine = gpufl::ProfilingEngine::Trace;
    options.deep_window_only = false;
    options.pm_sampling_scope_only = false;

    const auto actual = gpufl::detail::buildMonitorOptions(options);

    EXPECT_EQ(actual.profiling_engine,
              gpufl::ProfilingEngine::RangeProfilerKernelReplay);
    EXPECT_EQ(actual.pc_sampling_period, 17u);
    EXPECT_EQ(actual.deep_arm_mode, gpufl::DeepArmMode::WindowOnly);
    EXPECT_TRUE(actual.pm_sampling_scope_only);
}

TEST_F(MonitorConfigurationTest, MapsMetalBackendToMetalMonitorBackend) {
    gpufl::InitOptions options;
    options.backend = gpufl::BackendKind::Metal;

    const auto actual = gpufl::detail::buildMonitorOptions(options);

    EXPECT_EQ(actual.backend_kind, gpufl::MonitorBackendKind::Metal);
}

TEST_F(MonitorConfigurationTest, InvalidOverridesPreserveConfiguredValues) {
    setEnv(gpufl::env::kProfilingEngine, "not-an-engine");
    setEnv(gpufl::env::kPcSamplingPeriod, "32");
    setEnv(gpufl::env::kDeepArm, "sometimes");

    gpufl::InitOptions options;
    options.profiling_engine = gpufl::ProfilingEngine::SassMetrics;
    options.deep_window_only = true;

    const auto actual = gpufl::detail::buildMonitorOptions(options);

    EXPECT_EQ(actual.profiling_engine, gpufl::ProfilingEngine::SassMetrics);
    EXPECT_EQ(actual.pc_sampling_period, 10u);
    EXPECT_EQ(actual.deep_arm_mode, gpufl::DeepArmMode::WindowOnly);
    EXPECT_TRUE(actual.pm_sampling_scope_only);
}

TEST_F(MonitorConfigurationTest, AlwaysOverrideCanDisableWindowOnlyArming) {
    setEnv(gpufl::env::kDeepArm, "always");

    gpufl::InitOptions options;
    options.deep_window_only = true;
    options.pm_sampling_scope_only = false;

    const auto actual = gpufl::detail::buildMonitorOptions(options);

    EXPECT_EQ(actual.deep_arm_mode, gpufl::DeepArmMode::Always);
    EXPECT_FALSE(actual.pm_sampling_scope_only);
}

}  // namespace
