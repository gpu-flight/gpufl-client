#include <gtest/gtest.h>
#include "common/test_utils.hpp"
#include "gpufl/backends/nvidia/cupti_backend.hpp"

#if GPUFL_ENABLE_NVIDIA && GPUFL_HAS_CUPTI

class CuptiBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_CUDA();
    }
};

TEST_F(CuptiBackendTest, Lifecycle) {
    gpufl::MonitorOptions opts;
    opts.enableDebugOutput = true;
    
    gpufl::CuptiBackend backend;
    
    // Initial state
    EXPECT_FALSE(backend.isActive());
    
    // Initialize
    backend.initialize(opts);
    EXPECT_TRUE(backend.isMonitoringMode());
    EXPECT_FALSE(backend.isProfilingMode());
    
    // Start
    backend.start();
    EXPECT_TRUE(backend.isActive());
    
    // Stop
    backend.stop();
    EXPECT_FALSE(backend.isActive());
    
    // Shutdown
    backend.shutdown();
}

TEST_F(CuptiBackendTest, ProfilingMode) {
    gpufl::MonitorOptions opts;
    opts.isProfiling = true;
    
    gpufl::CuptiBackend backend;
    backend.initialize(opts);
    
    EXPECT_TRUE(backend.isMonitoringMode());
    EXPECT_TRUE(backend.isProfilingMode());
    
    backend.start();
    EXPECT_TRUE(backend.isActive());
    
    // onScopeStart/Stop should not crash even if PC Sampling fails to enable on some GPUs
    backend.onScopeStart("test_scope");
    backend.onScopeStop("test_scope");
    
    backend.stop();
    backend.shutdown();
}

TEST_F(CuptiBackendTest, ScopeCallbacks) {
    gpufl::MonitorOptions opts;
    gpufl::CuptiBackend backend;
    backend.initialize(opts);
    backend.start();
    
    // Should be safe to call even in non-profiling mode
    backend.onScopeStart("test_scope");
    backend.onScopeStop("test_scope");
    
    backend.stop();
    backend.shutdown();
}

#endif
