#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gpufl/report/text_report.hpp"

namespace fs = std::filesystem;

namespace {

class TextReportTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        log_dir_ = fs::temp_directory_path() /
                   (std::string("gpufl_text_report_") + info->name());
        fs::remove_all(log_dir_);
        fs::create_directories(log_dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(log_dir_, ec);
    }

    void WriteLog(const std::string& channel,
                  const std::vector<std::string>& records) const {
        std::ofstream out(log_dir_ / ("fixture." + channel + ".log"),
                          std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.good());
        for (const auto& record : records) out << record << '\n';
    }

    std::string Generate() const {
        gpufl::report::TextReport::Options options;
        options.log_dir = log_dir_.string();
        options.log_prefix = "fixture";
        return gpufl::report::TextReport(options).generate();
    }

    fs::path log_dir_;
};

TEST_F(TextReportTest, AmdPmReportUsesKernelNameAndBackendAwareLabels) {
    WriteLog("device", {
        R"({"type":"job_start","session_id":"s1","app":"amd_report","ts_ns":1000,"gpu_static_devices":[{"name":"AMD Radeon Test","vendor":"AMD","multi_processor_count":32}]})",
        R"({"type":"capture_capabilities","session_id":"s1","requested_engine":"pm_sampling","selected_engine":"amd.device_counting","capabilities":[]})",
        R"({"type":"dictionary_update","session_id":"s1","kernel_dict":{"1":"(anonymous namespace)::sampleRowsWorkload(float*, int, int) [clone .kd]"}})",
        R"({"type":"kernel_event_batch","session_id":"s1","base_time_ns":1000,"columns":["dt_ns","duration_ns","kernel_id","stream_id","corr_id","num_regs","dyn_shared","has_details"],"rows":[[100,1000,1,0,7,136,0,1]]})",
        R"json({"type":"kernel_detail","session_id":"s1","corr_id":7,"grid":"(4096,1,1)","block":"(256,1,1)","occupancy":1.0,"reg_occupancy":1.0,"smem_occupancy":1.0,"warp_occupancy":1.0,"block_occupancy":1.0,"limiting_resource":"waves","max_active_blocks":4,"user_scope":"pm_rows_phase_a"})json",
        R"({"type":"shutdown","session_id":"s1","ts_ns":3000})",
    });
    WriteLog("scope", {
        R"({"type":"dictionary_update","session_id":"s1","function_dict":{"1":"sampleRowsWorkload"},"metric_dict":{"1":"isa_inst_present"}})",
        R"({"type":"profile_sample_batch","session_id":"s1","columns":["function_id","metric_id","metric_value","stall_reason","sample_kind"],"rows":[[1,1,49,0,1]]})",
    });

    const std::string report = Generate();
    EXPECT_NE(report.find("sampleRowsWorkload"), std::string::npos);
    EXPECT_NE(report.find("Compute Units:"), std::string::npos);
    EXPECT_NE(report.find("LDS Occupancy:"), std::string::npos);
    EXPECT_NE(report.find("Wave Occupancy:"), std::string::npos);
    EXPECT_NE(report.find("Waves/CU:"), std::string::npos);
    EXPECT_NE(report.find("GPU Time by Scope (kernel execution only):"),
              std::string::npos);
    EXPECT_EQ(report.find("SM time from CUPTI"), std::string::npos);
    EXPECT_EQ(report.find("Profile / SASS Analysis"), std::string::npos);
    EXPECT_EQ(report.find("Profile / Instruction Analysis"),
              std::string::npos);
    EXPECT_EQ(report.find("isa_inst_present"), std::string::npos);
}

TEST_F(TextReportTest, MeaningfulProfileRowsUseBackendNeutralSectionNames) {
    WriteLog("device", {
        R"({"type":"job_start","session_id":"s1","app":"amd_dispatch","ts_ns":1000,"gpu_static_devices":[{"name":"AMD Radeon Test","vendor":"AMD","multi_processor_count":32}]})",
        R"({"type":"capture_capabilities","session_id":"s1","requested_engine":"sass_metrics","selected_engine":"amd.dispatch_counting","capabilities":[]})",
        R"({"type":"shutdown","session_id":"s1","ts_ns":3000})",
    });
    WriteLog("scope", {
        R"({"type":"dictionary_update","session_id":"s1","function_dict":{"1":"dispatchKernel"},"metric_dict":{"1":"SQ_WAVES"}})",
        R"({"type":"profile_sample_batch","session_id":"s1","columns":["function_id","metric_id","metric_value","stall_reason","sample_kind"],"rows":[[1,1,9,0,1]]})",
    });

    const std::string report = Generate();
    EXPECT_NE(report.find("Profile / Instruction Analysis"),
              std::string::npos);
    EXPECT_NE(report.find("Other Profile Metrics:"), std::string::npos);
    EXPECT_NE(report.find("SQ_WAVES"), std::string::npos);
    EXPECT_EQ(report.find("Other SASS Metrics:"), std::string::npos);
}

TEST_F(TextReportTest, MemoryAllocationBatchRowsProduceSummary) {
    WriteLog("scope", {
        R"({"type":"job_start","session_id":"s1","app":"memory_report","ts_ns":1000,"gpu_static_devices":[{"name":"AMD Radeon Test","vendor":"AMD","multi_processor_count":32}]})",
        R"({"type":"memory_alloc_event_batch","session_id":"s1","base_time_ns":1000,"columns":["dt_ns","duration_ns","memory_op","memory_kind","address","bytes","device_id","stream_id","corr_id"],"rows":[[10,1,1,3,4096,1024,0,0,1],[20,1,1,3,8192,2048,0,0,2],[30,1,2,3,4096,1024,0,0,3],[40,1,1,0,12288,512,0,0,4],[50,1,2,3,8192,2048,0,0,5]]})",
        R"({"type":"shutdown","session_id":"s1","ts_ns":3000})",
    });

    const std::string report = Generate();
    EXPECT_NE(report.find("Memory Allocation Summary"), std::string::npos);
    EXPECT_NE(report.find("Total Events:         5"), std::string::npos);
    EXPECT_NE(report.find("Allocations:          3"), std::string::npos);
    EXPECT_NE(report.find("Frees:                2"), std::string::npos);
    EXPECT_NE(report.find("Bytes Allocated:      3.5 KB"), std::string::npos);
    EXPECT_NE(report.find("Bytes Freed:          3.0 KB"), std::string::npos);
    EXPECT_NE(report.find("Peak Tracked Live:    3.0 KB"), std::string::npos);
    EXPECT_NE(report.find("By Memory Kind:"), std::string::npos);
    EXPECT_NE(report.find("Device"), std::string::npos);
    EXPECT_NE(report.find("Unknown"), std::string::npos);

    const auto allocations = report.find("Memory Allocation Summary");
    const auto system_metrics = report.find("System Metrics");
    EXPECT_LT(allocations, system_metrics);
}

TEST_F(TextReportTest, LegacyMemoryAllocationEventProducesSummary) {
    WriteLog("device", {
        R"({"type":"job_start","session_id":"s1","app":"legacy_memory_report","ts_ns":1000,"gpu_static_devices":[{"name":"NVIDIA Test GPU","vendor":"NVIDIA","multi_processor_count":10}]})",
        R"({"type":"memory_alloc_event","session_id":"s1","start_ns":1100,"duration_ns":0,"memory_op":1,"memory_kind":5,"address":4096,"bytes":2048,"device_id":0,"stream_id":0,"corr_id":1})",
        R"({"type":"shutdown","session_id":"s1","ts_ns":3000})",
    });

    const std::string report = Generate();
    EXPECT_NE(report.find("Total Events:         1"), std::string::npos);
    EXPECT_NE(report.find("Allocations:          1"), std::string::npos);
    EXPECT_NE(report.find("Frees:                0"), std::string::npos);
    EXPECT_NE(report.find("Bytes Allocated:      2.0 KB"), std::string::npos);
    EXPECT_NE(report.find("Managed"), std::string::npos);
}

}  // namespace
