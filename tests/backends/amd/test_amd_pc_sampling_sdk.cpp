#include <gtest/gtest.h>

#include <rocprofiler-sdk/pc_sampling.h>

#include <set>
#include <string>

#include "gpufl/backends/amd/amd_pc_sampling_policy.hpp"

// Built only with rocprofiler-sdk. The name functions are pure lookups: no
// agent, context or GPU is involved.

TEST(AmdPcSamplingSdk, BucketsComeFromTheSdksNotIssuedReasonNames) {
    std::set<std::string> buckets;
    for (uint32_t value = 0;
         value < ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST;
         ++value) {
        const char* name =
            rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
                static_cast<
                    rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
                    value));
        ASSERT_NE(name, nullptr) << value;
        buckets.insert(gpufl::amd::AmdPcSamplingBucketName(false, name, value));
    }
    EXPECT_EQ(buckets,
              (std::set<std::string>{
                  "amd_pcs_not_issued_none",
                  "amd_pcs_not_issued_no_instruction_available",
                  "amd_pcs_not_issued_alu_dependency",
                  "amd_pcs_not_issued_waitcnt",
                  "amd_pcs_not_issued_internal_instruction",
                  "amd_pcs_not_issued_barrier_wait",
                  "amd_pcs_not_issued_arbiter_not_win",
                  "amd_pcs_not_issued_arbiter_win_ex_stall",
                  "amd_pcs_not_issued_other_wait",
                  "amd_pcs_not_issued_sleep_wait",
              }));
}

TEST(AmdPcSamplingSdk, BucketsComeFromTheSdksInstructionTypeNames) {
    std::set<std::string> buckets;
    for (uint32_t value = 0;
         value < ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LAST; ++value) {
        const char* name = rocprofiler_get_pc_sampling_instruction_type_name(
            static_cast<rocprofiler_pc_sampling_instruction_type_t>(value));
        ASSERT_NE(name, nullptr) << value;
        buckets.insert(gpufl::amd::AmdPcSamplingBucketName(true, name, value));
    }
    EXPECT_EQ(buckets, (std::set<std::string>{
                           "amd_pcs_issued_none",
                           "amd_pcs_issued_valu",
                           "amd_pcs_issued_matrix",
                           "amd_pcs_issued_scalar",
                           "amd_pcs_issued_tex",
                           "amd_pcs_issued_lds",
                           "amd_pcs_issued_lds_direct",
                           "amd_pcs_issued_flat",
                           "amd_pcs_issued_export",
                           "amd_pcs_issued_message",
                           "amd_pcs_issued_barrier",
                           "amd_pcs_issued_branch_not_taken",
                           "amd_pcs_issued_branch_taken",
                           "amd_pcs_issued_jump",
                           "amd_pcs_issued_other",
                           "amd_pcs_issued_no_inst",
                           "amd_pcs_issued_dual_valu",
                       }));
}
