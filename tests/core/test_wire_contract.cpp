// Wire-format contract test.
//
// This file pins the on-the-wire shape of *every* NDJSON record type the
// client emits to the GPUFlight backend. The backend's columnar ingestion
// parsers key off two things and nothing else:
//
//   1. the top-level "type" discriminator, and
//   2. the exact "columns":[...] ordering for the batch records.
//
// If a client-side edit renames a type, reorders a column, drops a field,
// or bumps the in-band "version", that is a backwards-incompatible wire
// change — it MUST be paired with a coordinated backend change and a
// kWireVersion bump. This test exists so such a change trips the build
// here, loudly, instead of silently corrupting ingested data in prod.
//
// When a failure here is intentional:
//   - bump kWireVersion in include/gpufl/core/version.hpp,
//   - update the matching backend parser,
//   - update the expectations below.
//
// Follows the snapshot-test style of test_batch_models.cpp.

#include <gtest/gtest.h>

#include <string>

#include "gpufl/core/events.hpp"
#include "gpufl/core/version.hpp"
#include "gpufl/core/model/batch_models.hpp"
#include "gpufl/core/model/lifecycle_model.hpp"

namespace {

// Substring assert with a readable message on failure.
::testing::AssertionResult JsonContains(const std::string& json,
                                        const std::string& needle) {
    if (json.find(needle) != std::string::npos) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << "expected wire JSON to contain:\n  " << needle
           << "\nbut got:\n  " << json;
}

}  // namespace

// ── Wire version constant ─────────────────────────────────────────────────
//
// kWireVersion is the schema version of the in-band "version":<n> field on
// every batch envelope. Today's payloads are version "1". Bumping this is a
// deliberate, backend-coordinated act — pin it so an accidental edit fails
// the build.
TEST(WireContract, WireVersionIsPinned) {
    EXPECT_STREQ(gpufl::kWireVersion, "1");
}

// ── job_start ─────────────────────────────────────────────────────────────
TEST(WireContract, JobStartShape) {
    gpufl::InitEvent e;
    e.pid = 4242;
    e.app = "wire_contract_app";
    e.session_id = "sess-1";
    e.log_path = "/tmp/gpufl/sess-1";
    e.ts_ns = 1'700'000'000'000'000'000LL;
    e.session_kind = "trace";
    e.profiling_engine = "nvidia.pc_sampling";

    const std::string json = gpufl::model::InitEventModel(e).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"version\":1"));
    EXPECT_TRUE(JsonContains(json, "\"type\":\"job_start\""));
    EXPECT_TRUE(JsonContains(json, "\"pid\":4242"));
    EXPECT_TRUE(JsonContains(json, "\"app\":\"wire_contract_app\""));
    EXPECT_TRUE(JsonContains(json, "\"session_id\":\"sess-1\""));
    EXPECT_TRUE(JsonContains(json, "\"log_path\":\"/tmp/gpufl/sess-1\""));
    EXPECT_TRUE(JsonContains(json, "\"ts_ns\":1700000000000000000"));
    // Host/device label fields the backend copies onto every downstream row.
    EXPECT_TRUE(JsonContains(json, "\"hostname\":"));
    EXPECT_TRUE(JsonContains(json, "\"ip_addr\":"));
    EXPECT_TRUE(JsonContains(json, "\"gpu_static_devices\":"));
    EXPECT_TRUE(JsonContains(json, "\"cuda_static_devices\":"));
    EXPECT_TRUE(JsonContains(json, "\"rocm_static_devices\":"));
    // session_kind is always present (V40+); profiling_engine only when set.
    EXPECT_TRUE(JsonContains(json, "\"session_kind\":\"trace\""));
    EXPECT_TRUE(JsonContains(json, "\"profiling_engine\":\"nvidia.pc_sampling\""));
}

TEST(WireContract, JobStartOmitsEmptyProfilingEngine) {
    gpufl::InitEvent e;
    e.pid = 1;
    e.app = "monitor_app";
    e.session_id = "sess-2";
    e.ts_ns = 1;
    e.session_kind = "monitor";
    e.profiling_engine = "";

    const std::string json = gpufl::model::InitEventModel(e).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"session_kind\":\"monitor\""));
    EXPECT_EQ(json.find("\"profiling_engine\""), std::string::npos)
        << "profiling_engine must be omitted (not emitted empty) for monitor "
           "sessions: " << json;
}

TEST(WireContract, JobStartEmitsNvidiaNoneSentinel) {
    gpufl::InitEvent e;
    e.pid = 1;
    e.app = "explicit_none_app";
    e.session_id = "sess-3";
    e.ts_ns = 1;
    e.session_kind = "monitor";
    e.profiling_engine = "nvidia.none";

    const std::string json = gpufl::model::InitEventModel(e).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"session_kind\":\"monitor\""));
    EXPECT_TRUE(JsonContains(json, "\"profiling_engine\":\"nvidia.none\""))
        << "explicit-None sessions must emit profiling_engine: nvidia.none: " << json;
}

// ── shutdown ──────────────────────────────────────────────────────────────
TEST(WireContract, ShutdownShape) {
    gpufl::ShutdownEvent e;
    e.pid = 4242;
    e.app = "wire_contract_app";
    e.session_id = "sess-1";
    e.ts_ns = 1'700'000'000'000'000'500LL;

    const std::string json = gpufl::model::ShutdownEventModel(e).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"type\":\"shutdown\""));
    EXPECT_TRUE(JsonContains(json, "\"pid\":4242"));
    EXPECT_TRUE(JsonContains(json, "\"app\":\"wire_contract_app\""));
    EXPECT_TRUE(JsonContains(json, "\"session_id\":\"sess-1\""));
    EXPECT_TRUE(JsonContains(json, "\"ts_ns\":1700000000000000500"));
}

// ── sass_config ───────────────────────────────────────────────────────────
TEST(WireContract, SassConfigShape) {
    gpufl::SassConfigEvent e;
    e.session_id = "sess-1";
    e.ts_ns = 1234;
    e.device_id = 0;
    e.configured_metrics = {"smsp__inst_executed", "smsp__warps_active"};
    e.skipped_metrics = {"dram__bytes"};

    const std::string json = gpufl::model::SassConfigModel(e).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"version\":1"));
    EXPECT_TRUE(JsonContains(json, "\"type\":\"sass_config\""));
    EXPECT_TRUE(JsonContains(json, "\"session_id\":\"sess-1\""));
    EXPECT_TRUE(JsonContains(json, "\"device_id\":0"));
    EXPECT_TRUE(JsonContains(
        json,
        "\"configured_metrics\":[\"smsp__inst_executed\",\"smsp__warps_active\"]"));
    EXPECT_TRUE(JsonContains(json, "\"skipped_metrics\":[\"dram__bytes\"]"));
}

// ── kernel_event_batch ────────────────────────────────────────────────────
//
// The columnar batch records are the heart of the wire contract: the
// backend reads each row positionally against this exact "columns" header.
// Reordering or renaming a column here silently misattributes every value.
TEST(WireContract, KernelEventBatchColumns) {
    gpufl::BatchBuffer<gpufl::KernelBatchRow> batch;
    gpufl::KernelBatchRow row{};
    row.start_ns = 5000;
    row.kernel_id = 7;
    row.stream_id = 2;
    row.duration_ns = 1500;
    row.corr_id = 99;
    row.dyn_shared = 1024;
    row.num_regs = 64;
    row.has_details = 1;
    row.external_kind = 3;
    row.external_id = 808;
    batch.push(row);

    const std::string json =
        gpufl::model::KernelEventBatchModel(batch, "sess-1", 1).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"version\":1"));
    EXPECT_TRUE(JsonContains(json, "\"type\":\"kernel_event_batch\""));
    EXPECT_TRUE(JsonContains(json, "\"base_time_ns\":5000"));
    EXPECT_TRUE(JsonContains(
        json,
        "\"columns\":[\"dt_ns\",\"kernel_id\",\"stream_id\",\"duration_ns\","
        "\"corr_id\",\"dyn_shared\",\"num_regs\",\"has_details\","
        "\"external_kind\",\"external_id\"]"));
    // dt_ns is row.start_ns - base; first row is always 0.
    EXPECT_TRUE(JsonContains(json, "\"rows\":[[0,7,2,1500,99,1024,64,1,3,808]]"));
}

// ── kernel_detail ─────────────────────────────────────────────────────────
TEST(WireContract, KernelDetailShape) {
    gpufl::KernelDetailRow r{};
    r.corr_id = 99;
    r.session_id = "sess-1";
    r.pid = 4242;
    r.app = "wire_contract_app";
    r.grid_x = 16; r.grid_y = 1; r.grid_z = 1;
    r.block_x = 256; r.block_y = 1; r.block_z = 1;
    r.static_shared = 48;
    r.local_bytes = 0;
    r.const_bytes = 0;
    r.max_active_blocks = 8;
    r.shared_mem_executed = 48;

    const std::string json = gpufl::model::KernelDetailModel(r).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"version\":1"));
    EXPECT_TRUE(JsonContains(json, "\"type\":\"kernel_detail\""));
    EXPECT_TRUE(JsonContains(json, "\"corr_id\":99"));
    EXPECT_TRUE(JsonContains(json, "\"grid\":\"(16,1,1)\""));
    EXPECT_TRUE(JsonContains(json, "\"block\":\"(256,1,1)\""));
    EXPECT_TRUE(JsonContains(json, "\"max_active_blocks\":8"));
    EXPECT_TRUE(JsonContains(json, "\"occupancy\":"));
    EXPECT_TRUE(JsonContains(json, "\"limiting_resource\":"));
}

// ── memcpy_event_batch ────────────────────────────────────────────────────
TEST(WireContract, MemcpyEventBatchColumns) {
    gpufl::BatchBuffer<gpufl::MemcpyBatchRow> batch;
    gpufl::MemcpyBatchRow row{};
    row.start_ns = 800;
    row.stream_id = 1;
    row.duration_ns = 400;
    row.bytes = 4096;
    row.copy_kind = 1;
    row.corr_id = 100;
    batch.push(row);

    const std::string json =
        gpufl::model::MemcpyEventBatchModel(batch, "sess-1", 1).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"type\":\"memcpy_event_batch\""));
    EXPECT_TRUE(JsonContains(
        json,
        "\"columns\":[\"dt_ns\",\"stream_id\",\"duration_ns\",\"bytes\","
        "\"copy_kind\",\"corr_id\"]"));
    EXPECT_TRUE(JsonContains(json, "\"rows\":[[0,1,400,4096,1,100]]"));
}

// ── device_metric_batch ───────────────────────────────────────────────────
//
// Two shapes: the base 9-column form, and the extended 18-column form when
// any extended metric is non-zero. The backend's parser must accept both.
TEST(WireContract, DeviceMetricBatchBaseColumns) {
    gpufl::BatchBuffer<gpufl::DeviceMetricBatchRow> batch;
    gpufl::DeviceMetricBatchRow row{};
    row.ts_ns = 1000;
    row.device_id = 0;
    row.gpu_util = 77;
    row.mem_util = 33;
    row.temp_c = 70;
    row.power_mw = 150000;
    row.used_mib = 2048;
    row.total_mib = 24576;
    row.clock_sm = 1987;
    batch.push(row);

    const std::string json =
        gpufl::model::DeviceMetricBatchModel(batch, "sess-1", 1).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"type\":\"device_metric_batch\""));
    EXPECT_TRUE(JsonContains(
        json,
        "\"columns\":[\"dt_ns\",\"device_id\",\"gpu_util\",\"mem_util\","
        "\"temp_c\",\"power_mw\",\"used_mib\",\"total_mib\",\"clock_sm\"]"));
    EXPECT_TRUE(
        JsonContains(json, "\"rows\":[[0,0,77,33,70,150000,2048,24576,1987]]"));
}

TEST(WireContract, DeviceMetricBatchExtendedColumns) {
    gpufl::BatchBuffer<gpufl::DeviceMetricBatchRow> batch;
    gpufl::DeviceMetricBatchRow row{};
    row.ts_ns = 1000;
    row.device_id = 0;
    row.gpu_util = 77;
    row.mem_util = 33;
    row.temp_c = 70;
    row.power_mw = 150000;
    row.used_mib = 2048;
    row.total_mib = 24576;
    row.clock_sm = 1987;
    // Any non-zero extended metric flips the batch to the extended shape.
    row.energy_uj = 123456;
    batch.push(row);

    const std::string json =
        gpufl::model::DeviceMetricBatchModel(batch, "sess-1", 1).buildJson();

    EXPECT_TRUE(JsonContains(
        json,
        "\"columns\":[\"dt_ns\",\"device_id\",\"gpu_util\",\"mem_util\","
        "\"temp_c\",\"power_mw\",\"used_mib\",\"total_mib\",\"clock_sm\","
        "\"fan_speed_pct\",\"temp_mem_c\",\"temp_junction_c\",\"voltage_mv\","
        "\"energy_uj\",\"clock_mem\",\"pcie_bw_bps\",\"ecc_corrected\","
        "\"ecc_uncorrected\"]"));
    EXPECT_TRUE(JsonContains(
        json,
        "\"rows\":[[0,0,77,33,70,150000,2048,24576,1987,0,0,0,0,123456,0,0,0,"
        "0]]"));
}

// ── scope_event_batch ─────────────────────────────────────────────────────
TEST(WireContract, ScopeEventBatchColumns) {
    gpufl::BatchBuffer<gpufl::ScopeBatchRow> batch;
    gpufl::ScopeBatchRow begin{};
    begin.ts_ns = 500;
    begin.scope_instance_id = 1;
    begin.name_id = 1;
    begin.event_type = 0;  // begin
    begin.depth = 0;
    batch.push(begin);
    gpufl::ScopeBatchRow end{};
    end.ts_ns = 6000;
    end.scope_instance_id = 1;
    end.name_id = 1;
    end.event_type = 1;  // end
    end.depth = 0;
    batch.push(end);

    const std::string json =
        gpufl::model::ScopeEventBatchModel(batch, "sess-1", 1).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"type\":\"scope_event_batch\""));
    EXPECT_TRUE(JsonContains(
        json,
        "\"columns\":[\"dt_ns\",\"scope_instance_id\",\"name_id\","
        "\"event_type\",\"depth\"]"));
    EXPECT_TRUE(JsonContains(json, "\"rows\":[[0,1,1,0,0],[5500,1,1,1,0]]"));
}

// ── host_metric_batch ─────────────────────────────────────────────────────
TEST(WireContract, HostMetricBatchColumns) {
    gpufl::BatchBuffer<gpufl::HostMetricBatchRow> batch;
    gpufl::HostMetricBatchRow row{};
    row.ts_ns = 1000;
    row.cpu_pct_x100 = 2500;
    row.ram_used_mib = 4096;
    row.ram_total_mib = 16384;
    batch.push(row);

    const std::string json =
        gpufl::model::HostMetricBatchModel(batch, "sess-1", 1).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"type\":\"host_metric_batch\""));
    EXPECT_TRUE(JsonContains(json, "\"hostname\":"));
    EXPECT_TRUE(JsonContains(json, "\"ip_addr\":"));
    EXPECT_TRUE(JsonContains(
        json,
        "\"columns\":[\"dt_ns\",\"cpu_pct_x100\",\"ram_used_mib\","
        "\"ram_total_mib\"]"));
    EXPECT_TRUE(JsonContains(json, "\"rows\":[[0,2500,4096,16384]]"));
}

// ── synchronization_event_batch ───────────────────────────────────────────
TEST(WireContract, SynchronizationEventBatchColumns) {
    gpufl::BatchBuffer<gpufl::SynchronizationEventBatchRow> batch;
    gpufl::SynchronizationEventBatchRow row{};
    row.start_ns = 1'000'000;
    row.duration_ns = 500'000;
    row.sync_type = 3;
    row.stream_id = 7;
    row.event_id = 0;
    row.context_id = 1;
    row.corr_id = 42;
    row.function_id = 9;
    batch.push(row);

    const std::string json =
        gpufl::model::SynchronizationEventBatchModel(batch, "sess-1", 1)
            .buildJson();

    EXPECT_TRUE(JsonContains(json, "\"type\":\"synchronization_event_batch\""));
    EXPECT_TRUE(JsonContains(
        json,
        "\"columns\":[\"dt_ns\",\"duration_ns\",\"sync_type\",\"stream_id\","
        "\"event_id\",\"context_id\",\"corr_id\",\"function_id\"]"));
    EXPECT_TRUE(JsonContains(json, "\"rows\":[[0,500000,3,7,0,1,42,9]]"));
}

// ── memory_alloc_event_batch ──────────────────────────────────────────────
TEST(WireContract, MemoryAllocEventBatchColumns) {
    gpufl::BatchBuffer<gpufl::MemoryAllocEventBatchRow> batch;
    gpufl::MemoryAllocEventBatchRow row{};
    row.start_ns = 1'000'000;
    row.duration_ns = 0;
    row.memory_op = 1;    // ALLOC
    row.memory_kind = 3;  // DEVICE
    row.address = 139637976727552ULL;
    row.bytes = 1048576;
    row.device_id = 0;
    row.stream_id = 0;
    row.corr_id = 42;
    batch.push(row);

    const std::string json =
        gpufl::model::MemoryAllocEventBatchModel(batch, "sess-1", 1).buildJson();

    EXPECT_TRUE(JsonContains(json, "\"type\":\"memory_alloc_event_batch\""));
    EXPECT_TRUE(JsonContains(
        json,
        "\"columns\":[\"dt_ns\",\"duration_ns\",\"memory_op\",\"memory_kind\","
        "\"address\",\"bytes\",\"device_id\",\"stream_id\",\"corr_id\"]"));
    EXPECT_TRUE(JsonContains(
        json, "\"rows\":[[0,0,1,3,139637976727552,1048576,0,0,42]]"));
}
