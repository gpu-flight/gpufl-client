#include <gtest/gtest.h>

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/batch_models.hpp"
#include "gpufl/core/model/synchronization_event_model.hpp"
#include "gpufl/core/model/memory_alloc_event_model.hpp"
#include "gpufl/core/model/graph_launch_event_model.hpp"

TEST(BatchModels, DeviceMetricBatchIncludesClockSm) {
    gpufl::BatchBuffer<gpufl::DeviceMetricBatchRow> batch;
    gpufl::DeviceMetricBatchRow row{};
    row.ts_ns = 1000;
    row.device_id = 0;
    row.gpu_util = 77;
    row.mem_util = 33;
    row.temp_c = 70;
    row.power_mw = 150000;
    row.used_mib = 2048;
    row.clock_sm = 1987;
    batch.push(row);

    gpufl::model::DeviceMetricBatchModel model(batch, "s", 1);
    const std::string json = model.buildJson();

    EXPECT_NE(json.find("\"total_mib\""), std::string::npos);
    EXPECT_NE(json.find("\"clock_sm\""), std::string::npos);
    EXPECT_NE(json.find("[0,0,77,33,70,150000,2048,0,1987]"), std::string::npos);
}

TEST(BatchModels, DeviceMetricBatchIncludesExtendedColumnsWhenPresent) {
    gpufl::BatchBuffer<gpufl::DeviceMetricBatchRow> batch;
    gpufl::DeviceMetricBatchRow row{};
    row.ts_ns = 1000;
    row.device_id = 0;
    row.gpu_util = 77;
    row.mem_util = 33;
    row.temp_c = 70;
    row.power_mw = 150000;
    row.used_mib = 2048;
    row.clock_sm = 1987;
    row.fan_speed_pct = 42;
    row.clock_mem = 1200;
    batch.push(row);

    gpufl::model::DeviceMetricBatchModel model(batch, "s", 1);
    const std::string json = model.buildJson();

    EXPECT_NE(json.find("\"fan_speed_pct\""), std::string::npos);
    EXPECT_NE(json.find("\"clock_mem\""), std::string::npos);
    EXPECT_NE(json.find("[0,0,77,33,70,150000,2048,0,1987,42,0,0,0,0,1200,0,0,0]"),
              std::string::npos);
}

// SynchronizationEventModel emits the wire shape the Java backend
// dispatches on. This snapshot test pins the JSON layout so a future
// edit (e.g. accidentally renaming "sync_type" → "type") trips the
// build before it trips an end-to-end test.
TEST(BatchModels, SynchronizationEventModelEmitsExpectedShape) {
    gpufl::SynchronizationEvent ev;
    ev.pid = 1234;
    ev.app = "Heavy_Stress_App";
    ev.session_id = "abc-123";
    ev.start_ns = 1'000'000;
    ev.end_ns   = 1'500'000;
    ev.duration_ns = 500'000;
    ev.sync_type = 3;             // STREAM_SYNCHRONIZE
    ev.stream_id = 7;
    ev.event_id = 0;              // not an event sync
    ev.context_id = 1;
    ev.corr_id = 42;

    gpufl::model::SynchronizationEventModel m(ev);
    const std::string json = m.buildJson();

    // Top-level type discriminator — Java backend dispatches on this.
    EXPECT_NE(json.find("\"type\":\"synchronization_event\""), std::string::npos);
    // Critical fields present.
    EXPECT_NE(json.find("\"sync_type\":3"), std::string::npos);
    EXPECT_NE(json.find("\"stream_id\":7"), std::string::npos);
    EXPECT_NE(json.find("\"corr_id\":42"), std::string::npos);
    EXPECT_NE(json.find("\"duration_ns\":500000"), std::string::npos);
    EXPECT_NE(json.find("\"app\":\"Heavy_Stress_App\""), std::string::npos);
    EXPECT_NE(json.find("\"session_id\":\"abc-123\""), std::string::npos);
}

// MemoryAllocEventModel emits the wire shape the Java backend
// dispatches on. Snapshot test — pin the JSON layout so a future
// rename like "memory_op" → "op" trips the build.
TEST(BatchModels, MemoryAllocEventModelEmitsExpectedShape) {
    gpufl::MemoryAllocEvent ev;
    ev.pid = 1234;
    ev.app = "Heavy_Stress_App";
    ev.session_id = "abc-123";
    ev.start_ns = 1'000'000;
    ev.duration_ns = 0;
    ev.memory_op = 1;             // ALLOC
    ev.memory_kind = 3;           // DEVICE
    ev.address = 0x7f0000000000ULL;
    ev.bytes = 1024 * 1024;
    ev.device_id = 0;
    ev.stream_id = 0;             // sync alloc
    ev.corr_id = 42;

    gpufl::model::MemoryAllocEventModel m(ev);
    const std::string json = m.buildJson();

    EXPECT_NE(json.find("\"type\":\"memory_alloc_event\""), std::string::npos);
    EXPECT_NE(json.find("\"memory_op\":1"), std::string::npos);
    EXPECT_NE(json.find("\"memory_kind\":3"), std::string::npos);
    EXPECT_NE(json.find("\"bytes\":1048576"), std::string::npos);
    EXPECT_NE(json.find("\"corr_id\":42"), std::string::npos);
    // Address: rendered as decimal because we don't want the JSON
    // emitter to depend on stream manipulators across all platforms;
    // frontend formats as hex for display.
    EXPECT_NE(json.find("\"address\":139637976727552"), std::string::npos);
    EXPECT_NE(json.find("\"app\":\"Heavy_Stress_App\""), std::string::npos);
}

// F4: GraphLaunchEventModel emits the wire shape the Java backend
// dispatches on. Snapshot test pins the JSON layout.
TEST(BatchModels, GraphLaunchEventModelEmitsExpectedShape) {
    gpufl::GraphLaunchEvent ev;
    ev.pid = 1234;
    ev.app = "Heavy_Stress_App";
    ev.session_id = "abc-123";
    ev.start_ns = 5'000'000;
    ev.end_ns   = 7'500'000;
    ev.duration_ns = 2'500'000;
    ev.graph_id  = 42;
    ev.device_id = 0;
    ev.stream_id = 7;
    ev.corr_id   = 999;

    gpufl::model::GraphLaunchEventModel m(ev);
    const std::string json = m.buildJson();

    EXPECT_NE(json.find("\"type\":\"graph_launch_event\""), std::string::npos);
    EXPECT_NE(json.find("\"graph_id\":42"), std::string::npos);
    EXPECT_NE(json.find("\"corr_id\":999"), std::string::npos);
    EXPECT_NE(json.find("\"duration_ns\":2500000"), std::string::npos);
    EXPECT_NE(json.find("\"stream_id\":7"), std::string::npos);
    EXPECT_NE(json.find("\"app\":\"Heavy_Stress_App\""), std::string::npos);
    EXPECT_NE(json.find("\"session_id\":\"abc-123\""), std::string::npos);
}
