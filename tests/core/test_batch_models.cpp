#include <gtest/gtest.h>

#include "gpufl/core/events.hpp"
#include "gpufl/core/model/batch_models.hpp"

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

    EXPECT_NE(json.find("\"clock_sm\""), std::string::npos);
    EXPECT_NE(json.find("[0,0,77,33,70,150000,2048,1987]"), std::string::npos);
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
    EXPECT_NE(json.find("[0,0,77,33,70,150000,2048,1987,42,0,0,0,0,1200,0,0,0]"),
              std::string::npos);
}
