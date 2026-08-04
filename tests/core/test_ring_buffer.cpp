#include <gtest/gtest.h>

#include "gpufl/core/ring_buffer.hpp"

namespace gpufl {
namespace {

TEST(RingBufferTest, DroppedPushDoesNotPoisonFutureConsumption) {
    RingBuffer<int, 2> buffer;

    ASSERT_TRUE(buffer.Push(1));
    ASSERT_TRUE(buffer.Push(2));
    EXPECT_FALSE(buffer.Push(3));
    EXPECT_EQ(buffer.droppedCount(), 1u);

    int value = 0;
    ASSERT_TRUE(buffer.Consume(value));
    EXPECT_EQ(value, 1);
    ASSERT_TRUE(buffer.Consume(value));
    EXPECT_EQ(value, 2);

    ASSERT_TRUE(buffer.Push(4));
    ASSERT_TRUE(buffer.Consume(value));
    EXPECT_EQ(value, 4);
}

}  // namespace
}  // namespace gpufl
