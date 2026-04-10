#pragma once

#include <cstddef>
#include <vector>

namespace gpufl {

template <typename Row>
class BatchBuffer {
   public:
    static constexpr size_t kMaxRows = 2048;

    void push(const Row& row) { rows_.push_back(row); }
    bool empty() const { return rows_.empty(); }
    bool needsFlush() const { return rows_.size() >= kMaxRows; }
    void clear() { rows_.clear(); }
    const std::vector<Row>& rows() const { return rows_; }

   private:
    std::vector<Row> rows_;
};

}  // namespace gpufl
