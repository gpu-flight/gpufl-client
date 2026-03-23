#pragma once

#include <cstdint>

#include "gpufl/core/batch_buffer.hpp"
#include "gpufl/core/events.hpp"
#include "gpufl/core/model/serializable.hpp"

namespace gpufl::model {

// ── kernel_event_batch ────────────────────────────────────────────────────

struct KernelEventBatchModel final : IJsonSerializable {
    KernelEventBatchModel(const BatchBuffer<KernelBatchRow>& buf,
                          std::string session_id, uint64_t batch_id)
        : buf_(buf),
          session_id_(std::move(session_id)),
          batch_id_(batch_id) {}

    std::string buildJson() const override;
    Channel channel() const override { return Channel::Device; }

   private:
    const BatchBuffer<KernelBatchRow>& buf_;
    std::string session_id_;
    uint64_t batch_id_;
};

// ── kernel_detail  (verbose, emitted immediately for has_details kernels) ──

struct KernelDetailModel final : IJsonSerializable {
    explicit KernelDetailModel(const KernelDetailRow& r) : r_(r) {}
    std::string buildJson() const override;
    Channel channel() const override { return Channel::Device; }

   private:
    const KernelDetailRow& r_;
};

// ── memcpy_event_batch ────────────────────────────────────────────────────

struct MemcpyEventBatchModel final : IJsonSerializable {
    MemcpyEventBatchModel(const BatchBuffer<MemcpyBatchRow>& buf,
                          std::string session_id, uint64_t batch_id)
        : buf_(buf),
          session_id_(std::move(session_id)),
          batch_id_(batch_id) {}

    std::string buildJson() const override;
    Channel channel() const override { return Channel::Device; }

   private:
    const BatchBuffer<MemcpyBatchRow>& buf_;
    std::string session_id_;
    uint64_t batch_id_;
};

// ── device_metric_batch ───────────────────────────────────────────────────

struct DeviceMetricBatchModel final : IJsonSerializable {
    DeviceMetricBatchModel(const BatchBuffer<DeviceMetricBatchRow>& buf,
                           std::string session_id, uint64_t batch_id)
        : buf_(buf),
          session_id_(std::move(session_id)),
          batch_id_(batch_id) {}

    std::string buildJson() const override;
    Channel channel() const override { return Channel::System; }

   private:
    const BatchBuffer<DeviceMetricBatchRow>& buf_;
    std::string session_id_;
    uint64_t batch_id_;
};

// ── scope_event_batch ─────────────────────────────────────────────────────

struct ScopeEventBatchModel final : IJsonSerializable {
    ScopeEventBatchModel(const BatchBuffer<ScopeBatchRow>& buf,
                         std::string session_id, uint64_t batch_id)
        : buf_(buf),
          session_id_(std::move(session_id)),
          batch_id_(batch_id) {}

    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }

   private:
    const BatchBuffer<ScopeBatchRow>& buf_;
    std::string session_id_;
    uint64_t batch_id_;
};

// ── profile_sample_batch ──────────────────────────────────────────────────

struct ProfileSampleBatchModel final : IJsonSerializable {
    ProfileSampleBatchModel(const BatchBuffer<ProfileSampleBatchRow>& buf,
                            std::string session_id, uint64_t batch_id)
        : buf_(buf),
          session_id_(std::move(session_id)),
          batch_id_(batch_id) {}

    std::string buildJson() const override;
    Channel channel() const override { return Channel::Scope; }

   private:
    const BatchBuffer<ProfileSampleBatchRow>& buf_;
    std::string session_id_;
    uint64_t batch_id_;
};

// ── host_metric_batch ─────────────────────────────────────────────────────

struct HostMetricBatchModel final : IJsonSerializable {
    HostMetricBatchModel(const BatchBuffer<HostMetricBatchRow>& buf,
                         std::string session_id, uint64_t batch_id)
        : buf_(buf),
          session_id_(std::move(session_id)),
          batch_id_(batch_id) {}

    std::string buildJson() const override;
    Channel channel() const override { return Channel::System; }

   private:
    const BatchBuffer<HostMetricBatchRow>& buf_;
    std::string session_id_;
    uint64_t batch_id_;
};

}  // namespace gpufl::model
