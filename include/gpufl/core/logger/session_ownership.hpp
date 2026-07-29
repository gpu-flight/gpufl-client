#pragma once

#include <filesystem>
#include <memory>
#include <string>

namespace gpufl {

/**
 * Process-lifetime ownership of one session spool directory.
 *
 * The lock is an OS advisory lock, not a PID-file convention: the kernel
 * releases it when the process exits or crashes. Finished root windows remain
 * readable while the lock is held; only operations that mutate `.tmp` or
 * declare the session complete must acquire ownership.
 */
class SessionOwnershipLock {
   public:
    ~SessionOwnershipLock();

    SessionOwnershipLock(const SessionOwnershipLock&) = delete;
    SessionOwnershipLock& operator=(const SessionOwnershipLock&) = delete;

    SessionOwnershipLock(SessionOwnershipLock&&) noexcept;
    SessionOwnershipLock& operator=(SessionOwnershipLock&&) noexcept;

    /**
     * Try to acquire exclusive ownership without waiting.
     *
     * Returns null when another live process owns the session or when the lock
     * file cannot be opened. `error` receives an operator-readable reason.
     */
    static std::unique_ptr<SessionOwnershipLock> tryAcquire(
        const std::filesystem::path& session_dir,
        std::string* error = nullptr);

    [[nodiscard]] bool owns() const noexcept;
    [[nodiscard]] const std::filesystem::path& path() const noexcept;

    static constexpr const char* kFilename = ".gpufl-session.lock";

   private:
    SessionOwnershipLock() = default;
    void release() noexcept;

    std::filesystem::path path_;
    std::string registry_key_;
#if defined(_WIN32)
    void* handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace gpufl
