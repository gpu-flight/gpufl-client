#include "gpufl/core/logger/session_ownership.hpp"

#include <cerrno>
#include <cstring>
#include <mutex>
#include <set>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace gpufl {
namespace fs = std::filesystem;

namespace {
std::mutex g_owned_sessions_mu;
std::set<std::string> g_owned_sessions;

std::string ownershipKey(const fs::path& path) {
    std::error_code ec;
    fs::path normalized = fs::weakly_canonical(path, ec);
    if (ec) {
        ec.clear();
        normalized = fs::absolute(path, ec);
        if (ec) normalized = path;
    }
    return normalized.lexically_normal().generic_string();
}

bool reserveInProcess(const std::string& key) {
    std::lock_guard<std::mutex> guard(g_owned_sessions_mu);
    return g_owned_sessions.insert(key).second;
}

void releaseInProcess(const std::string& key) noexcept {
    if (key.empty()) return;
    std::lock_guard<std::mutex> guard(g_owned_sessions_mu);
    g_owned_sessions.erase(key);
}
}  // namespace

std::unique_ptr<SessionOwnershipLock> SessionOwnershipLock::tryAcquire(
    const fs::path& session_dir, std::string* error) {
    if (error) error->clear();
    std::error_code dir_ec;
    fs::create_directories(session_dir, dir_ec);
    if (dir_ec) {
        if (error) {
            *error = "cannot create session directory: " + dir_ec.message();
        }
        return nullptr;
    }

    auto lock = std::unique_ptr<SessionOwnershipLock>(
        new SessionOwnershipLock());
    lock->path_ = session_dir / kFilename;
    lock->registry_key_ = ownershipKey(lock->path_);
    if (!reserveInProcess(lock->registry_key_)) {
        if (error) *error = "owned by another live process";
        return nullptr;
    }

#if defined(_WIN32)
    HANDLE handle = ::CreateFileW(
        lock->path_.c_str(), GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        if (error) {
            *error = std::system_category()
                         .message(static_cast<int>(::GetLastError()));
        }
        lock->release();
        return nullptr;
    }

    OVERLAPPED overlapped{};
    if (!::LockFileEx(handle,
                      LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                      0, 1, 0, &overlapped)) {
        const DWORD code = ::GetLastError();
        ::CloseHandle(handle);
        if (error) {
            *error = code == ERROR_LOCK_VIOLATION
                         ? "owned by another live process"
                         : std::system_category().message(
                               static_cast<int>(code));
        }
        lock->release();
        return nullptr;
    }
    lock->handle_ = handle;
#else
    const int fd =
        ::open(lock->path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        if (error) *error = std::strerror(errno);
        lock->release();
        return nullptr;
    }
    struct flock file_lock {};
    file_lock.l_type = F_WRLCK;
    file_lock.l_whence = SEEK_SET;
    file_lock.l_start = 0;
    file_lock.l_len = 0;
    if (::fcntl(fd, F_SETLK, &file_lock) != 0) {
        const int code = errno;
        ::close(fd);
        if (error) {
            *error = (code == EACCES || code == EAGAIN)
                         ? "owned by another live process"
                         : std::strerror(code);
        }
        lock->release();
        return nullptr;
    }
    lock->fd_ = fd;
#endif
    return lock;
}

SessionOwnershipLock::~SessionOwnershipLock() { release(); }

SessionOwnershipLock::SessionOwnershipLock(
    SessionOwnershipLock&& other) noexcept
    : path_(std::move(other.path_)),
      registry_key_(std::move(other.registry_key_))
#if defined(_WIN32)
      ,
      handle_(other.handle_)
#else
      ,
      fd_(other.fd_)
#endif
{
    other.registry_key_.clear();
#if defined(_WIN32)
    other.handle_ = nullptr;
#else
    other.fd_ = -1;
#endif
}

SessionOwnershipLock& SessionOwnershipLock::operator=(
    SessionOwnershipLock&& other) noexcept {
    if (this == &other) return *this;
    release();
    path_ = std::move(other.path_);
    registry_key_ = std::move(other.registry_key_);
    other.registry_key_.clear();
#if defined(_WIN32)
    handle_ = other.handle_;
    other.handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
    return *this;
}

bool SessionOwnershipLock::owns() const noexcept {
#if defined(_WIN32)
    return handle_ != nullptr;
#else
    return fd_ >= 0;
#endif
}

const fs::path& SessionOwnershipLock::path() const noexcept { return path_; }

void SessionOwnershipLock::release() noexcept {
#if defined(_WIN32)
    if (handle_) {
        OVERLAPPED overlapped{};
        (void)::UnlockFileEx(static_cast<HANDLE>(handle_), 0, 1, 0,
                             &overlapped);
        (void)::CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
#else
    if (fd_ >= 0) {
        struct flock file_lock {};
        file_lock.l_type = F_UNLCK;
        file_lock.l_whence = SEEK_SET;
        file_lock.l_start = 0;
        file_lock.l_len = 0;
        (void)::fcntl(fd_, F_SETLK, &file_lock);
        (void)::close(fd_);
        fd_ = -1;
    }
#endif
    releaseInProcess(registry_key_);
    registry_key_.clear();
}

}  // namespace gpufl
