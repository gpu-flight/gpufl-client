#if !defined(_WIN32)

#include "gpufl/inject/sigterm_flush_linux.hpp"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <thread>

namespace gpufl::inject {
namespace {

// Published before the handler is installed; the handler reads them, so every
// one is a lock-free atomic.
std::atomic<void (*)()> g_flush{nullptr};
std::atomic<int> g_wake_fd{-1};
std::atomic<pid_t> g_owner_pid{0};
std::atomic<bool> g_requested{false};
std::atomic<bool> g_installed{false};
static_assert(std::atomic<pid_t>::is_always_lock_free &&
              std::atomic<int>::is_always_lock_free &&
              std::atomic<bool>::is_always_lock_free);

// Async-signal-safe.
void RestoreDefaultAction() {
    struct sigaction dfl {};
    dfl.sa_handler = SIG_DFL;
    sigemptyset(&dfl.sa_mask);
    (void)sigaction(SIGTERM, &dfl, nullptr);
}

void OnSigterm(int) {
    const int saved_errno = errno;
    if (getpid() != g_owner_pid.load(std::memory_order_relaxed) ||
        g_requested.exchange(true, std::memory_order_relaxed)) {
        // No flush thread in this process (forked without exec), or a stop is
        // already under way: SIGTERM's own meaning, now. The re-raised signal
        // stays blocked while this handler runs and is delivered on return.
        RestoreDefaultAction();
        (void)raise(SIGTERM);
    } else {
        const char wake = 1;
        const ssize_t n = write(g_wake_fd.load(std::memory_order_relaxed), &wake, 1);
        (void)n;
    }
    errno = saved_errno;
}

void WaitForStop(const int wake_read_fd) {
    char wake = 0;
    for (;;) {
        const ssize_t n = read(wake_read_fd, &wake, 1);
        if (n == 1) break;
        if (n < 0 && errno == EINTR) continue;
        // Nothing can wake this thread any more: give SIGTERM back its
        // default action rather than swallow the first one.
        RestoreDefaultAction();
        return;
    }
    try {
        g_flush.load(std::memory_order_acquire)();
    } catch (...) {
        // Ends by SIGTERM below either way, not by std::terminate.
    }
    RestoreDefaultAction();
    // This thread was started with every signal blocked.
    sigset_t term;
    sigemptyset(&term);
    sigaddset(&term, SIGTERM);
    (void)pthread_sigmask(SIG_UNBLOCK, &term, nullptr);
    (void)raise(SIGTERM);
}

}  // namespace

bool InstallSigtermFlush(void (*flush)()) {
    if (!flush || g_installed.exchange(true)) return false;

    struct sigaction current {};
    if (sigaction(SIGTERM, nullptr, &current) != 0 ||
        (current.sa_flags & SA_SIGINFO) != 0 || current.sa_handler != SIG_DFL) {
        return false;  // the application owns SIGTERM
    }

    int fds[2] = {-1, -1};
    if (pipe2(fds, O_CLOEXEC) != 0) return false;
    // The handler must never block on the pipe.
    if (const int flags = fcntl(fds[1], F_GETFL); flags >= 0) {
        (void)fcntl(fds[1], F_SETFL, flags | O_NONBLOCK);
    }
    g_flush.store(flush, std::memory_order_release);
    g_wake_fd.store(fds[1], std::memory_order_release);
    g_owner_pid.store(getpid(), std::memory_order_release);

    // Started with every signal blocked, so no signal the application expects
    // on one of its own threads is ever delivered on this one.
    sigset_t all;
    sigset_t previous;
    sigfillset(&all);
    (void)pthread_sigmask(SIG_SETMASK, &all, &previous);
    bool started = true;
    try {
        std::thread(WaitForStop, fds[0]).detach();
    } catch (...) {
        started = false;
    }
    (void)pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    if (!started) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }

    struct sigaction stop {};
    stop.sa_handler = OnSigterm;
    sigemptyset(&stop.sa_mask);
    // The application never expected SIGTERM to return into it, so restart
    // whatever system call the signal interrupted.
    stop.sa_flags = SA_RESTART;
    // On failure the parked thread just waits on a pipe nobody writes.
    return sigaction(SIGTERM, &stop, nullptr) == 0;
}

}  // namespace gpufl::inject

#endif  // !defined(_WIN32)
