// SIGTERM as an orderly stop in an injected process: the flush runs, and the
// process still ends by SIGTERM. Each case installs the handler in a forked
// child, never in the test process itself.
#include <gtest/gtest.h>

#if defined(__linux__)

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gpufl/inject/sigterm_flush_linux.hpp"

namespace {

int g_report_fd = -1;

void ReportFlush() {
    const char flushed = 'F';
    const ssize_t n = ::write(g_report_fd, &flushed, 1);
    (void)n;
}

// Child side: a bounded life, the default SIGTERM action whatever the test
// runner inherited, and the flush wired to `report_fd`.
void InstallInChild(const int report_fd) {
    ::alarm(10);
    ::signal(SIGTERM, SIG_DFL);
    g_report_fd = report_fd;
    if (!gpufl::inject::InstallSigtermFlush(ReportFlush)) ::_exit(3);
}

TEST(SigtermFlushTest, FlushesThenStillEndsBySigterm) {
    int report[2];
    ASSERT_EQ(::pipe(report), 0);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        ::close(report[0]);
        InstallInChild(report[1]);
        ::kill(::getpid(), SIGTERM);
        for (;;) ::pause();  // the flush thread ends the process
    }
    ::close(report[1]);

    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    char flushed = 0;
    EXPECT_EQ(::read(report[0], &flushed, 1), 1);
    EXPECT_EQ(flushed, 'F');
    ::close(report[0]);
    ASSERT_TRUE(WIFSIGNALED(status)) << "status " << status;
    EXPECT_EQ(WTERMSIG(status), SIGTERM);
}

TEST(SigtermFlushTest, ForkWithoutExecEndsAtOnceWithoutFlushing) {
    int report[2];
    ASSERT_EQ(::pipe(report), 0);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        ::close(report[0]);
        InstallInChild(report[1]);
        // Inherits the handler but not the flush thread.
        const pid_t copy = ::fork();
        if (copy < 0) ::_exit(4);
        if (copy == 0) {
            ::alarm(10);  // not inherited across fork
            ::kill(::getpid(), SIGTERM);
            for (;;) ::pause();
        }
        int copy_status = 0;
        if (::waitpid(copy, &copy_status, 0) != copy) ::_exit(5);
        ::_exit(WIFSIGNALED(copy_status) && WTERMSIG(copy_status) == SIGTERM
                    ? 0
                    : 6);
    }
    ::close(report[1]);

    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status)) << "status " << status;
    EXPECT_EQ(WEXITSTATUS(status), 0);
    char flushed = 0;
    EXPECT_EQ(::read(report[0], &flushed, 1), 0) << "a flush ran";
    ::close(report[0]);
}

}  // namespace

#endif  // __linux__
