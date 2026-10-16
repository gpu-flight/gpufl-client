// Shared event log for the capture-gate end-to-end fakes: one line per event,
// appended to FAKE_LOG with O_APPEND so lines from threads and processes do
// not interleave.
#pragma once

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static inline void fake_log(const char* event) {
    const char* path = getenv("FAKE_LOG");
    if (!path || !*path) return;
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) return;
    char line[256];
    int n = snprintf(line, sizeof(line), "%s\n", event);
    if (n > 0) {
        ssize_t written = write(fd, line, (size_t)n);
        (void)written;
    }
    close(fd);
}
