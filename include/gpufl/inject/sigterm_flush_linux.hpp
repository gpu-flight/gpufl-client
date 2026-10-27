#pragma once

// SIGTERM as an orderly stop for an injected process. SIGTERM's default action
// ends the process without running atexit, so the injection library's
// shutdown (the final flush, capture_capabilities and the shutdown row) never
// ran and the launcher had to write a synthetic shutdown. Separate from
// inject_entry.cpp so the signal machinery is testable without gpufl.

namespace gpufl::inject {

/**
 * Installs a SIGTERM handler that runs `flush` once and then still ends the
 * process by SIGTERM, so its exit status is unchanged for whoever waits on it.
 *
 * Installs only while SIGTERM's disposition is SIG_DFL. An application that
 * handles or ignores SIGTERM owns its shutdown, and its normal exit runs the
 * atexit flush; one that replaces this handler later takes the same path.
 *
 * The handler is async-signal-safe: it only wakes a dedicated thread, which
 * runs `flush`, restores SIG_DFL and re-raises SIGTERM. A second SIGTERM, or
 * one received by a process forked without exec (it inherits the handler but
 * not the thread), terminates at once. A flush that hangs is bounded by
 * whoever sent the signal (the launcher's stop grace, then SIGKILL).
 *
 * Call once per process. Returns false when nothing was installed.
 */
bool InstallSigtermFlush(void (*flush)());

}  // namespace gpufl::inject
