# Changelog

All notable changes to `gpufl-client` are documented here. Format
inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows PEP 440 for the Python wheel and semver-style
`MAJOR.MINOR.PATCH` for the C++ library.

## [1.2.1] - 2026-06-23

Headline: **Windows `gpufl trace` now captures real kernel details** — kernel
timing, occupancy, and register counts on short runs (previously synthetic
rows with zeros), and no shutdown hang.

### Fixed

- **Real kernel details on Windows injection.** On short runs the CUPTI activity
  buffers were never flushed before the target exited, so kernels fell back to
  synthetic rows with zero occupancy / registers / GPU duration. gpufl now drains
  activity on a `cudaStreamSynchronize` / `cudaDeviceSynchronize` (kernels done,
  context still alive), so real kernel timing, occupancy, and register counts are
  captured.
- **Shutdown hang on context destroy (Windows injection).** The kernel-occupancy
  path could re-enter cudart (`cudaGetDeviceProperties`) while the driver held the
  context-teardown lock, deadlocking at exit. The SM-properties cache is now warmed
  on the first kernel launch and made teardown-safe.

## [1.2.0] - 2026-06-21

Headline: **injection-mode profiling** — a native `gpufl` launcher that
profiles any process (Windows included) with no code changes, plus
multi-pass capture that runs several engines over one workload and merges
the results.

### Added

- **`gpufl trace` — injection-mode launcher.** Profile any process by injecting
  the gpufl library at launch (POSIX `LD_PRELOAD`, Windows-native) — no changes
  to the target. See [docs/guides/trace-launcher.md](docs/guides/trace-launcher.md).
- **Multi-pass profiling (`--passes`).** `--passes=Trace,PcSampling,SassMetrics`
  runs each engine as its own isolated pass and tags them into one **analysis
  group** (shown as a merged kernel/source view in the dashboard). `--passes=Deep`
  runs the Deep engine (PcSampling + SassMetrics) in one pass, same as the
  embedded engine; embedded Python targets can drive multi-pass too.
  See [docs/guides/multi-pass-profiling.md](docs/guides/multi-pass-profiling.md).
- **Composite passes (`+`).** `--passes=Trace+PcSampling` runs the listed engines
  together in one process (timeline + PC stalls in one window) via
  `GPUFL_ENGINE_COMBO`; a comma still means one isolated pass each. `SassMetrics`
  and `Deep` are rejected inside a `+` group.
- **Bounded-window profiling.** `--warmup` / `--window` / `--window-timeout` /
  `--after-window` time-box a capture of a target that never exits on its own
  (e.g. an inference server), then stop it gracefully (POSIX SIGTERM → grace →
  SIGKILL; Windows stop-event → flush → `TerminateProcess` fallback). Durations
  accept `30s` / `500ms` / `5m` / `1h` / a bare number of seconds.
- **`gpufl monitor`.** Monitoring-only telemetry (GPU/host health, no CUPTI) as
  a standalone command.
- **`RangeProfilerKernelReplay` engine.** Kernel-owned hardware counters via
  AutoRange + Kernel Replay.
- **Windows support.** Native `gpufl trace` injection; NVAPI fallback for
  GPU/memory utilization on WDDM (where NVML reports 0%); real kernel symbol
  resolution in Deep mode; PC sampling under Windows injection.

### Changed

- **Lower per-kernel overhead** — stack symbolization, name demangling, and
  metadata joins moved off the CUPTI callback path onto the collector thread;
  the per-kernel critical section shrunk and several locks removed.
- **PC sampling reports measured kernel timings** (no false "estimated" banner),
  with standalone kernel-collect modes and a split SASS channel.
- Environment-variable names centralized in `include/gpufl/core/env_vars.hpp`.
- `enable_memory_tracking` now **defaults to on**. PyTorch's caching allocator
  generates <1k events/session (negligible overhead); set it false for
  alloc-heavy workloads (e.g. TensorFlow eager).

### Fixed

- Memory allocation tracking (`enable_memory_tracking`) now works in SASS / Deep
  mode. CUPTI `MEMORY2` was gated behind kernel-activity collection — off by
  default in SASS-safe mode — so `gpufl trace` (default Deep) recorded no
  `cudaMalloc`/`cudaFree` events despite the flag being set. Allocation tracking
  is now independent of kernel activity (gated only on `enable_memory_tracking`
  plus the SASS-safety policy), so it collects in every engine.
- Drop kernel activity with invalid zero CUPTI timestamps (they anchored to
  system-boot time and sorted to the top of the kernel list).
- Drop non-finite PM sample values (a priming-sample `NaN` produced `-nan(ind)`
  that broke ingest); anchor PM samples to wall-clock time.
- Synthesize a `shutdown` marker for ungracefully-stopped sessions (window stop
  / SIGKILL / crash), written as an indexed window so the agent uploads it; the
  launcher waits for the upload to finish instead of hard-killing the agent.
- The no-CUDA Windows wheel links cleanly (CUPTI-only binding calls are guarded).

### Removed

- The deprecated `gpufl trace --engine` flag (use `--passes`).
- **`remote_upload`** — the Python kwarg, the C++ `InitOptions::remote_upload`
  field, and the `GPUFL_REMOTE_UPLOAD` env var. The v1.1 backward-compat shim
  (a Python `atexit` handler + the C++ shutdown auto-upload) is gone; passing it
  now raises. Use `with gpufl.session(backend_url=..., api_key=...):` or call
  `gpufl.upload_logs()` / `gpufl::uploadLogs()` after shutdown.
- **`sampling_auto_start`** — the Python kwarg (renamed to
  `continuous_system_sampling` in v1.1; C++ renamed then). The compatibility
  shim is removed and the old name now raises `TypeError`.
- **`InitOptions.backend_url` / `api_key`** — backend credentials no longer live
  on `InitOptions` (the C++ fields and the Python `init()` kwargs are both gone;
  `init()` now rejects them). Pass them to `gpufl.upload_logs()` /
  `gpufl::uploadLogs()` or `gpufl.session(backend_url=..., api_key=...)`; the
  version-discovery probe reads `GPUFL_BACKEND_URL` from the environment.

### Build

- The Windows wheel builds with Ninja + nvcc + MSVC `cl` (not the Visual Studio
  CUDA toolset); per-version wheel generation fixed.

## [1.1.0] - 2026-06-03

### Breaking changes

#### `HttpLogSink` removed - upload is now a separate post-shutdown step

The in-process `HttpLogSink` that POSTed every NDJSON event live
during a session has been deleted. Network failures during the
workload could leak into the GPU job's exit code, and per-event HTTP
added measurable jitter to PyTorch training runs. Upload now happens
as an explicit step after `gpufl::shutdown()` returns.

For Python customers, the migration is **soft** - `remote_upload=True`
still works in v1.1 as a deprecation shim (see Deprecations below).
For pure-C++ customers who `#include`'d the header directly, the
break is a compile error.

| Surface | Before (v1.0.x) | New (v1.1+) | v1.1 backward-compat behavior |
|---|---|---|---|
| Python `init(remote_upload=True)` | Live HttpLogSink during session | `with gpufl.session(...)` or `gpufl.upload_logs(...)` after shutdown | **Still works** - `DeprecationWarning` at init + `atexit` handler that calls `upload_logs()` at interpreter exit |
| C++ `opts.remote_upload = true;` | Live HttpLogSink during session | `gpufl::uploadLogs(uopts)` after `shutdown()` | **Still works** - deprecation log at init + auto-call to `gpufl::uploadLogs()` at the end of `gpufl::shutdown()` (shutdown now blocks until upload completes) |
| Env var `GPUFL_REMOTE_UPLOAD=1` | Live HttpLogSink during session | `gpufl.upload_logs()` post-shutdown | **Still works** - routes through the Python shim above |
| `#include "gpufl/core/logger/http_log_sink.hpp"` | The header | gone | **Compile error** - drop the include |

See [docs/getting-started/sending-data.md](docs/getting-started/sending-data.md)
for the full migration guide.

#### `gpufl` Python console-script removed - `upload` folded into the native binary

The pip-installed `gpufl` console-script (shipped in `1.1.0rc1`/`rc2`,
whose only subcommand was `gpufl upload`) has been removed. The new
native `gpufl` binary - the injection-mode launcher - now provides
`upload` directly alongside `trace` and `version`, so a single command
owns the `gpufl` name instead of a pip script and a binary fighting over
it on `PATH`.

| Surface | Before (1.1.0rc1/rc2) | New (1.1.0+) |
|---|---|---|
| `gpufl upload <dir> …` (pip console-script) | Python `gpufl.cli:main` | Native binary subcommand - **same flags, same 0/1/2 exit codes** |
| Cross-platform / no native binary | (only path) | `python -m gpufl.cli upload <dir> …` - unchanged behavior |
| In-process API `gpufl.upload_logs(...)` | - | unchanged |

Migration: on Linux with the binary installed, `gpufl upload …` works as
before. Elsewhere, switch scripts from `gpufl upload …` to
`python -m gpufl.cli upload …`.

### Deprecations (scheduled for v1.2 removal)

| Field / kwarg | Status in v1.1 | What to use instead |
|---|---|---|
| `InitOptions::remote_upload` (Python kwarg + C++ field) | DeprecationWarning + atexit shim that calls `upload_logs()` at interpreter exit | `with gpufl.session(...)` or call `gpufl.upload_logs()` explicitly after `shutdown()` |
| `InitOptions::backend_url` | Still functional; read by the version-discovery probe and stored for `upload_logs()` to read back | Pass `backend_url` directly to `UploadOptions` / `gpufl.upload_logs()` |
| `InitOptions::api_key` | Same as `backend_url` | Pass `api_key` directly to `UploadOptions` / `gpufl.upload_logs()` |
| `GPUFL_REMOTE_UPLOAD` env var | Still read; routes to the Python atexit shim | Drop from container manifests / start scripts |

All three fields ship in v1.1 to keep the migration painless and will
be removed together in v1.2 - at which point creds live exclusively on
`UploadOptions` and `gpufl::init()` stops touching network config
entirely.

### Breaking changes (cont.)

#### `sampling_auto_start` renamed to `continuous_system_sampling`

The old name only described init-time behavior. The new flag covers
the full policy - the semantics also got fixed (see Bug fixes).

- **Python**: old kwarg still accepted for this release with a
  `DeprecationWarning`. Will be removed in the next release.
- **C++**: hard rename. Compile error points at the call site with
  a clear "no member named 'sampling_auto_start'" message.

### Added

#### Deferred upload - `gpufl.upload_logs()` / `gpufl::uploadLogs()`

A new module under `include/gpufl/upload/`. Reads the session's
NDJSON files post-shutdown, POSTs each event to the existing
`/api/v1/events/{eventType}` backend endpoints. Never throws on
network errors; returns an `UploadResult` with `.success`,
`.events_uploaded`, `.warnings`, etc.

Python orchestration via context manager:

```python
with gpufl.session(app_name="train",
                   backend_url="https://api.gpuflight.com",
                   api_key="gpfl_xxxxx"):
    train_one_epoch()
# On __exit__: shutdown() then upload_logs() - automatic.
```

#### `gpufl upload` CLI

Post-mortem / ad-hoc shipping tool. A subcommand of the native `gpufl`
binary (see the Breaking changes note above - it was briefly a pip
console-script during `rc1`/`rc2` before being folded into the binary).
The cross-platform Python equivalent is `python -m gpufl.cli upload`:

```bash
gpufl upload /tmp/runs/train --backend-url ... --api-key ...
gpufl upload /tmp/runs/train --session-id <uuid>
gpufl upload /tmp/runs/train --all-sessions
gpufl upload /tmp/runs/train --force                 # bypass cursor check
```

Default behavior uploads only the **latest** session found in the
directory (most recent `job_start.ts_ns`). `--session-id` picks a
specific one; `--all-sessions` ships every session present.

#### Session-aware cursor file

`.gpufl-upload-cursor.json` (in the log directory) tracks which
sessions have completed a successful upload. Re-running `gpufl
upload` on a completed session refuses with a clear message
suggesting `--force`; `--all-sessions` mode silently skips completed
sessions and uploads the rest. Survives across runs to skip
already-uploaded rotated files.

#### `ProfilingEngine` - clarified names

The engine enum was reworked into a single, plainly-named ladder
(no aliases). New default is `Monitor` (telemetry only, no CUPTI).

| Name | What it captures |
|---|---|
| `Monitor` | GPU/host health metrics only - no CUPTI. The default. |
| `Trace` | + activity trace: kernels, memcpy, sync (no sampling) |
| `PcSampling` | + PC stall-reason sampling |
| `SassMetrics` | + per-instruction SASS counters |
| `RangeProfiler` | + hardware throughput counters |
| `Deep` | `PcSampling` + `SassMetrics` in one run |

Replaces the earlier `None` / `KernelTrace` / `Continuous` / `Range` /
`PcSamplingWithSass` names. Pre-1.0, no deprecation shim - the old
names are gone.

#### Ref-counted system-metric sampler

`Sampler::configure()` / `activate()` / `deactivate()` / `shutdown()`
replaces the old `start()` / `stop()`. Activation count composes
across `continuous_system_sampling` baseline, `GFL_SCOPE` enter/exit,
and explicit `systemStart()` / `systemStop()` calls - the sampler
runs while any activator is in flight.

### Bug fixes

#### Scope-driven system sampling now works

Before: setting `sampling_auto_start=false` silently disabled all
system metrics, even inside `GFL_SCOPE` regions. The flag's name
suggested "wait for explicit start" semantics but the code disabled
sampling entirely. Now, under the renamed `continuous_system_sampling
= false`, the sampler activates while inside any scope or between
`systemStart` / `systemStop` calls, then idles outside that window.

#### EventWrapper envelope on upload POSTs

The initial `uploadLogs()` draft POSTed bare NDJSON event lines.
The backend's `EventIngestionController` deserialized those into an
`EventWrapper` with every field null, the inner `objectMapper.readValue
(null, ...)` threw, the exception was caught and swallowed, and the
controller returned 200 OK anyway - silent data loss. Every event is
now correctly wrapped in `{data, agentSendingTime, hostname, ipAddr}`.

Regression test added in `tests/upload/test_upload_logs.cpp`.

### Tests added

- `tests/core/test_sampler.cpp` - 8 scenarios for the ref-counted
  Sampler (activate/deactivate, nesting, force-shutdown, unbalanced
  deactivate clamping).
- `tests/upload/test_upload_logs.cpp` - 12 scenarios for the upload
  path (happy path, headers, cursor refusal + force override, auth
  failure, malformed lines, session-id filter, all-sessions,
  lifecycle ordering, EventWrapper envelope regression guard).
- `tests/python/test_continuous_system_sampling.py` - 5 integration
  scenarios for the three sampling modes plus deprecation behavior.

### Internal / build

- Removed `include/gpufl/core/logger/http_log_sink.{hpp,cpp}`.
- Added `include/gpufl/upload/upload_logs.{hpp,cpp}` to the CMake
  target sources.
- `CMakeLists.txt` `project(VERSION)` bumped to 1.1.0; new
  `GPUFL_VERSION_SUFFIX` variable layers optional PEP 440 pre-release
  tokens onto `GPUFL_CLIENT_VERSION`.

### Migration checklist for 1.0.x → 1.1.0

**Optional in v1.1, required by v1.2:**

- [ ] Python: replace every `gpufl.init(remote_upload=True, ...)` call
  with `with gpufl.session(backend_url=..., api_key=...):` or an
  explicit `gpufl.upload_logs(...)` after `shutdown()`. The old form
  still works in v1.1 with a `DeprecationWarning`; v1.2 will remove it.
- [ ] C++: replace `opts.remote_upload = true;` with an explicit
  `gpufl::uploadLogs(uopts)` after `gpufl::shutdown()`. The field
  still compiles in v1.1 but is a no-op; v1.2 will delete it.
- [ ] Container manifests: prefer dropping `GPUFL_REMOTE_UPLOAD` and
  driving upload via your app code (or the `gpufl upload` CLI in a
  lifecycle hook). The env var still routes through the Python shim
  in v1.1; v1.2 stops reading it.
- [ ] Future-proof: start passing `backend_url` and `api_key` directly
  to `gpufl::uploadLogs()` / `gpufl.upload_logs()` rather than relying
  on the InitOptions fields. Those InitOptions fields will move to
  UploadOptions only in v1.2.

**Required in v1.1 (no grace period):**

- [ ] Python: rename `sampling_auto_start` → `continuous_system_sampling`.
  The old name still works with a `DeprecationWarning` (removed in v1.2).
- [ ] C++: rename `opts.sampling_auto_start` → `opts.continuous_system_sampling`
  (compile-time error otherwise - no grace period for C++).
- [ ] If you `#include`'d `http_log_sink.hpp` directly anywhere,
  drop the include - the header is gone.

---

## Releases prior to 1.1.0

See git tags for the historical record. Highlights:

- **1.0.3** - `ScopeMeta` benchmark-iteration helper, scope iterator
  form, `gpufl.report` text summary improvements.
- **1.0.2** - first version published to PyPI; "Stable" status.
- **1.0.1** - `kernel_sample_rate_ms` deprecated (no-op).
- **1.0.0** - first stable contract.
