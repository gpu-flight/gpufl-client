# Changelog

All notable changes to `gpufl-client` are documented here. Format
inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows PEP 440 for the Python wheel and semver-style
`MAJOR.MINOR.PATCH` for the C++ library.

## [1.1.0] — Unreleased

Currently validating as **`1.1.0rc1`**. Once it survives a full smoke
cycle in dev + a sample PyTorch run + the example Dockerfile build,
the `rc1` suffix gets dropped to ship as `1.1.0`.

### Breaking changes

#### `HttpLogSink` removed — upload is now a separate post-shutdown step

The in-process `HttpLogSink` that POSTed every NDJSON event live
during a session has been deleted. Network failures during the
workload could leak into the GPU job's exit code, and per-event HTTP
added measurable jitter to PyTorch training runs. Upload now happens
as an explicit step after `gpufl::shutdown()` returns.

For Python customers, the migration is **soft** — `remote_upload=True`
still works in v1.1 as a deprecation shim (see Deprecations below).
For pure-C++ customers who `#include`'d the header directly, the
break is a compile error.

| Surface | Before (v1.0.x) | New (v1.1+) | v1.1 backward-compat behavior |
|---|---|---|---|
| Python `init(remote_upload=True)` | Live HttpLogSink during session | `with gpufl.session(...)` or `gpufl.upload_logs(...)` after shutdown | **Still works** — `DeprecationWarning` at init + `atexit` handler that calls `upload_logs()` at interpreter exit |
| C++ `opts.remote_upload = true;` | Live HttpLogSink during session | `gpufl::uploadLogs(uopts)` after `shutdown()` | **Still works** — deprecation log at init + auto-call to `gpufl::uploadLogs()` at the end of `gpufl::shutdown()` (shutdown now blocks until upload completes) |
| Env var `GPUFL_REMOTE_UPLOAD=1` | Live HttpLogSink during session | `gpufl.upload_logs()` post-shutdown | **Still works** — routes through the Python shim above |
| `#include "gpufl/core/logger/http_log_sink.hpp"` | The header | gone | **Compile error** — drop the include |

See [docs/getting-started/sending-data.md](docs/getting-started/sending-data.md)
for the full migration guide.

### Deprecations (scheduled for v1.2 removal)

| Field / kwarg | Status in v1.1 | What to use instead |
|---|---|---|
| `InitOptions::remote_upload` (Python kwarg + C++ field) | DeprecationWarning + atexit shim that calls `upload_logs()` at interpreter exit | `with gpufl.session(...)` or call `gpufl.upload_logs()` explicitly after `shutdown()` |
| `InitOptions::backend_url` | Still functional; read by the version-discovery probe and stored for `upload_logs()` to read back | Pass `backend_url` directly to `UploadOptions` / `gpufl.upload_logs()` |
| `InitOptions::api_key` | Same as `backend_url` | Pass `api_key` directly to `UploadOptions` / `gpufl.upload_logs()` |
| `GPUFL_REMOTE_UPLOAD` env var | Still read; routes to the Python atexit shim | Drop from container manifests / start scripts |

All three fields ship in v1.1 to keep the migration painless and will
be removed together in v1.2 — at which point creds live exclusively on
`UploadOptions` and `gpufl::init()` stops touching network config
entirely.

### Breaking changes (cont.)

#### `sampling_auto_start` renamed to `continuous_system_sampling`

The old name only described init-time behavior. The new flag covers
the full policy — the semantics also got fixed (see Bug fixes).

- **Python**: old kwarg still accepted for this release with a
  `DeprecationWarning`. Will be removed in the next release.
- **C++**: hard rename. Compile error points at the call site with
  a clear "no member named 'sampling_auto_start'" message.

### Added

#### Deferred upload — `gpufl.upload_logs()` / `gpufl::uploadLogs()`

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
# On __exit__: shutdown() then upload_logs() — automatic.
```

#### `gpufl upload` CLI

Post-mortem / ad-hoc shipping tool. Registered via
`[project.scripts]` in `pyproject.toml`:

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

#### `ProfilingEngine` — clarified names

The engine enum was reworked into a single, plainly-named ladder
(no aliases). New default is `Monitor` (telemetry only, no CUPTI).

| Name | What it captures |
|---|---|
| `Monitor` | GPU/host health metrics only — no CUPTI. The default. |
| `Trace` | + activity trace: kernels, memcpy, sync (no sampling) |
| `PcSampling` | + PC stall-reason sampling |
| `SassMetrics` | + per-instruction SASS counters |
| `RangeProfiler` | + hardware throughput counters |
| `Deep` | `PcSampling` + `SassMetrics` in one run |

Replaces the earlier `None` / `KernelTrace` / `Continuous` / `Range` /
`PcSamplingWithSass` names. Pre-1.0, no deprecation shim — the old
names are gone.

#### Ref-counted system-metric sampler

`Sampler::configure()` / `activate()` / `deactivate()` / `shutdown()`
replaces the old `start()` / `stop()`. Activation count composes
across `continuous_system_sampling` baseline, `GFL_SCOPE` enter/exit,
and explicit `systemStart()` / `systemStop()` calls — the sampler
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
controller returned 200 OK anyway — silent data loss. Every event is
now correctly wrapped in `{data, agentSendingTime, hostname, ipAddr}`.

Regression test added in `tests/upload/test_upload_logs.cpp`.

### Tests added

- `tests/core/test_sampler.cpp` — 8 scenarios for the ref-counted
  Sampler (activate/deactivate, nesting, force-shutdown, unbalanced
  deactivate clamping).
- `tests/upload/test_upload_logs.cpp` — 12 scenarios for the upload
  path (happy path, headers, cursor refusal + force override, auth
  failure, malformed lines, session-id filter, all-sessions,
  lifecycle ordering, EventWrapper envelope regression guard).
- `tests/python/test_continuous_system_sampling.py` — 5 integration
  scenarios for the three sampling modes plus deprecation behavior.

### Internal / build

- Removed `include/gpufl/core/logger/http_log_sink.{hpp,cpp}`.
- Added `include/gpufl/upload/upload_logs.{hpp,cpp}` to the CMake
  target sources.
- `CMakeLists.txt` `project(VERSION)` bumped to 1.1.0; new
  `GPUFL_VERSION_SUFFIX` variable layers the PEP 440 pre-release
  token onto `GPUFL_CLIENT_VERSION` (currently `"rc1"`; set to `""`
  to promote to 1.1.0 final).

### Migration checklist for 1.0.x → 1.1.0rc1

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
  (compile-time error otherwise — no grace period for C++).
- [ ] If you `#include`'d `http_log_sink.hpp` directly anywhere,
  drop the include — the header is gone.

---

## Releases prior to 1.1.0

See git tags for the historical record. Highlights:

- **1.0.3** — `ScopeMeta` benchmark-iteration helper, scope iterator
  form, `gpufl.report` text summary improvements.
- **1.0.2** — first version published to PyPI; "Stable" status.
- **1.0.1** — `kernel_sample_rate_ms` deprecated (no-op).
- **1.0.0** — first stable contract.
