"""Integration tests for the `continuous_system_sampling` flag.

Three scenarios from the spec — each runs the real gpufl init/shutdown,
optionally inside a Scope, then reads back the NDJSON system log and
counts `device_metric_batch` rows to verify what was (or wasn't)
sampled.

These tests require a real GPU + CUPTI/NVML — they auto-skip in
stub mode (where `gpufl.init()` returns False).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import gpufl


SAMPLE_INTERVAL_MS = 50  # fast enough to accumulate samples in ~1s


def _count_device_metric_rows(log_dir: Path, app_name: str) -> int:
    """Sum `device_metric_batch` row counts across all system channel
    files for `app_name` in `log_dir`. Includes rotated `.gz` slices.

    Each `device_metric_batch` event has a `rows` array; one row per
    NVML poll cycle per device. We count rows (not events) because the
    Sampler batches multiple samples into a single event before flushing
    — the row count is what reflects actual sampling activity.
    """
    import gzip

    total = 0
    patterns = (
        f"{app_name}.system.log",         # active file
        f"{app_name}.system.*.log",       # rotated (uncompressed, if any)
        f"{app_name}.system.*.log.gz",    # rotated (compressed)
    )
    seen: set[Path] = set()
    for pat in patterns:
        for f in log_dir.glob(pat):
            if f in seen:
                continue
            seen.add(f)
            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rt") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "device_metric_batch":
                        total += len(ev.get("rows", []))
    return total


def _run_session(
    log_dir: Path,
    *,
    continuous_system_sampling: bool,
    use_scope: bool,
    scope_duration_s: float = 0.8,
    pre_post_sleep_s: float = 0.3,
) -> int:
    """Run a session with the given config, return device-metric-row count.

    Sleeps `pre_post_sleep_s` before any scope (and after, if used) so
    we can distinguish "sampling during scope" from "sampling regardless
    of scope" — those two windows produce different row counts.
    """
    app = f"test_sampling_{int(time.time_ns())}"
    log_prefix = str(log_dir / app)
    ok = gpufl.init(
        app_name=app,
        log_path=log_prefix,
        continuous_system_sampling=continuous_system_sampling,
        system_sample_rate_ms=SAMPLE_INTERVAL_MS,
        enable_debug_output=False,
    )
    if not ok:
        pytest.skip("gpufl.init returned False — no GPU / stub mode")
    try:
        time.sleep(pre_post_sleep_s)
        if use_scope:
            with gpufl.Scope("test_scope"):
                time.sleep(scope_duration_s)
        time.sleep(pre_post_sleep_s)
    finally:
        gpufl.shutdown()
    return _count_device_metric_rows(log_dir, app)


# Each scenario maps to one of the three rows in the user's expected
# behavior matrix. We don't compare against exact counts (timing-
# dependent) — only the qualitative shape (zero vs many samples).
#
# A "many-samples" lower bound: with a 50 ms sample interval and an
# 800 ms scope, we'd expect ~16 raw NVML polls; the Sampler flushes
# every kMetricBatchSize (4) samples so the row count tracks the poll
# count. We use a conservative floor of 3 to avoid flaky CI under
# scheduler jitter.
_FLOOR_SAMPLES_DURING_SCOPE = 3


def test_continuous_true_always_samples(tmp_path):
    """continuous=True: rows show up regardless of scopes."""
    rows = _run_session(
        tmp_path,
        continuous_system_sampling=True,
        use_scope=False,
    )
    assert rows >= _FLOOR_SAMPLES_DURING_SCOPE, (
        f"continuous=True session should emit many device_metric rows; "
        f"got {rows}. Either the sampler thread never started or the "
        f"interval never elapsed."
    )


def test_continuous_false_no_scope_emits_no_samples(tmp_path):
    """continuous=False with no scope: zero device-metric rows.

    This is the regression test for the original bug — previously
    continuous=False (then `sampling_auto_start=False`) silently
    suppressed all system metrics. We're locking in that the new
    behavior is "no scope, no manual start → no samples" rather than
    "sampler never starts no matter what."
    """
    rows = _run_session(
        tmp_path,
        continuous_system_sampling=False,
        use_scope=False,
    )
    assert rows == 0, (
        f"continuous=False with no scope should emit zero device_metric "
        f"rows; got {rows}. Sampler is leaking samples outside of any "
        f"activation window."
    )


def test_continuous_false_with_scope_samples_during_scope(tmp_path):
    """continuous=False inside a Scope: samples only during the scope.

    This is the new behavior unlocked by the ref-counted Sampler — what
    `continuous_system_sampling=False` means as a *policy*: sample only
    while bracketed by GFL_SCOPE / systemStart.
    """
    rows = _run_session(
        tmp_path,
        continuous_system_sampling=False,
        use_scope=True,
    )
    assert rows >= _FLOOR_SAMPLES_DURING_SCOPE, (
        f"continuous=False + Scope should emit device_metric rows "
        f"during the scope; got {rows}. Either the scope failed to "
        f"activate the sampler or the worker never polled NVML."
    )


def test_sampling_auto_start_kwarg_deprecation_warning(tmp_path):
    """Old kwarg `sampling_auto_start` still works for one release, with
    a DeprecationWarning. Removed in the next major.
    """
    app = f"test_deprecated_kwarg_{int(time.time_ns())}"
    log_prefix = str(tmp_path / app)
    with pytest.warns(DeprecationWarning, match="continuous_system_sampling"):
        ok = gpufl.init(
            app_name=app,
            log_path=log_prefix,
            sampling_auto_start=False,  # deprecated name
            system_sample_rate_ms=SAMPLE_INTERVAL_MS,
        )
    if ok:
        gpufl.shutdown()


def test_sampling_auto_start_and_new_name_together_raises(tmp_path):
    """Passing both old and new kwargs is an error — not a silent winner."""
    app = f"test_dup_kwarg_{int(time.time_ns())}"
    log_prefix = str(tmp_path / app)
    with pytest.raises(TypeError, match="pass only the new name"):
        gpufl.init(
            app_name=app,
            log_path=log_prefix,
            sampling_auto_start=True,
            continuous_system_sampling=True,
        )
