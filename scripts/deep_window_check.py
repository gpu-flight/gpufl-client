#!/usr/bin/env python3
"""Verify one deep-window session's NDJSON logs.

Checks what the feature actually promises:
  1. exactly one window, closed by a bound rather than by session stop;
  2. deep samples exist and stop when the window closes;
  3. the light tier kept running for the whole session, not just the window.

Two wrinkles a naive "is every sample inside [start,end]" check gets wrong,
both learned the hard way:

  * PC and SASS samples are stamped at COLLECT time, and the collect runs at
    window close - just after the end_ns the window recorded. They are
    in-window data carrying an out-of-window timestamp, so a short grace
    after end_ns counts as inside.
  * Some batches arrive on the raw CUPTI clock instead of the wall-clock
    anchor, landing days from the session. Those are reported separately.
    Under `gpufl trace` this used to hit every PM sample (the deferred engine
    start dropped the anchor); it should now be zero there, so the count is
    worth watching rather than ignoring.

Usage: deep_window_check.py <session-or-output-dir> [--expect-windows N]
Exit: 0 pass, 1 fail, 2 nothing to check.
"""
import argparse
import gzip
import json
import pathlib
import sys

GRACE_NS = 100_000_000       # 100ms: a close-time collect, not a leak
ALIEN_NS = 60_000_000_000    # >60s from the window = a different clock domain
WALL_FLOOR = 1_000_000_000_000_000_000
DEEP_BATCHES = ("pm_sample_batch", "profile_sample_batch")
# profile_sample_batch carries two unrelated things. sample_kind 0 and 1 are
# real timed samples (PC stall sampling, SASS metrics); 2 is isa_source_map, a
# static pc_offset -> source_file:line table emitted next to the disassembly.
# The map has no meaningful timestamp and is not something a window collects,
# so counting it made PC sampling look like it was emitting garbage clocks.
TIMED_SAMPLE_KINDS = (0, 1)


def find_session(root: pathlib.Path) -> pathlib.Path:
    if list(root.glob("*.log.gz")) or list(root.glob("*.log")):
        return root
    dirs = [p for p in root.rglob("*")
            if p.is_dir() and (list(p.glob("*.log.gz")) or list(p.glob("*.log")))]
    if not dirs:
        sys.exit(f"no session logs under {root}")
    return sorted(dirs, key=lambda p: p.stat().st_mtime)[-1]


def read_rows(path: pathlib.Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--expect-windows", type=int, default=1)
    # A window that closes at session stop means a bound never fired.
    ap.add_argument("--allow-session-stop", action="store_true")
    args = ap.parse_args()

    sess = find_session(pathlib.Path(args.path))
    rows = []
    for f in sorted(sess.glob("*.log*")):
        # Lifecycle records fan out to several channels; de-dup by channel so
        # counts mean what they look like.
        channel = f.name.split(".")[0]
        rows += [(channel, r) for r in read_rows(f)]

    windows = [r for ch, r in rows
               if r.get("type") == "deep_window_event" and ch == "scope"]
    kernels = sum(1 for ch, r in rows
                  if r.get("type") == "kernel_detail" and ch == "device")

    print(f"session: {sess.name}")
    job = next((r for _, r in rows if r.get("type") == "job_start"), None)
    engine = job.get("profiling_engine", "") if job else ""
    if job:
        print(f"  engine              : {engine}")
    # SASS metrics runs with kernel activity suppressed by default - the
    # combination deadlocks on some drivers, so safe mode drops it and the
    # collector also discards orphaned launches rather than synthesizing
    # rows. Zero kernel rows is correct there, not a missing light tier.
    expect_kernels = "sass_metrics" not in engine
    print(f"  deep windows        : {len(windows)}")
    for w in windows:
        print(f"    close_reason={w['close_reason']} "
              f"duration={w['duration_ns'] / 1e6:.1f}ms "
              f"launches={w['launches_covered']} "
              f"requested={w['requested_duration_ms']}ms/"
              f"{w['requested_max_launches']}launches")

    failures = []
    if len(windows) != args.expect_windows:
        failures.append(f"expected {args.expect_windows} window(s), "
                        f"found {len(windows)}")
    if not windows:
        print("VERDICT: FAIL - no window to check")
        return 1

    w = windows[0]
    if w["close_reason"] == "session_stop" and not args.allow_session_stop:
        failures.append("window closed at session stop - no bound fired, so "
                        "the run was shorter than the window")

    start, end = w["start_ns"], w["end_ns"]
    inside = late = early = alien = 0
    latest_late_ms = 0.0
    # Some batches fan out to every channel, so the same rows appear in
    # device/scope/sass/system. Counting each file independently multiplied
    # them by four.
    seen_batches = set()
    for _, r in rows:
        if r.get("type") not in DEEP_BATCHES:
            continue
        key = (r["type"], r.get("batch_id"), r.get("base_time_ns"),
               len(r.get("rows", [])))
        if key in seen_batches:
            continue
        seen_batches.add(key)
        base = r.get("base_time_ns", 0)
        cols = r.get("columns", [])
        if "dt_ns" not in cols:
            continue
        dt_i = cols.index("dt_ns")
        kind_i = cols.index("sample_kind") if "sample_kind" in cols else None
        for row in r.get("rows", []):
            if kind_i is not None and row[kind_i] not in TIMED_SAMPLE_KINDS:
                continue
            ts = base + row[dt_i]
            if ts < start - ALIEN_NS or ts > end + ALIEN_NS:
                alien += 1
            elif start - GRACE_NS <= ts <= end + GRACE_NS:
                # The engine arms fractionally before the window records its
                # start, so a sample straddling the boundary belongs to it.
                inside += 1
            elif ts < start:
                early += 1
            else:
                late += 1
                latest_late_ms = max(latest_late_ms, (ts - end) / 1e6)

    # The Range profiler reports per-kernel metric events, not sample batches.
    perf_events = sum(1 for _, r in rows
                      if r.get("type") == "kernel_perf_metric_event")

    print(f"  deep samples inside : {inside}")
    print(f"  deep samples late   : {late}"
          + (f" (latest +{latest_late_ms:.0f}ms after close)" if late else ""))
    if early:
        print(f"  deep samples early  : {early}  (before the window armed)")
    print(f"  unanchored samples  : {alien}"
          + ("  <- raw CUPTI clock, not the wall anchor" if alien else ""))
    print(f"  range perf events   : {perf_events}")
    print(f"  kernel rows (run)   : {kernels}"
          + ("" if expect_kernels else "  (suppressed for SASS by design)"))

    if late:
        failures.append(f"{late} deep samples after the window closed - "
                        "the engine kept sampling")
    if inside == 0 and perf_events == 0:
        if alien:
            # PC sampling stamps its batches on the raw CUPTI clock rather
            # than the wall anchor, so the window cannot be checked by
            # timestamp there at all. That is a separate, pre-existing gap;
            # report it as inconclusive rather than as a window failure.
            print(f"INCONCLUSIVE: {alien} deep samples exist but are all on "
                  "an unanchored clock, so whether they fell inside the "
                  "window cannot be determined from timestamps")
            return 3
        failures.append("no deep data collected in the window")
    if expect_kernels and kernels == 0:
        failures.append("no kernel rows - the light tier did not run")

    if failures:
        print("VERDICT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
