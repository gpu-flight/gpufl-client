#!/usr/bin/env python3
"""Check one completed AMD example session's timeline clock without uploading it.

Usage: python3 example/amd/verify_trace_timestamps.py <session-log-directory>
Static ISA mappings are checked for duplicate delivery, not for timestamps.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


TIMED_BATCHES = {
    "kernel_event_batch", "memcpy_event_batch", "memory_alloc_event_batch",
    "synchronization_event_batch", "scope_event_batch", "pm_sample_batch",
}


def validate(events):
    session_ids = {e["session_id"] for e in events if "session_id" in e}
    if len(session_ids) != 1:
        raise ValueError("Provide the log directory for exactly one session")
    starts = [e["ts_ns"] for e in events if e["type"] == "job_start"]
    ends = [e["ts_ns"] for e in events if e["type"] == "shutdown"]
    if not starts or not ends:
        raise ValueError("Expected job_start and shutdown in a completed capture")
    # Tracing can start before job_start is emitted, including HIP's internal
    # allocations. A one-second margin admits setup, not a different epoch.
    lower, upper = min(starts) - 1_000_000_000, max(ends) + 1_000_000_000
    static_batches = set()
    counts = Counter()
    timestamps = []
    for event in events:
        kind = event["type"]
        if kind == "profile_sample_batch":
            rows = [dict(zip(event["columns"], values)) for values in event["rows"]]
            if any(row.get("sample_kind") == 2 for row in rows):
                key = (event["session_id"], event["batch_id"])
                if key in static_batches:
                    raise ValueError("Duplicate static ISA batch across log channels")
                static_batches.add(key)
        if kind not in TIMED_BATCHES:
            continue
        for values in event["rows"]:
            row = dict(zip(event["columns"], values))
            start = event["base_time_ns"] + row["dt_ns"]
            duration = row.get("duration_ns", 0)
            if duration < 0 or not lower <= start <= start + duration <= upper:
                raise ValueError(f"{kind}: timestamp {start} / duration {duration} outside the session clock")
            counts[kind] += 1
            timestamps.extend((start, start + duration))
    if not counts:
        raise ValueError("No timed activity rows found")
    return counts, (max(timestamps) - min(timestamps)) / 1_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    events = []
    for path in sorted(args.session_dir.iterdir()):
        if path.name.endswith(".log.gz"):
            opener = gzip.open
        elif path.name.endswith(".log"):
            opener = open
        else:
            continue
        with opener(path, "rt", encoding="utf-8") as stream:
            events.extend(json.loads(line) for line in stream if line.strip())
    counts, span_ms = validate(events)
    for kind, count in sorted(counts.items()):
        print(f"{kind}: {count} rows")
    print(f"PASS: all timed rows share the session epoch; activity span {span_ms:.3f} ms")


if __name__ == "__main__":
    main()
