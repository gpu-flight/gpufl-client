#!/usr/bin/env python3
"""Verify the injected early-memcpy regression capture from raw NDJSON logs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable, TextIO


EXPECTED_BYTES = 4_194_304


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def events(root: Path) -> Iterable[dict]:
    for path in sorted(root.rglob("*.log")) + sorted(root.rglob("*.log.gz")):
        with open_text(path) as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    args = parser.parse_args()

    matching = {1: [], 2: []}
    batch_count = 0
    kernel_rows = 0
    synchronization_rows = 0
    for event in events(args.log_root):
        event_type = event.get("type")
        if event_type == "kernel_event_batch":
            kernel_rows += len(event.get("rows", []))
            continue
        if event_type == "synchronization_event_batch":
            synchronization_rows += len(event.get("rows", []))
            continue
        if event_type != "memcpy_event_batch":
            continue
        batch_count += 1
        columns = event.get("columns", [])
        try:
            bytes_index = columns.index("bytes")
            kind_index = columns.index("copy_kind")
        except ValueError as exc:
            raise RuntimeError("memcpy_event_batch is missing bytes/copy_kind") from exc
        for row in event.get("rows", []):
            kind = int(row[kind_index])
            size = int(row[bytes_index])
            if kind in matching and size == EXPECTED_BYTES:
                matching[kind].append(size)

    print(
        "memcpy capture:"
        f" batches={batch_count}"
        f" H2D_count={len(matching[1])} H2D_bytes={sum(matching[1])}"
        f" D2H_count={len(matching[2])} D2H_bytes={sum(matching[2])}"
        f" kernel_rows={kernel_rows}"
        f" synchronization_rows={synchronization_rows}"
    )
    if matching[1] != [EXPECTED_BYTES] or matching[2] != [EXPECTED_BYTES]:
        print("VERIFY FAIL: expected exactly one 4 MiB H2D and one 4 MiB D2H")
        return 1
    if kernel_rows < 1 or synchronization_rows < 1:
        print("VERIFY FAIL: expected kernel and synchronization capture to remain active")
        return 1
    print("VERIFY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
