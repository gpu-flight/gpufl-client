from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

# ---- Schema normalization ----
# We normalize a variety of possible log keys to a small common set:
#   event_type: "kernel" | "scope" | "sample" | ...
#   name: str
#   tag: str (optional)
#   start_ns: int (optional)
#   end_ns: int (optional)
#   ts_ns: int (for instantaneous samples, optional)
#   pid, tid, device_id, gpu_uuid, grid, block, bytes, status, ...
#
# Unknown keys are preserved.

_START_KEYS = ("start_ns", "ts_start_ns", "t_start_ns", "begin_ns", "startTimeNs", "start")
_END_KEYS = ("end_ns", "ts_end_ns", "t_end_ns", "finish_ns", "endTimeNs", "end")
_TS_KEYS = ("ts_ns", "timestamp_ns", "time_ns", "ts")

_TYPE_KEYS = ("event_type", "type", "event", "kind")
_NAME_KEYS = ("name", "kernel", "scope", "op", "event_name")
_TAG_KEYS = ("tag", "label", "phase")

def _first_present(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None

    try:
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
            return int(float(v))
        return int(v)
    except ValueError:
        return None

def normalize_event(raw: Dict[str, Any]) -> Dict[str, Any]:

    e = dict(raw)

    et = _first_present(e, _TYPE_KEYS)
    if isinstance(et, str):
        e["event_type"] = et.lower()
    elif et is not None:
        e["event_type"] = str(et).lower()

    name = _first_present(e, _NAME_KEYS)
    if name is not None:
        e["name"] = str(name)

    tag = _first_present(e, _TAG_KEYS)
    if tag is not None:
        e["tag"] = str(tag)

    start = _to_int(_first_present(e, _START_KEYS))
    end = _to_int(_first_present(e, _END_KEYS))
    ts = _to_int(_first_present(e, _TS_KEYS))

    if start is not None:
        e["start_ns"] = start
    if end is not None:
        e["end_ns"] = end
    if ts is not None:
        e["ts_ns"] = ts

    if "duration_ns" not in e and start is not None and end is not None and end >= start:
        e["duration_ns"] = end - start

    if "event_type" not in e:
        if start is not None and end is not None and ("grid" in e or "block" in e or "gx" in e or "bx" in e):
            e["event_type"] = "kernel"

    return e

def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Allow comments
            if line.startswith("#"):
                continue
            # Only JSON objects per line
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                continue


def read_events(paths: Union[str, Sequence[str]]) -> List[Dict[str, Any]]:
    """Read gpufl log events from one or many files.

    Args:
      paths: A file path, a glob (e.g., "logs/*.jsonl"), or a list of paths/globs.

    Returns:
      List of normalized event dicts.
    """
    if isinstance(paths, str):
        patterns = [paths]
    else:
        patterns = list(paths)

    files: List[str] = []
    for p in patterns:
        expanded = glob.glob(p)
        if expanded:
            files.extend(sorted(expanded))
        elif os.path.exists(p):
            files.append(p)

    events: List[Dict[str, Any]] = []
    for fp in files:
        for raw in iter_jsonl(fp):
            events.append(normalize_event(raw))
    return events


def read_df(paths: Union[str, Sequence[str]]):
    """Read events and return a pandas DataFrame.

    Requires pandas. Install with:
      pip install gpufl[viz]
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError("gpufl.viz requires pandas. Install with: pip install gpufl[viz]") from e

    events = read_events(paths)
    df = pd.DataFrame(events)

    if "duration_ns" not in df.columns:
        if "start_ns" in df.columns and "end_ns" in df.columns:
            df["duration_ns"] = df["end_ns"] - df["start_ns"]

    for col in ("start_ns", "end_ns", "ts_ns", "duration_ns"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



