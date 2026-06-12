"""NDJSON log reader for the visualization layer.

The C++ client emits two flavors of records into the .log files:

  1. **Lifecycle / one-off events** - `job_start`, `shutdown`, `sass_config`,
     `dictionary_update`. One record = one event.

  2. **Columnar batch events** - `device_metric_batch`,
     `host_metric_batch`, `scope_event_batch`, `kernel_event_batch`,
     `memcpy_event_batch`, `profile_sample_batch`,
     `synchronization_event_batch`, `memory_alloc_event_batch`. Each
     record is an envelope with `columns:[...]`, `rows:[[...],[...]]`,
     and a `base_time_ns` from which per-row `dt_ns` deltas are
     reconstructed.

The visualization layer wants flat per-sample dicts with absolute
`ts_ns`, so this module:

  - parses NDJSON lines,
  - explodes every batch row into a per-sample dict with `ts_ns =
    base_time_ns + dt_ns`,
  - resolves dictionary IDs (`kernel_id` → kernel name, `name_id` →
    scope name) from `dictionary_update` events so downstream code
    sees human-readable strings.

Pre-v1.0.0 this file did none of the above and silently dropped every
batch row - which is why the legacy `gpufl.viz` plots looked empty.
"""

import glob
import gzip
import json
import os
from typing import Any, Dict, List

import pandas as pd


# Map from batch envelope `type` to the per-sample synthetic `type` that
# downstream code (timeline.py / visualizer.py) keys off after expansion.
_BATCH_TYPE_MAP: Dict[str, str] = {
    "device_metric_batch":          "device_metric_sample",
    "host_metric_batch":            "host_metric_sample",
    "kernel_event_batch":           "kernel_event",
    "scope_event_batch":            "scope_event",
    "memcpy_event_batch":           "memcpy_event",
    "profile_sample_batch":         "profile_sample",
    "synchronization_event_batch":  "synchronization_event",
    "memory_alloc_event_batch":     "memory_alloc_event",
}

# A few batch columns share a name with viz's own conventions; rename on
# expansion to avoid collisions. Today the only case is `event_type` on
# `scope_event_batch` (0=begin, 1=end) clashing with viz's top-level
# event-type column.
_COLUMN_RENAMES: Dict[str, str] = {
    "event_type": "phase",
}


def _parse_line(line: str) -> dict:
    try:
        return json.loads(line)
    except Exception:
        return {}


def _build_dictionaries(raw_events: List[dict]) -> Dict[str, Dict[int, str]]:
    """Aggregate every `dictionary_update` event into id→name maps.

    Dictionary updates can arrive incrementally across the session; this
    folds them all into one map per kind. The C++ side guarantees the
    relevant dict update is emitted before any batch that references its
    ids, so by the time we resolve ids during expansion, every id we'll
    encounter is already in the merged map.
    """
    scope_names: Dict[int, str] = {}
    kernel_names: Dict[int, str] = {}
    function_names: Dict[int, str] = {}
    metric_names: Dict[int, str] = {}
    for evt in raw_events:
        if evt.get("type") != "dictionary_update":
            continue
        # Dictionary keys are JSON strings - coerce to int for lookup.
        for src, dst in (
            ("scope_name_dict", scope_names),
            ("kernel_dict",     kernel_names),
            ("function_dict",   function_names),
            ("metric_dict",     metric_names),
        ):
            for k, v in evt.get(src, {}).items():
                try:
                    dst[int(k)] = v
                except (TypeError, ValueError):
                    pass
    return {
        "scope": scope_names,
        "kernel": kernel_names,
        "function": function_names,
        "metric": metric_names,
    }


def _expand_batch(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Explode a `*_batch` envelope into a list of per-sample dicts.

    Per-sample dicts get:
      - `type` set to the synthetic per-sample name (e.g.
        `device_metric_sample`),
      - every envelope field that isn't `columns`/`rows`/`base_time_ns`
        (so `session_id`, `app`, `batch_id`, etc. propagate),
      - every column from the batch's `columns` header as a top-level
        field (with `event_type` renamed to `phase` to avoid clash with
        viz's own column),
      - `ts_ns` reconstructed as `base_time_ns + dt_ns`.

    Non-batch events pass through unchanged in a single-element list.
    """
    batch_type = event.get("type", "")
    if batch_type not in _BATCH_TYPE_MAP:
        return [event]
    if "columns" not in event or "rows" not in event:
        return [event]

    columns = event["columns"]
    rows = event["rows"]
    base_ns = event.get("base_time_ns", 0) or 0
    sample_type = _BATCH_TYPE_MAP[batch_type]

    envelope = {
        k: v for k, v in event.items()
        if k not in ("type", "columns", "rows", "base_time_ns")
    }

    expanded: List[Dict[str, Any]] = []
    for row in rows:
        sample = dict(envelope)
        sample["type"] = sample_type
        for col, val in zip(columns, row):
            sample[_COLUMN_RENAMES.get(col, col)] = val
        if "dt_ns" in sample:
            sample["ts_ns"] = base_ns + sample["dt_ns"]
        expanded.append(sample)
    return expanded


def _resolve_names(sample: Dict[str, Any],
                   dicts: Dict[str, Dict[int, str]]) -> None:
    """Mutate a sample to add human-readable `name` from dictionary ids.

    - `scope_event` rows get `name` from `scope_name_dict` via `name_id`.
    - `kernel_event` rows get `name` from `kernel_dict` via `kernel_id`.

    Unresolved ids fall back to `"#<id>"` so the user sees *something*
    rather than NaN.
    """
    stype = sample.get("type")
    if stype == "scope_event" and "name_id" in sample:
        nid = sample["name_id"]
        sample["name"] = dicts["scope"].get(nid, f"#{nid}")
    elif stype == "kernel_event" and "kernel_id" in sample:
        kid = sample["kernel_id"]
        sample["name"] = dicts["kernel"].get(kid, f"#{kid}")


def _gather_files(file_pattern: str) -> List[str]:
    """Resolve a path / dir / glob into a concrete list of log files.

    Understands the current client layout (gzipped, session-nested):
      * a directory          -> all `<dir>/**/*.log[.gz]` (recursive)
      * a `*.log` glob/path  -> that glob *plus* its `.gz` siblings
      * an explicit `*.gz`   -> as given

    so callers can pass `read_df("logs/")`, `read_df("logs/scope.*.log")`,
    or `read_df("logs/run/scope.log.gz")` interchangeably.
    """
    if os.path.isdir(file_pattern):
        pats = [os.path.join(file_pattern, "**", "*.log"),
                os.path.join(file_pattern, "**", "*.log.gz")]
    elif file_pattern.endswith(".gz"):
        pats = [file_pattern]
    else:
        pats = [file_pattern, file_pattern + ".gz"]

    out: List[str] = []
    seen = set()
    for pat in pats:
        for f in glob.glob(pat, recursive=True):
            if f not in seen and os.path.isfile(f):
                seen.add(f)
                out.append(f)
    return out


def read_events(file_pattern: str) -> List[dict]:
    """Read NDJSON files and return one dict per *expanded* event.

    Batch envelopes are exploded into per-sample dicts with absolute
    `ts_ns` and resolved name fields where dictionary ids exist. Non-
    batch lines (lifecycle, dictionary_update, sass_config) pass through
    unchanged so downstream code can still see them if it needs to.

    Accepts a file path, a directory, or a glob; gzipped (`.log.gz`) and
    session-nested logs written by the current client are handled.
    """
    files = _gather_files(file_pattern)

    # Pass 1: parse every line. We have to buffer to do dictionary
    # resolution in pass 2 - `dictionary_update` events are usually
    # emitted before the batches that reference them, but a defensive
    # two-pass approach handles any out-of-order edge case.
    raw_events: List[dict] = []
    for fpath in files:
        if not os.path.isfile(fpath):
            continue
        open_fn = gzip.open if fpath.endswith(".gz") else open
        with open_fn(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                evt = _parse_line(line)
                if evt:
                    raw_events.append(evt)

    dicts = _build_dictionaries(raw_events)

    # Pass 2: expand and resolve names.
    expanded: List[dict] = []
    for evt in raw_events:
        for sample in _expand_batch(evt):
            _resolve_names(sample, dicts)
            expanded.append(sample)
    return expanded


def read_df(file_pattern: str) -> pd.DataFrame:
    """Read NDJSON logs into a flat per-sample pandas DataFrame.

    Columns vary by which event types are present. Common ones after
    expansion:
      - `ts_ns` (always for time-series rows; reconstructed from
        `base_time_ns + dt_ns` for batch rows)
      - `type` (synthetic per-sample name)
      - per-metric columns: `gpu_util`, `mem_util`, `temp_c`,
        `power_mw`, `used_mib`, `total_mib`, `clock_sm`, plus
        extended ones when present (`energy_uj`, `pcie_bw_bps`, ...)
      - per-row columns from kernel/scope/memcpy batches with names
        resolved into `name` via the dictionary maps.
    """
    events = read_events(file_pattern)
    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)

    # Ensure ts_ns is present and numeric, filling from common
    # alternate timestamp fields for record kinds that don't use
    # ts_ns directly (kernel intervals use start_ns; scope-end legacy
    # records use ts_end_ns).
    if "ts_ns" not in df.columns:
        df["ts_ns"] = pd.Series([None] * len(df), dtype="float64")
    if "ts_start_ns" in df.columns:
        df["ts_ns"] = df["ts_ns"].fillna(df["ts_start_ns"])
    if "start_ns" in df.columns:
        df["ts_ns"] = df["ts_ns"].fillna(df["start_ns"])

    for c in ("ts_ns", "start_ns", "end_ns", "ts_start_ns", "ts_end_ns"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("ts_ns").reset_index(drop=True)
    return df
