from __future__ import annotations
import json

from typing import Iterable, Optional

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ImportError as e:
        raise ImportError("gpufl.viz plotting requires matplotlib. Install with: pip install gpufl[viz]") from e

def _require_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except ImportError as e:
        raise ImportError("gpufl.viz requires pandas. Install with: pip install gpufl[viz]") from e


def _coerce_devices_cell(x):
    """
    devices column is ideally a list[dict], but sometimes may be a JSON string.
    Return list[dict] or [].
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def _explode_device_samples(df, *, sample_types: Iterable[str], gpu_id: Optional[int] = 0):
    """
    Build a flat dataframe with one row per (sample event, device).
    Expected event schema (from your logs):
      - type: 'scope_sample' or 'system_sample'
      - ts_ns: timestamp
      - devices: list of dicts containing used_mib, util_gpu, etc. :contentReference[oaicite:2]{index=2}
    """
    pd = _require_pandas()

    if df is None or len(df) == 0:
        return pd.DataFrame()

    d = df.copy()

    # Normalize column naming: prefer 'type' (your log), fallback to 'event_type'
    if "type" not in d.columns and "event_type" in d.columns:
        d["type"] = d["event_type"]

    if "type" not in d.columns:
        raise ValueError("Expected a 'type' column (e.g., scope_sample/system_sample).")

    if "ts_ns" not in d.columns:
        raise ValueError("Expected 'ts_ns' for sample events.")

    d = d[d["type"].isin(list(sample_types))].copy()
    if len(d) == 0:
        return pd.DataFrame()

    if "devices" not in d.columns:
        raise ValueError("Expected 'devices' column for sample events.")

    d["devices"] = d["devices"].apply(_coerce_devices_cell)

    rows = []
    for _, r in d.iterrows():
        ts_ns = r.get("ts_ns")
        if ts_ns is None:
            continue
        for dev in r["devices"]:
            if not isinstance(dev, dict):
                continue
            if gpu_id is not None and dev.get("id") != gpu_id:
                continue
            rows.append({
                "ts_ns": ts_ns,
                "sample_type": r.get("type"),
                "scope_name": r.get("name", ""),  # for scope_sample
                "scope_tag": r.get("tag", ""),
                "pid": r.get("pid"),
                "gpu_id": dev.get("id"),
                "gpu_name": dev.get("name", ""),
                "used_mib": dev.get("used_mib"),
                "free_mib": dev.get("free_mib"),
                "total_mib": dev.get("total_mib"),
                "util_gpu": dev.get("util_gpu"),
                "util_mem": dev.get("util_mem"),
                "temp_c": dev.get("temp_c"),
                "power_mw": dev.get("power_mw"),
                "clk_gfx": dev.get("clk_gfx"),
                "clk_sm": dev.get("clk_sm"),
                "clk_mem": dev.get("clk_mem"),
            })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out["ts_ns"] = pd.to_numeric(out["ts_ns"], errors="coerce")
    out = out.dropna(subset=["ts_ns"]).sort_values("ts_ns").reset_index(drop=True)
    out["t_s"] = (out["ts_ns"] - out["ts_ns"].min()) / 1e9
    return out

def plot_kernel_timeline(df, title: str = "GPU Flight — Kernel Timeline", max_events: int = 2000):
    """Plot kernel durations over time (scatter).

    Uses start_ns as x-axis (relative) and duration_ms as y-axis.
    """
    pd = _require_pandas()
    plt = _require_matplotlib()

    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe; nothing to plot.")

    d = df.copy()
    if "event_type" in d.columns:
        d = d[d["event_type"] == "kernel"]

    if len(d) == 0:
        raise ValueError("No kernel events found (event_type == 'kernel').")

    if "duration_ns" not in d.columns:
        d["duration_ns"] = d["end_ns"] - d["start_ns"]

    d = d.dropna(subset=["start_ns", "duration_ns"])
    d["t_ms"] = (d["start_ns"] - d["start_ns"].min()) / 1e6
    d["duration_ms"] = d["duration_ns"] / 1e6

    if len(d) > max_events:
        d = d.sort_values("start_ns").tail(max_events)

    fig = plt.figure()
    plt.scatter(d["t_ms"], d["duration_ms"])
    plt.xlabel("Time since first event (ms)")
    plt.ylabel("Kernel duration (ms)")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_scope_timeline(
        df,
        title: str = "GPU Flight — Scope Timeline",
        max_scopes: int = 500,
):
    """
    Plot scope durations as a simple Gantt-like chart.

    Supports logs where:
      - type == 'scope_begin' with ts_ns and scope_id
      - type == 'scope_end' with either:
          (A) ts_start_ns + ts_end_ns (preferred), OR
          (B) ts_ns (end only) + scope_id (paired with begin)
      - type == 'scope_sample' is ignored for bars
    """
    pd = _require_pandas()
    plt = _require_matplotlib()

    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe; nothing to plot.")

    d = df.copy()

    # Normalize column name: some sources might use 'event_type'
    if "type" not in d.columns and "event_type" in d.columns:
        d["type"] = d["event_type"]

    if "type" not in d.columns:
        raise ValueError("Missing 'type' column (expected scope_begin/scope_end/scope_sample).")

    begins = d[d["type"] == "scope_begin"].copy()
    ends = d[d["type"] == "scope_end"].copy()

    if len(ends) == 0:
        raise ValueError("No scope_end events found.")

    # --- Build a clean scopes table with UNIQUE columns only ---
    if ("ts_start_ns" in ends.columns) and ("ts_end_ns" in ends.columns):
        # Path A: scope_end already includes both timestamps.
        scopes = pd.DataFrame({
            "scope_id": ends["scope_id"] if "scope_id" in ends.columns else None,
            "name": ends["name"] if "name" in ends.columns else "scope",
            "tag": ends["tag"] if "tag" in ends.columns else "",
            "start_ns": ends["ts_start_ns"],
            "end_ns": ends["ts_end_ns"],
        })
    else:
        # Path B: pair begin/end using scope_id and ts_ns.
        if "scope_id" not in begins.columns or "scope_id" not in ends.columns:
            raise ValueError(
                "To pair scope_begin/scope_end, both must include 'scope_id', "
                "or scope_end must include 'ts_start_ns' and 'ts_end_ns'."
            )
        if "ts_ns" not in begins.columns or "ts_ns" not in ends.columns:
            raise ValueError("Expected ts_ns in scope_begin/scope_end to pair scopes.")

        b = pd.DataFrame({
            "scope_id": begins["scope_id"],
            "name": begins["name"] if "name" in begins.columns else "scope",
            "tag": begins["tag"] if "tag" in begins.columns else "",
            "start_ns": begins["ts_ns"],
        })

        e = pd.DataFrame({
            "scope_id": ends["scope_id"],
            "end_ns": ends["ts_ns"],
        })

        scopes = b.merge(e, on="scope_id", how="inner")

    # Force numeric so comparisons are safe even if fields are strings
    scopes["start_ns"] = pd.to_numeric(scopes["start_ns"], errors="coerce")
    scopes["end_ns"] = pd.to_numeric(scopes["end_ns"], errors="coerce")

    scopes = scopes.dropna(subset=["start_ns", "end_ns"]).copy()
    scopes = scopes.sort_values("start_ns").head(max_scopes)

    # Now safe: both are Series with unique names
    scopes = scopes[scopes["end_ns"] >= scopes["start_ns"]]

    if len(scopes) == 0:
        raise ValueError("No complete scopes found to plot (begin/end pairing failed).")

    t0 = scopes["start_ns"].min()
    scopes["start_ms"] = (scopes["start_ns"] - t0) / 1e6
    scopes["dur_ms"] = (scopes["end_ns"] - scopes["start_ns"]) / 1e6

    # Build labels
    scopes = scopes.reset_index(drop=True)
    name = scopes["name"].fillna("scope").astype(str)
    tag = scopes["tag"].fillna("").astype(str)
    labels = name.where(tag == "", name + " [" + tag + "]")

    y = list(range(len(scopes)))

    fig = plt.figure(figsize=(10, max(3, len(scopes) * 0.28)))
    plt.barh(y, scopes["dur_ms"], left=scopes["start_ms"])
    plt.yticks(y, labels)
    plt.xlabel("Time (ms)")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_memory_timeline(
        df,
        *,
        gpu_id: int = 0,
        title: str = "GPU Flight — Memory Usage (MiB)",
        sample_types: tuple[str, ...] = ("scope_sample", "system_sample"),
        only_scope: Optional[str] = None,
        only_tag: Optional[str] = None,
):
    """
    Time-series plot of used VRAM (MiB) vs time.

    Uses scope_sample/system_sample events with devices[].used_mib. :contentReference[oaicite:3]{index=3}
    """
    pd = _require_pandas()
    plt = _require_matplotlib()

    s = _explode_device_samples(df, sample_types=sample_types, gpu_id=gpu_id)
    if len(s) == 0:
        raise ValueError(f"No samples found for types={sample_types} and gpu_id={gpu_id}.")

    if only_scope is not None:
        s = s[s["scope_name"] == only_scope]
    if only_tag is not None:
        s = s[s["scope_tag"] == only_tag]
    if len(s) == 0:
        raise ValueError("No samples left after filtering (only_scope/only_tag).")

    s["used_mib"] = pd.to_numeric(s["used_mib"], errors="coerce")
    s = s.dropna(subset=["used_mib"])

    fig = plt.figure()
    plt.plot(s["t_s"], s["used_mib"])
    plt.xlabel("Time (s)")
    plt.ylabel("Used VRAM (MiB)")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_utilization_timeline(
        df,
        *,
        gpu_id: int = 0,
        title: str = "GPU Flight — Utilization (%)",
        sample_types: tuple[str, ...] = ("scope_sample", "system_sample"),
        only_scope: Optional[str] = None,
        only_tag: Optional[str] = None,
        include_mem: bool = True,
):
    """
    Time-series plot of utilization vs time.

    Uses devices[].util_gpu (and optionally util_mem). :contentReference[oaicite:4]{index=4}
    """
    pd = _require_pandas()
    plt = _require_matplotlib()

    s = _explode_device_samples(df, sample_types=sample_types, gpu_id=gpu_id)
    if len(s) == 0:
        raise ValueError(f"No samples found for types={sample_types} and gpu_id={gpu_id}.")

    if only_scope is not None:
        s = s[s["scope_name"] == only_scope]
    if only_tag is not None:
        s = s[s["scope_tag"] == only_tag]
    if len(s) == 0:
        raise ValueError("No samples left after filtering (only_scope/only_tag).")

    s["util_gpu"] = pd.to_numeric(s["util_gpu"], errors="coerce")
    if include_mem:
        s["util_mem"] = pd.to_numeric(s["util_mem"], errors="coerce")

    fig = plt.figure()
    plt.plot(s["t_s"], s["util_gpu"], label="util_gpu")
    if include_mem and "util_mem" in s.columns:
        plt.plot(s["t_s"], s["util_mem"], label="util_mem")

    plt.xlabel("Time (s)")
    plt.ylabel("Utilization (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig