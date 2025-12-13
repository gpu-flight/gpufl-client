from __future__ import annotations

from typing import Optional, Sequence, Union

def _require_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except ImportError as e:
        raise ImportError("gpufl.viz requires pandas. Install with: pip install gpufl[viz]") from e


def summarize_kernels(df, top_n: int = 20):
    """Return a per-kernel summary DataFrame (total time, count, avg).

    Expects normalized columns: event_type, name, duration_ns.
    """
    pd = _require_pandas()
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["name", "count", "total_ms", "avg_ms", "p95_ms"])

    d = df.copy()
    if "event_type" in d.columns:
        d = d[d["event_type"] == "kernel"]

    if "duration_ns" not in d.columns:
        if "start_ns" in d.columns and "end_ns" in d.columns:
            d["duration_ns"] = d["end_ns"] - d["start_ns"]

    d = d.dropna(subset=["name", "duration_ns"])
    d["duration_ms"] = d["duration_ns"] / 1e6

    g = d.groupby("name")["duration_ms"]
    out = pd.DataFrame({
        "count": g.size(),
        "total_ms": g.sum(),
        "avg_ms": g.mean(),
        "p95_ms": g.quantile(0.95),
    }).reset_index()

    out = out.sort_values("total_ms", ascending=False).head(top_n)
    return out


def summarize_scopes(df, top_n: int = 20):
    """Return a per-scope summary DataFrame.

    Supports real gpufl scope logs:
      - type == 'scope_begin' / 'scope_end' (+ optional 'scope_sample')
      - scope_end may include (ts_start_ns, ts_end_ns) OR just (ts_ns) with scope_id.
    """
    pd = _require_pandas()
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["name", "tag", "count", "total_ms", "avg_ms", "p95_ms"])

    d = df.copy()

    # normalize column naming
    if "type" not in d.columns and "event_type" in d.columns:
        d["type"] = d["event_type"]

    if "type" not in d.columns:
        return pd.DataFrame(columns=["name", "tag", "count", "total_ms", "avg_ms", "p95_ms"])

    begins = d[d["type"] == "scope_begin"].copy()
    ends = d[d["type"] == "scope_end"].copy()

    if len(ends) == 0:
        return pd.DataFrame(columns=["name", "tag", "count", "total_ms", "avg_ms", "p95_ms"])

    # Build interval rows (name/tag/start_ns/end_ns)
    if ("ts_start_ns" in ends.columns) and ("ts_end_ns" in ends.columns):
        scopes = pd.DataFrame({
            "name": ends["name"] if "name" in ends.columns else "scope",
            "tag": ends["tag"] if "tag" in ends.columns else "",
            "start_ns": ends["ts_start_ns"],
            "end_ns": ends["ts_end_ns"],
        })
    else:
        # Need pairing via scope_id
        if "scope_id" not in begins.columns or "scope_id" not in ends.columns:
            return pd.DataFrame(columns=["name", "tag", "count", "total_ms", "avg_ms", "p95_ms"])
        if "ts_ns" not in begins.columns or "ts_ns" not in ends.columns:
            return pd.DataFrame(columns=["name", "tag", "count", "total_ms", "avg_ms", "p95_ms"])

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

    scopes["start_ns"] = pd.to_numeric(scopes["start_ns"], errors="coerce")
    scopes["end_ns"] = pd.to_numeric(scopes["end_ns"], errors="coerce")
    scopes = scopes.dropna(subset=["name", "start_ns", "end_ns"]).copy()
    scopes = scopes[scopes["end_ns"] >= scopes["start_ns"]]

    if len(scopes) == 0:
        return pd.DataFrame(columns=["name", "tag", "count", "total_ms", "avg_ms", "p95_ms"])

    scopes["duration_ms"] = (scopes["end_ns"] - scopes["start_ns"]) / 1e6
    if "tag" not in scopes.columns:
        scopes["tag"] = ""

    g = scopes.groupby(["name", "tag"])["duration_ms"]
    out = pd.DataFrame({
        "count": g.size(),
        "total_ms": g.sum(),
        "avg_ms": g.mean(),
        "p95_ms": g.quantile(0.95),
    }).reset_index()

    out = out.sort_values("total_ms", ascending=False).head(top_n)
    return out