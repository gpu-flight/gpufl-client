from __future__ import annotations
import json
from typing import Iterable, Optional

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("Visualization requires matplotlib.")

def _require_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError("Visualization requires pandas.")

# ==========================================
# 1. HELPERS
# ==========================================

def _ensure_event_type_col(df):
    if df is None: return df
    if "event_type" not in df.columns and "type" in df.columns:
        df = df.copy()
        df["event_type"] = df["type"]
    return df

def _coerce_devices_cell(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except: return []
    return []

def _coerce_host_cell(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except: return {}
    return {}

def _explode_device_samples(df, gpu_id=0):
    """Extract per-sample GPU metrics from `device_metric_sample` rows.

    The reader (reader.py) has already exploded `device_metric_batch`
    envelopes into flat per-sample dicts with absolute `ts_ns`, so this
    function just filters by event type and remaps the wire column
    names to the legacy field names the plotters expect. Columns that
    only appear in the extended-metric batch (e.g. `pcie_bw_bps`,
    `energy_uj`) are surfaced when present, NaN otherwise.
    """
    pd = _require_pandas()
    df = _ensure_event_type_col(df)
    if "event_type" not in df.columns: return pd.DataFrame()

    d = df[df["event_type"] == "device_metric_sample"].copy()
    if d.empty: return pd.DataFrame()
    if "device_id" in d.columns:
        d = d[d["device_id"] == gpu_id]
    if d.empty: return pd.DataFrame()

    # Map columnar wire field names → legacy field names used by plotters.
    # PCIe split into RX/TX was lost when the wire format moved to
    # the single `pcie_bw_bps` column; we surface only the combined rate.
    def _get(col, default=0):
        if col in d.columns:
            return pd.to_numeric(d[col], errors="coerce").fillna(default)
        return pd.Series([default] * len(d), index=d.index)

    out = pd.DataFrame({
        "ts_ns":        pd.to_numeric(d["ts_ns"], errors="coerce"),
        "util_gpu_pct": _get("gpu_util"),
        "util_mem_pct": _get("mem_util"),
        "used_mib":     _get("used_mib"),
        "total_mib":    _get("total_mib"),
        "temp_c":       _get("temp_c"),
        "power_mw":     _get("power_mw"),
        "clk_sm_mhz":   _get("clock_sm"),
        # Combined PCIe bandwidth (RX+TX) — the wire format no longer
        # splits direction; if absent (base shape, not extended), this
        # stays 0.
        "pcie_gbps":    _get("pcie_bw_bps") / 1e9,
        # Cumulative energy counter from the extended shape; users
        # plotting this should differentiate w.r.t. time to get power-
        # like behavior. Plotted as-is here so the column is available.
        "energy_uj":    _get("energy_uj"),
    })
    out = out.dropna(subset=["ts_ns"]).sort_values("ts_ns")
    if not out.empty:
        out["t_s_abs"] = (out["ts_ns"] - out["ts_ns"].min()) / 1e9
    return out

def _explode_host_samples(df):
    """Extract per-sample host metrics from `host_metric_sample` rows.

    Per-sample dicts come from `host_metric_batch` expansion (handled in
    reader.py). The wire stores CPU utilization as `cpu_pct_x100`
    (i.e. percent × 100 for two-decimal precision); divide by 100 here
    to restore the conventional 0–100 percent that plotters expect.
    """
    pd = _require_pandas()
    df = _ensure_event_type_col(df)
    if "event_type" not in df.columns: return pd.DataFrame()

    d = df[df["event_type"] == "host_metric_sample"].copy()
    if d.empty: return pd.DataFrame()

    def _get(col, default=0):
        if col in d.columns:
            return pd.to_numeric(d[col], errors="coerce").fillna(default)
        return pd.Series([default] * len(d), index=d.index)

    out = pd.DataFrame({
        "ts_ns":        pd.to_numeric(d["ts_ns"], errors="coerce"),
        "cpu_pct":      _get("cpu_pct_x100") / 100.0,
        "ram_used_mib": _get("ram_used_mib"),
        "ram_total_mib": _get("ram_total_mib"),
    })
    out = out.dropna(subset=["ts_ns"]).sort_values("ts_ns")
    if not out.empty:
        out["t_s_abs"] = (out["ts_ns"] - out["ts_ns"].min()) / 1e9
    return out

def _reconstruct_intervals(df, start_type, end_type, name_col="name", fallback_name="Scope"):
    pd = _require_pandas()
    # Support both "scope_start" and "scope_begin" for compatibility
    start_types = [start_type]
    if start_type == "scope_start":
        start_types.append("scope_begin")
    
    # [NEW] Handle single-event intervals like kernel_event
    is_kernel = (start_type == "kernel_start" or start_type == "kernel_event")
    if is_kernel:
        # Include kernel_event which has both start and end
        target_types = start_types + [end_type, "kernel_event"]
    else:
        target_types = start_types + [end_type]

    subset = df[df["event_type"].isin(target_types)].copy()
    if subset.empty: return []

    intervals = []
    # Use a dictionary of lists to handle multiple nested intervals with the same name
    stacks = {} 
    min_ts = df["ts_ns"].min()
    if pd.isna(min_ts):
        # try start_ns if ts_ns is all NaN
        if "start_ns" in df.columns:
            min_ts = df["start_ns"].min()
    if pd.isna(min_ts): min_ts = 0

    for _, r in subset.iterrows():
        etype = r["event_type"]
        name = r.get(name_col, fallback_name)
        if pd.isna(name): name = fallback_name

        if etype == "kernel_event" and "start_ns" in r and "end_ns" in r:
            start_ns = r["start_ns"]
            end_ns = r["end_ns"]
            if not pd.isna(start_ns) and not pd.isna(end_ns):
                start_sec = (start_ns - min_ts) / 1e9
                dur_sec = (end_ns - start_ns) / 1e9
                
                # Add extra metrics if present
                metrics = {
                    "occupancy": r.get("occupancy", 0),
                    "grid": r.get("grid", ""),
                    "block": r.get("block", ""),
                    "num_regs": r.get("num_regs", 0),
                    "dyn_shared": r.get("dyn_shared_bytes", 0),
                    "static_shared": r.get("static_shared_bytes", 0),
                }
                intervals.append((start_sec, dur_sec, name, metrics))
            continue

        ts = r.get("ts_ns")
        if pd.isna(ts): ts = r.get("ts_start_ns")
        if pd.isna(ts): ts = r.get("start_ns")
        if pd.isna(ts): continue

        if etype in start_types:
            if name not in stacks:
                stacks[name] = []
            stacks[name].append(ts)
        elif etype == end_type:
            if name in stacks and stacks[name]:
                start_ns = stacks[name].pop()
                start_sec = (start_ns - min_ts) / 1e9
                dur_sec = (ts - start_ns) / 1e9
                intervals.append((start_sec, dur_sec, name, {}))
                if not stacks[name]:
                    del stacks[name]
    return intervals

def _scope_intervals(df):
    """Pair `scope_event` begin/end rows into ordered intervals.

    The wire format emits two `scope_event` rows per scope — one with
    `phase=0` (begin) and one with `phase=1` (end), both sharing the
    same `scope_instance_id`. Pairing on the instance id (rather than
    on the scope name) is correct even for nested scopes with the same
    name.

    Falls back to an empty list if no `scope_event` rows exist (e.g.
    the session ran without `with gpufl.Scope(...)` blocks).
    """
    pd = _require_pandas()
    df = _ensure_event_type_col(df)
    if "event_type" not in df.columns:
        return []
    scopes = df[df["event_type"] == "scope_event"].copy()
    if scopes.empty or "scope_instance_id" not in scopes.columns:
        return []

    # Sort so begin precedes end within each instance.
    scopes = scopes.sort_values("ts_ns")

    min_ts = scopes["ts_ns"].min()
    if pd.isna(min_ts):
        min_ts = 0

    starts: dict = {}      # scope_instance_id -> (ts_ns, name)
    intervals = []
    for _, r in scopes.iterrows():
        inst = r.get("scope_instance_id")
        phase = r.get("phase")
        ts = r.get("ts_ns")
        name = r.get("name", f"#{r.get('name_id', '?')}")
        if pd.isna(ts) or pd.isna(inst):
            continue
        if phase == 0:
            starts[inst] = (ts, name)
        elif phase == 1 and inst in starts:
            start_ts, start_name = starts.pop(inst)
            start_sec = (start_ts - min_ts) / 1e9
            dur_sec = (ts - start_ts) / 1e9
            intervals.append((start_sec, dur_sec, start_name, {}))
    return intervals


def _kernel_intervals(df):
    """Extract kernel intervals from expanded `kernel_event` rows.

    Each row already carries `ts_ns` (start) and `duration_ns`, so the
    interval is a single-row read. `name` is dictionary-resolved by
    reader.py from `kernel_id`. Extra metadata fields (occupancy,
    grid/block, regs, shared) come from the kernel_detail records the
    backend merges separately; in the post-expansion DataFrame they
    won't be on the same row, so we surface what's available
    per-launch and leave occupancy off the plot until a kernel_detail
    join is added.
    """
    pd = _require_pandas()
    df = _ensure_event_type_col(df)
    if "event_type" not in df.columns:
        return []
    kernels = df[df["event_type"] == "kernel_event"].copy()
    if kernels.empty:
        return []

    kernels["ts_ns"] = pd.to_numeric(kernels["ts_ns"], errors="coerce")
    kernels["duration_ns"] = pd.to_numeric(
        kernels.get("duration_ns", 0), errors="coerce"
    ).fillna(0)
    kernels = kernels.dropna(subset=["ts_ns"])
    if kernels.empty:
        return []

    min_ts = kernels["ts_ns"].min()
    intervals = []
    for _, r in kernels.iterrows():
        start_sec = (r["ts_ns"] - min_ts) / 1e9
        dur_sec = r["duration_ns"] / 1e9
        name = r.get("name", f"#{r.get('kernel_id', '?')}")
        metrics = {
            "grid": r.get("grid", ""),
            "block": r.get("block", ""),
            "num_regs": r.get("num_regs", 0),
            "dyn_shared": r.get("dyn_shared", 0),
            "corr_id": r.get("corr_id", 0),
            # occupancy lives in kernel_detail (not the batch row); left
            # unset until reader.py joins kernel_detail by corr_id.
            "occupancy": 0,
        }
        intervals.append((start_sec, dur_sec, name, metrics))
    return intervals


# ==========================================
# 2. PLOTTERS
# ==========================================

def plot_combined_timeline(df, title="GPUFL Timeline"):
    pd = _require_pandas()
    plt = _require_matplotlib()

    df = _ensure_event_type_col(df)
    if "event_type" not in df.columns:
        print("[Viz] Error: No event_type column found.")
        return None

    min_ts = df["ts_ns"].min()
    if pd.isna(min_ts): min_ts = 0

    # --- Prepare Data ---
    # v1.0.0+ wire format: scopes come from expanded `scope_event` rows
    # paired by `scope_instance_id`; kernels from expanded
    # `kernel_event` rows that carry duration_ns directly. The legacy
    # `_reconstruct_intervals(scope_start/scope_end, ...)` API is kept
    # for any older-format logs but no longer the primary path.
    scope_data = _scope_intervals(df)
    if not scope_data:
        # Fall back to legacy event names for older log files.
        scope_data = _reconstruct_intervals(df, "scope_start", "scope_end")
        if not scope_data:
            scope_data = _reconstruct_intervals(df, "scope_begin", "scope_end")
        if not scope_data:
            scope_data = _reconstruct_intervals(
                df, "job_start", "shutdown",
                name_col="app", fallback_name="App",
            )

    kernel_data = _kernel_intervals(df)

    gpu_samples = _explode_device_samples(df, gpu_id=0)
    host_samples = _explode_host_samples(df)

    # --- Plotting (3 Rows) ---
    # Heights: GPU=2, PCIe=1.5, Host=2
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1.5, 2]})

    # --- Helper to Overlay Markers ---
    kernel_markers = [] # List of (vline, annotation)

    def overlay_markers(ax, y_lim_ref=None):
        """Draws vertical lines for Scopes and Kernels on the given axis."""
        # Get Y-limit to position text
        y_top = y_lim_ref if y_lim_ref else (ax.get_ylim()[1] if len(ax.get_lines()) > 0 else 100)

        # Scopes (Red dashed)
        if scope_data:
            for start_sec, dur_sec, name, _ in scope_data:
                ax.axvline(x=start_sec, color='tab:red', linestyle='--', alpha=0.6, linewidth=1)
                ax.text(start_sec, y_top * 0.95, name, rotation=90, va='top', ha='center', fontsize=7,
                        color='tab:red', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.3, edgecolor='none'))

        # Kernels (Orange solid/dashed)
        if kernel_data:
            for start_sec, dur_sec, name, metrics in kernel_data:
                end_sec = start_sec + (dur_sec if dur_sec is not None else 0)
                vl = ax.axvline(x=start_sec, color='tab:orange', linestyle='-', linewidth=1.2, picker=True)
                
                # Enrich text with occupancy if available
                display_name = name
                if metrics and metrics.get("occupancy", 0) > 0:
                    display_name += f" ({metrics['occupancy']*100:.1f}%)"
                
                # Create annotation but set it invisible by default
                ann = ax.annotate(display_name, xy=(start_sec, y_top * 0.85), 
                                  xytext=(5, 0), textcoords="offset points",
                                  rotation=90, va='top', ha='left', fontsize=7,
                                  color='tab:orange', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.8),
                                  visible=False)
                
                kernel_markers.append((vl, ann))

                if dur_sec and dur_sec > 0:
                    ax.axvline(x=end_sec, color='tab:orange', linestyle='--', linewidth=1.2)
                    # We usually don't need hover for "end" markers, but could add it.

    # --- Row 1: GPU Metrics ---
    if not gpu_samples.empty:
        t = gpu_samples["t_s_abs"]
        ax1.plot(t, gpu_samples["util_gpu_pct"], label="GPU %", color='tab:green')
        ax1.plot(t, gpu_samples["util_mem_pct"], label="Mem %", color='tab:purple', linestyle="--")

        # [NEW] Optional metrics from system log
        if "temp_c" in gpu_samples.columns and gpu_samples["temp_c"].max() > 0:
             ax1.plot(t, gpu_samples["temp_c"], label="Temp (C)", color='tab:red', alpha=0.3)
        if "clk_sm_mhz" in gpu_samples.columns and gpu_samples["clk_sm_mhz"].max() > 0:
             ax1.plot(t, gpu_samples["clk_sm_mhz"] / 10, label="SM Clock (x10 MHz)", color='tab:orange', alpha=0.3)

        # [NEW] Visualize Kernel Occupancy points on the timeline
        if kernel_data:
            k_t = [k[0] for k in kernel_data if k[3] and k[3].get("occupancy", 0) > 0]
            k_occ = [k[3]["occupancy"] * 100 for k in kernel_data if k[3] and k[3].get("occupancy", 0) > 0]
            if k_t:
                ax1.scatter(k_t, k_occ, color='tab:orange', marker='o', s=20, label="Kernel Occupancy", zorder=5)

        ax1.set_ylabel("GPU Util %")
        ax1.set_ylim(-5, 105)
        ax1.legend(loc="upper left", fontsize='x-small')

        ax1b = ax1.twinx()
        ax1b.fill_between(t, gpu_samples["used_mib"], color='tab:gray', alpha=0.1, label="VRAM Used")
        ax1b.set_ylabel("VRAM (MiB)", color='gray')
        ax1b.set_ylim(bottom=0)

    ax1.grid(True, alpha=0.3)
    ax1.set_title("GPU Metrics", fontsize=10)
    overlay_markers(ax1, y_lim_ref=105)

    # --- Row 2: PCIe Bandwidth ---
    # v1.0.0 wire format: `pcie_bw_bps` is a single column carrying the
    # combined RX+TX rate. The separate RX/TX channels available in the
    # older per-event format are no longer surfaced by the C++ side, so
    # this row plots one combined line.
    if not gpu_samples.empty and "pcie_gbps" in gpu_samples.columns \
            and gpu_samples["pcie_gbps"].max() > 0:
        t = gpu_samples["t_s_abs"]
        ax2.plot(t, gpu_samples["pcie_gbps"],
                 label="PCIe BW (RX+TX)", color='tab:blue')
        ax2.set_ylabel("BW (GB/s)")
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left", fontsize='x-small')
    else:
        # No PCIe data — surface the gap so users don't think the
        # axis is broken. PCIe only appears in the extended-metric
        # batch shape, which requires Blackwell-era or driver-side
        # support for the underlying counters.
        ax2.set_ylabel("BW (GB/s)")
        ax2.text(0.5, 0.5, "PCIe bandwidth not collected",
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=10, color='gray', style='italic')

    ax2.grid(True, alpha=0.3)
    ax2.set_title("PCIe Bandwidth", fontsize=10)
    # Overlay markers (passing None lets helper figure out Y-max from data)
    overlay_markers(ax2)

    # --- Row 3: Host Metrics ---
    if not host_samples.empty:
        t_host = host_samples["t_s_abs"]
        ax3.plot(t_host, host_samples["cpu_pct"], label="CPU %", color='tab:red')
        ax3.set_ylabel("CPU Util %", color='tab:red')
        ax3.set_ylim(-5, 105)
        ax3.tick_params(axis='y', labelcolor='tab:red')
        ax3.legend(loc="upper left", fontsize='x-small')

        ax3b = ax3.twinx()
        ax3b.plot(t_host, host_samples["ram_used_mib"] / 1024, label="RAM (GiB)", color='tab:blue', linestyle="--")
        ax3b.set_ylabel("Sys RAM (GiB)", color='tab:blue')
        ax3b.tick_params(axis='y', labelcolor='tab:blue')
        ax3b.set_ylim(bottom=0)
        ax3b.legend(loc="upper right", fontsize='x-small')

    ax3.set_xlabel("Time (seconds)")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Host Metrics", fontsize=10)
    overlay_markers(ax3, y_lim_ref=105)

    # --- Hover Interaction ---
    def on_hover(event):
        if event.inaxes is None: return
        
        changed = False
        for vl, ann in kernel_markers:
            # Check if mouse is near the vertical line (x-axis distance)
            if vl.axes == event.inaxes:
                # Calculate distance in pixels for better UX
                try:
                    # Convert data x to display x
                    x_display = vl.axes.transData.transform((vl.get_xdata()[0], 0))[0]
                    mouse_x = event.x
                    
                    is_near = abs(x_display - mouse_x) < 5 # 5 pixels tolerance
                    
                    if ann.get_visible() != is_near:
                        ann.set_visible(is_near)
                        changed = True
                except:
                    pass
        
        if changed:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    return fig

# Legacy wrappers
def plot_kernel_timeline(df, title="Kernels"): return plot_combined_timeline(df, title)
def plot_scope_timeline(df, title="Scopes"): return plot_combined_timeline(df, title)
def plot_host_timeline(df, title="Host"): return plot_combined_timeline(df, title)
def plot_memory_timeline(df, gpu_id=0, title="Mem"): return None
def plot_utilization_timeline(df, gpu_id=0, title="Util"): return None