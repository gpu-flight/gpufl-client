import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gpufl.analyzer.analyzer import (
    GpuFlightSession,
    _fmt_bytes,
    _shorten_kernel_name,
    _STALL_NAMES,
)

_COPY_KIND_NAMES = {
    1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH",
    5: "AtoA", 6: "AtoD", 7: "DtoA", 8: "DtoD",
    9: "HtoH", 10: "PtoP",
}

_SEP = "=" * 79
_THIN_SEP = "-" * 79


def _resolve_copy_kind(val):
    if isinstance(val, str):
        return val
    return _COPY_KIND_NAMES.get(int(val), f"Unknown({val})")


def _fmt_duration(ms):
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 1:
        return f"{ms:.2f} ms"
    return f"{ms * 1000:.2f} us"


def _fmt_power(mw):
    if mw >= 1000:
        return f"{mw / 1000:.1f} W"
    return f"{mw:.0f} mW"


class TextReport:
    def __init__(self, session: GpuFlightSession, top_n: int = 10):
        self.session = session
        self.top_n = top_n

    def generate(self) -> str:
        sections = [
            self._section_header(),
            self._section_session_summary(),
            self._section_kernel_summary(),
            self._section_top_kernels(),
            self._section_kernel_details(),
            self._section_memcpy_summary(),
            self._section_system_metrics(),
            self._section_scope_summary(),
            self._section_profile_analysis(),
        ]
        return "\n".join("\n".join(s) for s in sections) + "\n"

    def save(self, path: str) -> None:
        Path(path).write_text(self.generate(), encoding="utf-8")

    def print(self, file=None) -> None:
        print(self.generate(), file=file or sys.stdout)

    # -- Sections ----------------------------------------------------------

    def _section_header(self) -> list[str]:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return [
            _SEP,
            "GPU Flight Session Report".center(79),
            f"Generated: {now}".center(79),
            _SEP,
        ]

    def _section_session_summary(self) -> list[str]:
        s = self.session
        lines = ["", _SEP, "  Session Summary", _SEP]

        app = s.app_name
        if app is None and not s.kernels.empty and "app" in s.kernels.columns:
            app = s.kernels.iloc[0].get("app")
        lines.append(f"  Application:          {app or 'unknown'}")

        # Session ID
        sid = None
        if not s.kernels.empty and "session_id" in s.kernels.columns:
            sid = s.kernels.iloc[0].get("session_id")
        if sid is None and not s.device_metrics.empty and "session_id" in s.device_metrics.columns:
            sid = s.device_metrics.iloc[0].get("session_id")
        if sid:
            lines.append(f"  Session ID:           {sid}")

        # Duration
        if s.session_start_ns is not None and s.session_end_ns is not None:
            dur_ms = (s.session_end_ns - s.session_start_ns) / 1e6
            lines.append(f"  Duration:             {_fmt_duration(dur_ms)}")
        elif not s.kernels.empty:
            dur_ms = (s.kernels["end_ns"].max() - s.kernels["start_ns"].min()) / 1e6
            lines.append(f"  Duration (kernel):    {_fmt_duration(dur_ms)}")

        # GPU device info
        if s.static_devices:
            for dev in s.static_devices:
                name = dev.get("name", "Unknown GPU")
                lines.append(f"  GPU Device:           {name}")
                cc_major = dev.get("compute_capability_major", dev.get("major"))
                cc_minor = dev.get("compute_capability_minor", dev.get("minor"))
                if cc_major is not None and cc_minor is not None:
                    lines.append(f"    Compute:            {cc_major}.{cc_minor}")
                sm = dev.get("multi_processor_count")
                if sm is not None:
                    lines.append(f"    SMs:                {sm}")
                smem = dev.get("shared_mem_per_block")
                if smem is not None:
                    lines.append(f"    Shared Mem/Block:   {_fmt_bytes(smem)}")
                regs = dev.get("regs_per_block")
                if regs is not None:
                    lines.append(f"    Registers/Block:    {regs}")
                l2 = dev.get("l2_cache_size")
                if l2 is not None:
                    lines.append(f"    L2 Cache:           {_fmt_bytes(l2)}")

        return lines

    def _section_kernel_summary(self) -> list[str]:
        lines = ["", _SEP, "  Kernel Execution Summary", _SEP]
        k = self.session.kernels
        if k.empty:
            lines.append("  (No kernel data)")
            return lines

        total = len(k)
        unique = k["name"].nunique() if "name" in k.columns else "?"
        gpu_time_ms = k["duration_ms"].sum() if "duration_ms" in k.columns else 0
        avg_ms = k["duration_ms"].mean() if "duration_ms" in k.columns else 0
        med_ms = k["duration_ms"].median() if "duration_ms" in k.columns else 0
        min_ms = k["duration_ms"].min() if "duration_ms" in k.columns else 0
        max_ms = k["duration_ms"].max() if "duration_ms" in k.columns else 0

        lines.append(f"  Total Kernels:        {total}")
        lines.append(f"  Unique Kernels:       {unique}")
        lines.append(f"  Total GPU Time:       {_fmt_duration(gpu_time_ms)}")

        # GPU busy %
        s = self.session
        if s.session_start_ns is not None and s.session_end_ns is not None:
            total_dur_ms = (s.session_end_ns - s.session_start_ns) / 1e6
            if total_dur_ms > 0:
                busy_pct = gpu_time_ms / total_dur_ms * 100
                lines.append(f"  GPU Busy:             {busy_pct:.1f}%")

        lines.append(f"  Avg Duration:         {_fmt_duration(avg_ms)}")
        lines.append(f"  Median Duration:      {_fmt_duration(med_ms)}")
        lines.append(f"  Min Duration:         {_fmt_duration(min_ms)}")
        lines.append(f"  Max Duration:         {_fmt_duration(max_ms)}")
        return lines

    def _section_top_kernels(self) -> list[str]:
        lines = ["", _SEP, f"  Top {self.top_n} Kernels by Total GPU Time", _SEP]
        k = self.session.kernels
        if k.empty or "name" not in k.columns or "duration_ms" not in k.columns:
            lines.append("  (No kernel data)")
            return lines

        grouped = (
            k.groupby("name")["duration_ms"]
            .agg(["count", "sum", "mean", "max"])
            .sort_values("sum", ascending=False)
            .head(self.top_n)
        )

        hdr = f"  {'#':<4}{'Kernel':<40}{'Calls':>6}{'Total':>12}{'Avg':>12}{'Max':>12}"
        lines.append(hdr)
        lines.append("  " + "-" * 86)

        for i, (name, row) in enumerate(grouped.iterrows(), 1):
            short, _ = _shorten_kernel_name(str(name))
            if len(short) > 38:
                short = short[:35] + "..."
            lines.append(
                f"  {i:<4}{short:<40}{int(row['count']):>6}"
                f"{_fmt_duration(row['sum']):>12}"
                f"{_fmt_duration(row['mean']):>12}"
                f"{_fmt_duration(row['max']):>12}"
            )

        return lines

    def _section_kernel_details(self) -> list[str]:
        lines = ["", _SEP, f"  Kernel Details (Top {self.top_n})", _SEP]
        k = self.session.kernels
        if k.empty or "occupancy" not in k.columns:
            lines.append("  (No kernel detail data)")
            return lines

        # Show details for top kernels by total GPU time
        grouped = (
            k.groupby("name")["duration_ms"]
            .sum()
            .sort_values(ascending=False)
            .head(self.top_n)
        )

        for name in grouped.index:
            subset = k[k["name"] == name]
            short, _ = _shorten_kernel_name(str(name))
            lines.append(f"\n  {short}")
            lines.append(f"  {'=' * len(short)}")

            # Use the first row with detail data as representative
            row = subset.iloc[0]

            grid = row.get("grid", "?")
            block = row.get("block", "?")
            lines.append(f"    Grid:               {grid}")
            lines.append(f"    Block:              {block}")

            occ = row.get("occupancy")
            if pd.notna(occ):
                lines.append(f"    Occupancy:          {float(occ) * 100:.1f}%")
            for key, label in [
                ("reg_occupancy", "Reg Occupancy"),
                ("smem_occupancy", "SMem Occupancy"),
                ("warp_occupancy", "Warp Occupancy"),
                ("block_occupancy", "Block Occupancy"),
            ]:
                val = row.get(key)
                if pd.notna(val):
                    lines.append(f"    {label + ':':<20}{float(val) * 100:.1f}%")

            lim = row.get("limiting_resource")
            if pd.notna(lim):
                lines.append(f"    Limiting Resource:  {lim}")

            regs = row.get("num_regs")
            if pd.notna(regs):
                lines.append(f"    Registers/Thread:   {int(regs)}")

            dyn_shared = row.get("dyn_shared_bytes", row.get("dyn_shared"))
            static_shared = row.get("static_shared_bytes", row.get("static_shared"))
            if pd.notna(dyn_shared) or pd.notna(static_shared):
                d = int(dyn_shared) if pd.notna(dyn_shared) else 0
                s = int(static_shared) if pd.notna(static_shared) else 0
                lines.append(f"    Shared Memory:      {_fmt_bytes(d)} dyn + {_fmt_bytes(s)} static")

        return lines

    def _section_memcpy_summary(self) -> list[str]:
        lines = ["", _SEP, "  Memory Transfer Summary", _SEP]
        m = self.session.memcpy
        if m.empty:
            lines.append("  (No memory transfer data)")
            return lines

        total_xfers = len(m)
        total_bytes = m["bytes"].sum() if "bytes" in m.columns else 0
        lines.append(f"  Total Transfers:      {total_xfers}")
        lines.append(f"  Total Bytes:          {_fmt_bytes(total_bytes)}")
        lines.append("")

        if "copy_kind" in m.columns:
            hdr = f"  {'Direction':<12}{'Count':>8}{'Total Bytes':>16}{'Avg Throughput':>18}"
            lines.append(hdr)
            lines.append("  " + "-" * 54)

            for kind, group in m.groupby("copy_kind"):
                kind_name = _resolve_copy_kind(kind)
                count = len(group)
                total_b = group["bytes"].sum() if "bytes" in group.columns else 0
                avg_tp = ""
                if "throughput_gbps" in group.columns:
                    tp = group["throughput_gbps"].mean()
                    if pd.notna(tp):
                        avg_tp = f"{tp:.2f} GB/s"
                lines.append(
                    f"  {kind_name:<12}{count:>8}{_fmt_bytes(total_b):>16}{avg_tp:>18}"
                )

        return lines

    def _section_system_metrics(self) -> list[str]:
        lines = ["", _SEP, "  System Metrics", _SEP]
        dm = self.session.device_metrics
        hm = self.session.host_metrics

        has_gpu = not dm.empty and "gpu_util" in dm.columns
        has_host = not hm.empty and "cpu_pct" in hm.columns

        if not has_gpu and not has_host:
            lines.append("  (No system metric data)")
            return lines

        if has_gpu:
            lines.append("  GPU Metrics:")
            lines.append(f"    Utilization:        avg {dm['gpu_util'].mean():.1f}%  "
                         f"peak {dm['gpu_util'].max():.0f}%  "
                         f"min {dm['gpu_util'].min():.0f}%")

            if "temp_c" in dm.columns:
                lines.append(f"    Temperature:        avg {dm['temp_c'].mean():.1f} C  "
                             f"peak {dm['temp_c'].max():.0f} C")

            if "power_mw" in dm.columns:
                lines.append(f"    Power:              avg {_fmt_power(dm['power_mw'].mean())}  "
                             f"peak {_fmt_power(dm['power_mw'].max())}")

            if "used_mib" in dm.columns:
                lines.append(f"    VRAM Usage:         peak {int(dm['used_mib'].max())} MiB")

            if "clock_sm" in dm.columns:
                sm = pd.to_numeric(dm["clock_sm"], errors="coerce").dropna()
                if not sm.empty:
                    lines.append(f"    SM Clock:           avg {sm.mean():.0f} MHz  "
                                 f"peak {sm.max():.0f} MHz")

        if has_host:
            lines.append("")
            lines.append("  Host Metrics:")
            lines.append(f"    CPU Utilization:    avg {hm['cpu_pct'].mean():.1f}%  "
                         f"peak {hm['cpu_pct'].max():.1f}%")
            if "ram_used_mib" in hm.columns and "ram_total_mib" in hm.columns:
                peak_ram = int(hm["ram_used_mib"].max())
                total_ram = int(hm["ram_total_mib"].iloc[0])
                lines.append(f"    RAM Usage:          peak {peak_ram} / {total_ram} MiB "
                             f"({peak_ram / total_ram * 100:.1f}%)")

        return lines

    def _section_scope_summary(self) -> list[str]:
        lines = ["", _SEP, "  Scope Summary", _SEP]
        se = self.session.scope_events
        k = self.session.kernels

        has_scope_events = not se.empty and "event_type" in se.columns
        has_kernel_scopes = (
            not k.empty
            and "user_scope" in k.columns
            and k["user_scope"].notna().any()
        )

        if not has_scope_events and not has_kernel_scopes:
            lines.append("  (No scope data)")
            return lines

        # Scope events (begin/end pairs)
        if has_scope_events:
            begins = se[se["event_type"] == 0].copy()
            ends = se[se["event_type"] == 1].copy()

            if not begins.empty and not ends.empty:
                pairs = begins.merge(
                    ends[["scope_instance_id", "ts_ns"]],
                    on="scope_instance_id",
                    suffixes=("_begin", "_end"),
                )
                if not pairs.empty:
                    pairs["duration_ms"] = (pairs["ts_ns_end"] - pairs["ts_ns_begin"]) / 1e6
                    scope_stats = (
                        pairs.groupby("name")["duration_ms"]
                        .agg(["count", "sum", "mean", "max"])
                        .sort_values("sum", ascending=False)
                    )

                    lines.append("  Scope Timing:")
                    hdr = f"  {'Scope':<30}{'Calls':>6}{'Total':>12}{'Avg':>12}{'Max':>12}"
                    lines.append(hdr)
                    lines.append("  " + "-" * 72)
                    for name, row in scope_stats.iterrows():
                        sname = str(name)
                        if len(sname) > 28:
                            sname = sname[:25] + "..."
                        lines.append(
                            f"  {sname:<30}{int(row['count']):>6}"
                            f"{_fmt_duration(row['sum']):>12}"
                            f"{_fmt_duration(row['mean']):>12}"
                            f"{_fmt_duration(row['max']):>12}"
                        )

        # GPU time per scope (from kernel user_scope)
        if has_kernel_scopes:
            lines.append("")
            lines.append("  GPU Time by Scope:")
            scope_col = k["user_scope"].apply(
                lambda x: str(x).split("|")[0] if pd.notna(x) else None
            )
            k_with_scope = k.copy()
            k_with_scope["_scope"] = scope_col
            k_with_scope = k_with_scope[k_with_scope["_scope"].notna()]

            scope_gpu = (
                k_with_scope.groupby("_scope")["duration_ms"]
                .agg(["count", "sum", "mean"])
                .sort_values("sum", ascending=False)
            )

            hdr = f"  {'Scope':<30}{'Kernels':>8}{'GPU Time':>14}{'Avg':>12}"
            lines.append(hdr)
            lines.append("  " + "-" * 64)
            for name, row in scope_gpu.iterrows():
                sname = str(name)
                if len(sname) > 28:
                    sname = sname[:25] + "..."
                lines.append(
                    f"  {sname:<30}{int(row['count']):>8}"
                    f"{_fmt_duration(row['sum']):>14}"
                    f"{_fmt_duration(row['mean']):>12}"
                )

        return lines

    def _section_profile_analysis(self) -> list[str]:
        lines = ["", _SEP, "  Profile / SASS Analysis", _SEP]
        samples = self.session.scopes

        if samples.empty:
            lines.append("  (No profile sample data)")
            return lines

        # Check for stall reason data
        has_stalls = "reason_name" in samples.columns and samples["reason_name"].notna().any()
        has_sass = "metric_name" in samples.columns and samples["metric_name"].notna().any()

        if not has_stalls and not has_sass:
            lines.append("  (No stall or SASS metric data)")
            return lines

        # Stall reason distribution
        if has_stalls:
            stall_data = samples[samples["reason_name"].notna()]
            stall_counts = stall_data.groupby("reason_name")["sample_count"].sum()
            total_stalls = stall_counts.sum()

            if total_stalls > 0:
                lines.append("  Stall Reason Distribution:")
                hdr = f"  {'Reason':<30}{'Samples':>10}{'Pct':>8}"
                lines.append(hdr)
                lines.append("  " + "-" * 48)
                for reason, count in stall_counts.sort_values(ascending=False).items():
                    pct = count / total_stalls * 100
                    lines.append(f"  {str(reason):<30}{int(count):>10}{pct:>7.1f}%")

                # Per-kernel stall breakdown
                if "function_name" in stall_data.columns:
                    lines.append("")
                    lines.append(f"  Top {self.top_n} Kernels by Stall Samples:")
                    kern_stalls = (
                        stall_data.groupby("function_name")["sample_count"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(self.top_n)
                    )
                    for fn, count in kern_stalls.items():
                        short, _ = _shorten_kernel_name(str(fn))
                        if len(short) > 50:
                            short = short[:47] + "..."
                        lines.append(f"    {short:<52}{int(count):>8} samples")

        # SASS metrics summary
        if has_sass:
            lines.append("")
            lines.append("  SASS Metrics Summary:")

            sass = samples[samples["metric_name"].notna()]
            metric_sums = (
                sass.groupby("metric_name")["metric_value"]
                .sum()
                .sort_values(ascending=False)
            )

            hdr = f"  {'Metric':<50}{'Total':>12}"
            lines.append(hdr)
            lines.append("  " + "-" * 62)
            for metric, total in metric_sums.items():
                mname = str(metric)
                if len(mname) > 48:
                    mname = mname[:45] + "..."
                lines.append(f"  {mname:<50}{int(total):>12}")

            # Divergence analysis: thread_inst / warp_inst ratio
            warp_metric = "smsp__sass_inst_executed"
            thread_metric = "smsp__sass_thread_inst_executed"
            if warp_metric in metric_sums.index and thread_metric in metric_sums.index:
                warp_total = metric_sums[warp_metric]
                thread_total = metric_sums[thread_metric]
                if warp_total > 0:
                    # For a warp of 32 threads, thread/warp ratio should be 32 if no divergence
                    ratio = thread_total / warp_total
                    efficiency = ratio / 32.0 * 100
                    lines.append("")
                    lines.append(f"  Thread Divergence Analysis:")
                    lines.append(f"    Warp Instructions:    {int(warp_total)}")
                    lines.append(f"    Thread Instructions:  {int(thread_total)}")
                    lines.append(f"    Avg Threads/Warp:     {ratio:.1f} / 32")
                    lines.append(f"    Warp Efficiency:      {efficiency:.1f}%")

        return lines


def generate_report(
    log_dir: str,
    log_prefix: str = "gfl_block",
    session_id: str = None,
    top_n: int = 10,
    output_path: str = None,
) -> str:
    session = GpuFlightSession(log_dir, session_id=session_id, log_prefix=log_prefix)
    report = TextReport(session, top_n=top_n)
    text = report.generate()
    if output_path:
        report.save(output_path)
    return text
