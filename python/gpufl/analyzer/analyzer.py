import re
import pandas as pd
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout


def _fmt_bytes(n) -> str:
    """Format a byte count with an appropriate unit."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    if n == 0:
        return "0 B"
    if n >= 1024 * 1024:
        return f"{n / 1048576:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _shorten_kernel_name(name: str) -> tuple[str, str]:
    """
    Return (short_name, full_name).
    Strips C++ verbosity: return-type prefix, deep namespaces, template args.
    E.g.:
      'void at::native::vectorized_elementwise_kernel<4, CUDAFunctor>'
      → 'native::vectorized_elementwise_kernel<…>'
    """
    s = name.strip()
    # Strip return-type prefix
    s = re.sub(r'^(void|int|float|double|__global__)\s+', '', s)
    # Isolate the bare function name (before first '<' or '(')
    func_part = re.split(r'[<(]', s)[0]           # e.g. 'at::native::vectorized_kernel'
    parts = func_part.split('::')
    short_func = '::'.join(parts[-2:]) if len(parts) > 2 else func_part
    # Re-attach a collapsed template indicator
    if '<' in s:
        short_func += '<…>'
    return short_func, name

class GpuFlightSession:
    def __init__(self, log_dir: str, session_id: str = None, log_prefix: str = "gfl_block", max_stack_depth: int = 5):
        self.log_dir = Path(log_dir)
        self.console = Console()
        self.max_stack_depth = max_stack_depth

        # 1. Load DataFrames
        self.device = self._load_log(f"{log_prefix}.device.0.log")
        self.scopes = self._load_log(f"{log_prefix}.scope.0.log")
        self.system = self._load_log(f"{log_prefix}.system.0.log")

        # 2. Split device log by event type
        if not self.device.empty and 'type' in self.device.columns:
            self.kernels = self.device[self.device['type'] == 'kernel_event'].copy()
            self.memcpy  = self.device[self.device['type'] == 'memcpy_event'].copy()
            self.memset  = self.device[self.device['type'] == 'memset_event'].copy()
        else:
            self.kernels = pd.DataFrame()
            self.memcpy  = pd.DataFrame()
            self.memset  = pd.DataFrame()

        # 3. Filter by Session ID if provided (or pick the latest)
        if session_id:
            self.kernels = self.kernels[self.kernels['session_id'] == session_id]
            self.memcpy  = self.memcpy[self.memcpy['session_id'] == session_id]
            self.memset  = self.memset[self.memset['session_id'] == session_id]

        # 4. Pre-Calculate Metrics (The "Secret Sauce")
        self._enrich_data()

    def _load_log(self, filename):
        """Efficiently loads JSONL into Pandas"""
        path = self.log_dir / filename
        if not path.exists():
            return pd.DataFrame()

        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except: pass
        return pd.DataFrame(data)

    def _enrich_data(self):
        """Calculates derived metrics (Latency, Bandwidth, Duration)"""
        if not self.kernels.empty:
            k = self.kernels
            k['duration_ms'] = (k['end_ns'] - k['start_ns']) / 1e6
            k['cpu_overhead_ms'] = (k['api_exit_ns'] - k['api_start_ns']) / 1e6
            # Queue Latency: gap between CPU dispatch and GPU start (clamped — clock drift)
            k['queue_latency_ms'] = ((k['start_ns'] - k['api_exit_ns']) / 1e6).clip(lower=0)
            self.kernels = k

        # Phase 1b: memcpy throughput
        if not self.memcpy.empty and {'bytes', 'start_ns', 'end_ns'}.issubset(self.memcpy.columns):
            m = self.memcpy
            duration_ns = (m['end_ns'] - m['start_ns']).replace(0, float('nan'))
            m['throughput_gbps'] = m['bytes'] / duration_ns  # bytes/ns == GB/s
            m['duration_ms'] = (m['end_ns'] - m['start_ns']) / 1e6
            self.memcpy = m

    def print_summary(self):
        """Prints an 'Executive Summary' of the session"""
        if self.kernels.empty:
            self.console.print("[bold red]No kernel data found![/bold red]")
            return

        total_duration = self.kernels['end_ns'].max() - self.kernels['start_ns'].min()
        total_duration_ms = total_duration / 1e6
        gpu_busy_time = self.kernels['duration_ms'].sum()

        # Calculate global GPU Utilization % from logs if available, or estimate
        def get_device_stat(devices, key, agg='mean'):
            if not isinstance(devices, list) or len(devices) == 0:
                return 0
            stats = [d.get(key, 0) for d in devices if isinstance(d, dict)]
            if not stats: return 0
            return sum(stats) / len(stats) if agg == 'mean' else max(stats)

        avg_gpu_util = self.system['devices'].apply(lambda x: get_device_stat(x, 'util_gpu')).mean()
        peak_mem = self.system['devices'].apply(lambda x: get_device_stat(x, 'used_mib', 'max')).max()

        # Create Dashboard
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        stats = Table(show_header=False, box=None)
        stats.add_row("Total Duration:", f"[bold cyan]{total_duration_ms/1000:.2f} s[/bold cyan]")
        stats.add_row("Total Kernels:", f"[bold]{len(self.kernels)}[/bold]")
        stats.add_row("GPU Busy Time:", f"[green]{gpu_busy_time/1000:.2f} s[/green]")
        stats.add_row("Avg GPU Util:", f"[yellow]{avg_gpu_util:.1f}%[/yellow]")
        stats.add_row("Peak VRAM:", f"[red]{peak_mem} MiB[/red]")

        self.console.print(Panel(stats, title="[bold]GPUFlight Session Report[/bold]", subtitle=self.kernels.iloc[0]['app']))

    def inspect_hotspots(self, top_n=5, max_stack_depth=None):
        """Identify the most expensive kernels and show their stack traces"""
        if self.kernels.empty:
            self.console.print("[yellow]No kernel data to analyze hotspots.[/yellow]")
            return

        depth = max_stack_depth or self.max_stack_depth

        # Group by Kernel Name and Stack Trace
        # We include stack_trace in groupby to see hotspots per call site
        group_cols = ['name']
        if 'stack_trace' in self.kernels.columns:
            group_cols.append('stack_trace')

        def safe_mode(x):
            return x.mode()[0] if not x.empty else ''

        agg_dict = dict(
            count=('name', 'count'),
            total_time_ms=('duration_ms', 'sum'),
            avg_time_ms=('duration_ms', 'mean'),
            max_time_ms=('duration_ms', 'max'),
            avg_occupancy=('occupancy', 'mean'),
            grid=('grid', 'first'),
            block=('block', 'first'),
            dyn_shared=('dyn_shared_bytes', 'first'),
            static_shared=('static_shared_bytes', 'first'),
            num_regs=('num_regs', 'first'),
            local_bytes=('local_bytes', 'first'),
            const_bytes=('const_bytes', 'first'),
        )
        for col, alias in [
            ('reg_occupancy',  'reg_occ'),
            ('smem_occupancy', 'smem_occ'),
            ('warp_occupancy', 'warp_occ'),
            ('block_occupancy','block_occ'),
            ('limiting_resource', 'limiting'),
        ]:
            if col in self.kernels.columns:
                if col == 'limiting_resource':
                    agg_dict[alias] = (col, safe_mode)
                else:
                    agg_dict[alias] = (col, 'mean')

        summary = self.kernels.groupby(group_cols).agg(**agg_dict).sort_values('total_time_ms', ascending=False).head(top_n)

        table = Table(title=f"🔥 Top {top_n} Kernel Hotspots (Time Consuming)")
        table.add_column("Kernel Name / Stack Trace", style="cyan", no_wrap=False)
        table.add_column("Calls", justify="right")
        table.add_column("Total Time", justify="right", style="green")
        table.add_column("Occupancy", justify="right", style="magenta")
        table.add_column("Grid/Block", justify="center")
        table.add_column("Resources (Reg/SMem/DMem/LMem/CMem)", justify="left")

        for (name, *rest), row in summary.iterrows():
            stack_trace = rest[0] if rest else None

            # Show the raw kernel name from the JSON
            display_content = f"[bold]{name}[/bold]"

            if stack_trace and isinstance(stack_trace, str) and stack_trace.strip():
                frames = stack_trace.split('|')
                # Strip empty and gpufl-internal frames
                frames = [f.strip() for f in frames if f.strip() and not f.strip().startswith('gpufl::')]
                if frames:
                    # Show from outermost caller (rightmost) down to innermost
                    frames_reversed = frames[::-1]
                    limited_frames = frames_reversed[:depth]
                    stack_viz = ""
                    for i, frame in enumerate(limited_frames):
                        indent = "  " * i
                        prefix = "└─ " if i > 0 else "↳ "
                        stack_viz += f"\n{indent}{prefix}[dim]{frame}[/dim]"

                    if len(frames_reversed) > depth:
                        stack_viz += f"\n{'  ' * (depth + 1)}[dim]… ({len(frames_reversed) - depth} more)[/dim]"

                    display_content += stack_viz

            # Per-resource occupancy breakdown (available only when hasDetails=True)
            occ_parts = []
            for key, label in [('reg_occ', 'reg'), ('smem_occ', 'smem'), ('warp_occ', 'warp'), ('block_occ', 'blk')]:
                if key in row.index and pd.notna(row[key]):
                    occ_parts.append(f"{label} {row[key]*100:.1f}%")
            occ_breakdown = " | ".join(occ_parts) if occ_parts else ""

            limiting = row.get('limiting', '') if 'limiting' in row.index else ''
            bottleneck_str = f"\n⚑ Bottleneck: {limiting}" if limiting else ""

            static_b = row['static_shared'] if pd.notna(row.get('static_shared')) else 0
            dyn_b    = row['dyn_shared']    if pd.notna(row.get('dyn_shared'))    else 0
            local_b  = row['local_bytes']   if pd.notna(row.get('local_bytes'))   else 0
            const_b  = row['const_bytes']   if pd.notna(row.get('const_bytes'))   else 0

            resource_str = (
                f"{row['num_regs']} regs"
                + (f" ({occ_breakdown})" if occ_breakdown else "")
                + f"\nSMem {static_b} B · DMem {dyn_b} B"
                + f"\nLMem {local_b} B · CMem {const_b} B"
                + bottleneck_str
            )

            table.add_row(
                display_content,
                str(row['count']),
                f"{row['total_time_ms']:.2f} ms",
                f"{row['avg_occupancy']*100:.1f}%",
                f"[dim]Grid[/dim]  {row['grid']}\n[dim]Block[/dim] {row['block']}",
                resource_str
            )

        self.console.print(table)

    def inspect_stalls(self, top_n: int = 10):
        """Show per-kernel stall distribution from PC-sampling data.

        Requires ``enablePCSampling=true`` at session init.  Joins
        ``profile_sample`` events to kernels via ``corr_id``, then pivots by
        ``reason_name`` to show what fraction of samples each stall category
        accounts for in the hottest kernels.
        """
        if self.scopes.empty or 'type' not in self.scopes.columns:
            self.console.print("[yellow]No scope log data found.[/yellow]")
            return

        samples = self.scopes[self.scopes['type'] == 'profile_sample'].copy()
        if samples.empty:
            self.console.print("[yellow]No profile_sample events found — enable PC sampling at init.[/yellow]")
            return

        required = {'corr_id', 'reason_name', 'sample_count'}
        if not required.issubset(samples.columns):
            self.console.print(f"[yellow]profile_sample records missing columns: {required - set(samples.columns)}[/yellow]")
            return

        samples['sample_count'] = pd.to_numeric(samples['sample_count'], errors='coerce').fillna(0)

        # Aggregate sample counts: (corr_id, reason_name) → total samples
        stall_agg = (
            samples.groupby(['corr_id', 'reason_name'], as_index=False)['sample_count']
            .sum()
        )

        # Total samples per corr_id (used to compute percentages)
        total_per_corr = stall_agg.groupby('corr_id')['sample_count'].sum().rename('total_samples')
        stall_agg = stall_agg.join(total_per_corr, on='corr_id')
        stall_agg['pct'] = (stall_agg['sample_count'] / stall_agg['total_samples'] * 100).round(1)

        # Pivot: rows = corr_id, columns = reason_name, values = pct
        pivot = stall_agg.pivot_table(index='corr_id', columns='reason_name', values='pct', fill_value=0.0)

        # Join kernel names
        if not self.kernels.empty and 'corr_id' in self.kernels.columns:
            kernel_names = self.kernels[['corr_id', 'name']].drop_duplicates('corr_id').set_index('corr_id')
            pivot = pivot.join(kernel_names, how='left')
            pivot['name'] = pivot['name'].fillna('unknown')
        else:
            pivot['name'] = 'unknown'

        # Sort by total sample count (most sampled kernels first)
        pivot = pivot.join(total_per_corr, how='left').sort_values('total_samples', ascending=False).head(top_n)

        stall_cols = [c for c in pivot.columns if c not in ('name', 'total_samples')]

        table = Table(title=f"Stall Distribution — Top {top_n} Kernels (PC Sampling)")
        table.add_column("Kernel", style="cyan", no_wrap=False)
        table.add_column("Samples", justify="right")
        for col in stall_cols:
            table.add_column(col, justify="right")

        for corr_id, row in pivot.iterrows():
            stall_cells = []
            for col in stall_cols:
                val = row[col]
                # Highlight dominant stall reason in yellow
                cell = f"[yellow]{val:.1f}%[/yellow]" if val >= 20.0 else f"{val:.1f}%"
                stall_cells.append(cell)
            table.add_row(
                str(row['name']),
                str(int(row.get('total_samples', 0))),
                *stall_cells,
            )

        self.console.print(table)

    def inspect_scopes(self):
        """Analyze time spent in user-defined Scopes (e.g. 'Training_Epoch')"""
        if self.kernels.empty or 'user_scope' not in self.kernels.columns:
            self.console.print("[yellow]No scope data found or 'user_scope' column missing.[/yellow]")
            return

        # Aggregate metrics by user scope
        scope_stats = self.kernels.groupby('user_scope').agg(
            kernels=('name', 'count'),
            gpu_time_ms=('duration_ms', 'sum'),
            avg_queue_ms=('queue_latency_ms', 'mean'),
            cpu_overhead_ms=('cpu_overhead_ms', 'sum')
        ).sort_index()

        table = Table(title="📂 Scope Analysis (Hierarchical)")
        table.add_column("Scope / Phase", style="bold white")
        table.add_column("GPU Time", style="green", justify="right")
        table.add_column("Queue Latency", style="red", justify="right")
        table.add_column("CPU Overhead", style="yellow", justify="right")

        for scope, row in scope_stats.iterrows():
            # format the scope (e.g. replace | with >)
            formatted_scope = scope.replace("|", " [dim]>[/dim] ")
            table.add_row(
                formatted_scope,
                f"{row['gpu_time_ms']:.2f} ms",
                f"{row['avg_queue_ms']:.3f} ms",
                f"{row['cpu_overhead_ms']:.2f} ms"
            )

        self.console.print(table)