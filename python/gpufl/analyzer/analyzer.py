import re
import pandas as pd
import json
from pathlib import Path
import gzip
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


# CUPTI CUpti_ActivityPCSamplingStallReason — skip 0 (invalid) and 1 (none)
_STALL_NAMES: dict[int, str] = {
    2:  "Instruction Fetch",
    3:  "Execution Dependency",
    4:  "Memory Dependency",
    5:  "Texture",
    6:  "Sync",
    7:  "Constant Memory",
    8:  "Pipe Busy",
    9:  "Memory Throttle",
    10: "Branch Resolving",
    11: "Wait",
    12: "Barrier",
    13: "Sleeping",
}


class GpuFlightSession:
    def __init__(self, log_dir: str, session_id: str = None, log_prefix: str = "gfl_block", max_stack_depth: int = 5):
        self.log_dir = Path(log_dir)
        self.console = Console()
        self.max_stack_depth = max_stack_depth
        self.app_name = None
        self._requested_session = session_id

        # 1. Load raw DataFrames from JSONL. Channel logs live either flat in
        #    `log_dir` (legacy / demo layout) or nested under a per-run session
        #    subdir written by the current client
        #    (<log_dir>/<session_id>/<channel>.log[.gz]). Resolve which dir
        #    actually holds this run's logs before loading.
        self._read_dir = self._resolve_read_dir(session_id)
        device_df = self._load_log(self._resolve_log_path(log_prefix, "device"))
        scope_df  = self._load_log(self._resolve_log_path(log_prefix, "scope"))
        system_df = self._load_log(self._resolve_log_path(log_prefix, "system"))

        # 1.5 Resolve the target session and scope every raw frame to it. The
        # log files are append-only, so re-running into the same prefix leaves
        # MULTIPLE sessions in one file; aggregating them silently produces
        # nonsense (e.g. a kernel "duration" longer than the whole session).
        # Default to the most-recent session unless the caller named one.
        session_id = self._resolve_session_id(device_df, scope_df, system_df,
                                               requested=session_id)
        if session_id:
            device_df = self._filter_session(device_df, session_id)
            scope_df  = self._filter_session(scope_df, session_id)
            system_df = self._filter_session(system_df, session_id)

        # 2. Extract session boundaries — supports both job_start (new) and init (old)
        self.session_start_ns = None
        self.session_end_ns   = None
        self.static_devices   = []

        for df in [device_df, scope_df, system_df]:
            if df.empty or 'type' not in df.columns:
                continue
            for etype in ['job_start', 'init']:
                rows = df[df['type'] == etype]
                if not rows.empty:
                    if self.session_start_ns is None and 'ts_ns' in rows.columns:
                        self.session_start_ns = pd.to_numeric(rows.iloc[0]['ts_ns'], errors='coerce')
                    if not self.static_devices:
                        sd = []
                        if 'gpu_static_devices' in rows.columns:
                            sd = rows.iloc[0].get('gpu_static_devices')
                        if not isinstance(sd, list) or not sd:
                            sd = []
                            for field in ('cuda_static_devices', 'rocm_static_devices'):
                                if field not in rows.columns:
                                    continue
                                devices = rows.iloc[0].get(field)
                                if isinstance(devices, list) and devices:
                                    sd.extend(devices)
                        if isinstance(sd, list):
                            self.static_devices = sd
                    if self.app_name is None and 'app' in rows.columns:
                        self.app_name = rows.iloc[0].get('app')
                    break
            shut = df[df['type'] == 'shutdown']
            if not shut.empty and 'ts_ns' in shut.columns and self.session_end_ns is None:
                self.session_end_ns = pd.to_numeric(shut.iloc[-1]['ts_ns'], errors='coerce')

        # 2b. Capture capabilities — what was actually collected vs. enabled
        # but empty vs. skipped. Emitted once at shutdown (type
        # 'capture_capabilities'), routed to all channels.
        self.requested_engine = None
        self.selected_engine = None
        self.capture_capabilities = []
        for df in [device_df, scope_df, system_df]:
            if df.empty or 'type' not in df.columns:
                continue
            caps = df[df['type'] == 'capture_capabilities']
            if caps.empty:
                continue
            row = caps.iloc[-1]
            if 'requested_engine' in caps.columns:
                self.requested_engine = row.get('requested_engine')
            if 'selected_engine' in caps.columns:
                self.selected_engine = row.get('selected_engine')
            cap_list = row.get('capabilities') if 'capabilities' in caps.columns else None
            if isinstance(cap_list, list):
                self.capture_capabilities = cap_list
            break

        # 3. Detect format: batch (new) vs per-event (old)
        _BATCH_TYPES = {
            'kernel_event_batch', 'memcpy_event_batch', 'device_metric_batch',
            'scope_event_batch', 'profile_sample_batch', 'host_metric_batch',
        }
        is_batch_format = any(
            not df.empty and 'type' in df.columns and df['type'].isin(_BATCH_TYPES).any()
            for df in [device_df, scope_df, system_df]
        )

        if is_batch_format:
            dict_maps = self._build_dict_maps(device_df, scope_df, system_df)
            (self.kernels, self.memcpy, self.memset,
             self.scopes, self.perf, self.pm_samples,
             self.device_metrics, self.host_metrics,
             self.scope_events) = self._expand_batches(
                device_df, scope_df, system_df, dict_maps
            )
            self.system = pd.DataFrame()
        else:
            # Legacy per-event format
            if not device_df.empty and 'type' in device_df.columns:
                self.kernels = device_df[device_df['type'] == 'kernel_event'].copy()
                self.memcpy  = device_df[device_df['type'] == 'memcpy_event'].copy()
                self.memset  = device_df[device_df['type'] == 'memset_event'].copy()
            else:
                self.kernels = pd.DataFrame()
                self.memcpy  = pd.DataFrame()
                self.memset  = pd.DataFrame()

            if not scope_df.empty and 'type' in scope_df.columns:
                self.scopes = scope_df
                perf_frames = [scope_df[scope_df['type'] == 'perf_metric_event'].copy()]
                if not device_df.empty and 'type' in device_df.columns:
                    perf_frames.append(device_df[device_df['type'] == 'kernel_perf_metric_event'].copy())
                self.perf = pd.concat(
                    [f for f in perf_frames if not f.empty],
                    ignore_index=True,
                    sort=False,
                ) if any(not f.empty for f in perf_frames) else pd.DataFrame()
                self.pm_samples = scope_df[scope_df['type'] == 'pm_sample'].copy() if 'pm_sample' in set(scope_df['type']) else pd.DataFrame()
            else:
                self.scopes = pd.DataFrame()
                self.perf = (
                    device_df[device_df['type'] == 'kernel_perf_metric_event'].copy()
                    if not device_df.empty and 'type' in device_df.columns
                    else pd.DataFrame()
                )
                self.pm_samples = pd.DataFrame()

            self.system         = system_df
            self.device_metrics = pd.DataFrame()
            self.host_metrics   = pd.DataFrame()
            self.scope_events   = pd.DataFrame()

        # 4. Filter by session_id
        if session_id:
            for attr in ['kernels', 'memcpy', 'memset', 'scopes', 'scope_events']:
                df = getattr(self, attr)
                if not df.empty and 'session_id' in df.columns:
                    setattr(self, attr, df[df['session_id'] == session_id])
            if not self.perf.empty and 'session_id' in self.perf.columns:
                self.perf = self.perf[self.perf['session_id'] == session_id]
            if not self.pm_samples.empty and 'session_id' in self.pm_samples.columns:
                self.pm_samples = self.pm_samples[self.pm_samples['session_id'] == session_id]
            if not self.device_metrics.empty and 'session_id' in self.device_metrics.columns:
                self.device_metrics = self.device_metrics[self.device_metrics['session_id'] == session_id]
            if not self.host_metrics.empty and 'session_id' in self.host_metrics.columns:
                self.host_metrics = self.host_metrics[self.host_metrics['session_id'] == session_id]

        # 5. Pre-calculate derived metrics
        self._enrich_data()

    # ── Session selection ───────────────────────────────────────────────────────

    def _resolve_session_id(self, device_df, scope_df, system_df, requested=None):
        """Pick which session to report on.

        If the caller named one, honor it. Otherwise default to the MOST
        RECENT session in the logs (by job_start ts_ns) and warn when more
        than one is present — aggregating multiple runs from an un-cleared
        log file is almost never intended and corrupts the timing totals.
        Returns None only when there are no job_start markers (legacy logs),
        in which case the data is left unfiltered.
        """
        if requested:
            return requested
        seen = []  # (ts_ns, session_id) in file order
        for df in (device_df, scope_df, system_df):
            if df is None or df.empty or 'type' not in df.columns \
                    or 'session_id' not in df.columns:
                continue
            starts = df[df['type'].isin(['job_start', 'init'])]
            has_ts = 'ts_ns' in starts.columns
            for _, r in starts.iterrows():
                sid = r.get('session_id')
                if sid is None or (isinstance(sid, float) and pd.isna(sid)):
                    continue
                ts = pd.to_numeric(r.get('ts_ns'), errors='coerce') if has_ts else None
                seen.append((ts, sid))
        if not seen:
            return None
        uniq = list(dict.fromkeys(sid for _, sid in seen))
        latest = max(seen, key=lambda x: (-1 if x[0] is None or pd.isna(x[0]) else x[0]))[1]
        if len(uniq) > 1:
            self.console.print(
                f"[yellow]Note:[/yellow] {len(uniq)} sessions found in these logs; "
                f"reporting only the most recent ({latest}). Pass session_id=… to "
                f"choose another, or clear the *.log files between runs.")
        return latest

    @staticmethod
    def _filter_session(df, session_id):
        if df is None or df.empty or 'session_id' not in df.columns:
            return df
        return df[df['session_id'] == session_id].copy()

    # ── Log loading ───────────────────────────────────────────────────────────

    def _resolve_read_dir(self, session_id: str = None) -> Path:
        """Directory that actually holds this run's channel logs.

        Two on-disk layouts are supported:
          * flat (legacy / demo):    <log_dir>/[<prefix>.]<channel>.log[.gz]
          * nested (current client): <log_dir>/<session_id>/<channel>.log[.gz]

        If channel logs sit directly in `log_dir` we use it as-is; otherwise we
        descend into a session subdir, preferring an exact `session_id` match
        and falling back to the most-recently-written run.
        """
        if self._dir_has_any_channel(self.log_dir):
            return self.log_dir
        try:
            subs = [p for p in self.log_dir.iterdir()
                    if p.is_dir() and self._dir_has_any_channel(p)]
        except (OSError, FileNotFoundError):
            subs = []
        if not subs:
            return self.log_dir  # nothing found — _load_log() no-ops gracefully
        if session_id:
            exact = [p for p in subs if p.name == session_id]
            if exact:
                return exact[0]
        return max(subs, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _dir_has_any_channel(d: Path) -> bool:
        rx = re.compile(r"^(?:.+\.)?(?:device|scope|system)(?:\.\d+)?\.log(?:\.gz)?$")
        try:
            return any(rx.match(p.name) for p in d.iterdir() if p.is_file())
        except (OSError, FileNotFoundError):
            return False

    def _resolve_log_path(self, log_prefix: str, channel: str) -> "Path | None":
        """Find the best file for `channel` under the resolved read dir.

        Handles current (bare `<channel>.log[.gz]`) and legacy
        (`<prefix>.<channel>.log[.gz]`) names, active and rotated
        (`...<channel>.<N>.log[.gz]`), compressed or not. Active wins over
        rotated; among rotated the lowest index wins; uncompressed breaks ties.
        """
        base = log_prefix[:-4] if log_prefix.endswith(".log") else log_prefix
        ch = re.escape(channel)
        prefix = rf"(?:{re.escape(base)}\.)?" if base else ""
        rx = re.compile(rf"^{prefix}{ch}(?:\.(\d+))?\.log(\.gz)?$")

        d = getattr(self, "_read_dir", self.log_dir)
        try:
            entries = list(d.iterdir())
        except (OSError, FileNotFoundError):
            return None

        best = None  # (rank_tuple, Path)
        for p in entries:
            if not p.is_file():
                continue
            m = rx.match(p.name)
            if not m:
                continue
            idx = int(m.group(1)) if m.group(1) else -1
            compressed = 1 if m.group(2) else 0
            # active (idx -1) before rotated; lowest rotated index next;
            # uncompressed before compressed as a final tiebreak.
            key = (0 if idx < 0 else 1, idx if idx >= 0 else 0, compressed)
            if best is None or key < best[0]:
                best = (key, p)
        return best[1] if best else None

    def _load_log(self, path: Path | None):
        """Efficiently loads JSONL into Pandas"""
        if path is None or not path.exists():
            return pd.DataFrame()

        data = []
        open_fn = gzip.open if str(path).endswith(".gz") else open
        with open_fn(path, 'rt') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        pass
        return pd.DataFrame(data)

    # ── Batch format support ──────────────────────────────────────────────────

    def _build_dict_maps(self, *dfs) -> dict:
        """Merge all dictionary_update events into {dict_type: {id: name}} maps."""
        maps: dict[str, dict[int, str]] = {
            'kernel': {}, 'scope_name': {}, 'function': {}, 'metric': {}
        }
        for df in dfs:
            if df.empty or 'type' not in df.columns:
                continue
            for _, row in df[df['type'] == 'dictionary_update'].iterrows():
                for col_key, map_key in [
                    ('kernel_dict',     'kernel'),
                    ('scope_name_dict', 'scope_name'),
                    ('function_dict',   'function'),
                    ('metric_dict',     'metric'),
                ]:
                    d = row.get(col_key)
                    if isinstance(d, dict):
                        maps[map_key].update({int(k): v for k, v in d.items()})
        return maps

    def _expand_batches(self, device_df, scope_df, system_df, dict_maps) -> tuple:
        """Expand all batch message types into flat DataFrames.

        Returns: (kernels, memcpy, memset, scopes, perf, device_metrics, host_metrics)
        """
        # ── kernel_event_batch ────────────────────────────────────────────────
        kernel_rows = []
        if not device_df.empty and 'type' in device_df.columns:
            for _, batch in device_df[device_df['type'] == 'kernel_event_batch'].iterrows():
                cols = batch.get('columns', [])
                base = int(batch.get('base_time_ns', 0))
                ci   = {c: i for i, c in enumerate(cols)}
                for row in (batch.get('rows') or []):
                    start = base + row[ci['dt_ns']]
                    kernel_rows.append({
                        'session_id':       batch.get('session_id'),
                        'start_ns':         start,
                        'end_ns':           start + row[ci['duration_ns']],
                        'duration_ns':      row[ci['duration_ns']],
                        'name':             dict_maps['kernel'].get(
                                                row[ci['kernel_id']],
                                                f"kernel_{row[ci['kernel_id']]}"
                                            ),
                        'stream_id':        row[ci['stream_id']],
                        'corr_id':          row[ci['corr_id']],
                        'dyn_shared_bytes': row[ci['dyn_shared']],
                        'num_regs':         row[ci['num_regs']],
                        'has_details':      bool(row[ci['has_details']]),
                    })
        kernels_df = pd.DataFrame(kernel_rows)

        # ── kernel_detail — merge into kernels via corr_id ───────────────────
        if not kernels_df.empty and not device_df.empty and 'type' in device_df.columns:
            detail_rows = []
            for _, det in device_df[device_df['type'] == 'kernel_detail'].iterrows():
                detail_rows.append({
                    'corr_id':                    det.get('corr_id'),
                    'grid':                       det.get('grid'),
                    'block':                      det.get('block'),
                    'static_shared_bytes':         det.get('static_shared'),
                    'local_bytes':                det.get('local_bytes'),
                    'const_bytes':                det.get('const_bytes'),
                    'occupancy':                  det.get('occupancy'),
                    'reg_occupancy':              det.get('reg_occupancy'),
                    'smem_occupancy':             det.get('smem_occupancy'),
                    'warp_occupancy':             det.get('warp_occupancy'),
                    'block_occupancy':            det.get('block_occupancy'),
                    'limiting_resource':           det.get('limiting_resource'),
                    'local_mem_total_bytes':       det.get('local_mem_total_bytes'),
                    'local_mem_per_thread_bytes':  det.get('local_mem_per_thread_bytes'),
                    'user_scope':                 det.get('user_scope'),
                    'stack_trace':                det.get('stack_trace'),
                })
            if detail_rows:
                detail_df = pd.DataFrame(detail_rows)
                detail_df['corr_id']  = pd.to_numeric(detail_df['corr_id'],  errors='coerce')
                kernels_df['corr_id'] = pd.to_numeric(kernels_df['corr_id'], errors='coerce')
                kernels_df = kernels_df.merge(detail_df, on='corr_id', how='left')

        # ── memcpy_event_batch ────────────────────────────────────────────────
        memcpy_rows = []
        if not device_df.empty and 'type' in device_df.columns:
            for _, batch in device_df[device_df['type'] == 'memcpy_event_batch'].iterrows():
                cols = batch.get('columns', [])
                base = int(batch.get('base_time_ns', 0))
                ci   = {c: i for i, c in enumerate(cols)}
                for row in (batch.get('rows') or []):
                    start = base + row[ci['dt_ns']]
                    memcpy_rows.append({
                        'session_id':  batch.get('session_id'),
                        'start_ns':    start,
                        'end_ns':      start + row[ci['duration_ns']],
                        'duration_ns': row[ci['duration_ns']],
                        'stream_id':   row[ci['stream_id']],
                        'bytes':       row[ci['bytes']],
                        'copy_kind':   row[ci['copy_kind']],
                        'corr_id':     row[ci['corr_id']],
                    })
        memcpy_df = pd.DataFrame(memcpy_rows)

        # ── profile_sample_batch (scope log) ──────────────────────────────────
        sample_rows = []
        if not scope_df.empty and 'type' in scope_df.columns:
            for _, batch in scope_df[scope_df['type'] == 'profile_sample_batch'].iterrows():
                cols = batch.get('columns', [])
                base = int(batch.get('base_time_ns', 0))
                ci   = {c: i for i, c in enumerate(cols)}
                for row in (batch.get('rows') or []):
                    sk_int      = row[ci['sample_kind']]
                    sample_kind = {0: 'pc_sampling', 1: 'sass_metric',
                                   2: 'isa_source_map'}.get(sk_int, 'sass_metric')
                    metric_id   = row[ci['metric_id']]
                    metric_name = dict_maps['metric'].get(metric_id) if metric_id else None
                    fn_id       = row[ci['function_id']]
                    fn_name     = dict_maps['function'].get(fn_id) if fn_id else None
                    sn_id       = row[ci['scope_name_id']]
                    scope_name  = dict_maps['scope_name'].get(sn_id) if sn_id else None
                    stall       = row[ci['stall_reason']]
                    mv          = row[ci['metric_value']]
                    sample_rows.append({
                        'type':          'profile_sample',
                        'session_id':    batch.get('session_id'),
                        'ts_ns':         base + row[ci['dt_ns']],
                        'corr_id':       row[ci['corr_id']],
                        'device_id':     row[ci['device_id']],
                        'function_name': fn_name,
                        'pc_offset':     row[ci['pc_offset']],
                        'metric_name':   metric_name,
                        'metric_value':  mv,
                        'stall_reason':  stall if stall > 0 else None,
                        'sample_kind':   sample_kind,
                        'scope_name':    scope_name,
                        # Compatibility aliases used by inspect_stalls / inspect_profile_samples
                        'reason_name':   _STALL_NAMES.get(stall, f"Stall_{stall}") if stall > 1 else None,
                        'sample_count':  mv if sample_kind == 'pc_sampling' else 0,
                    })
        scopes_df = pd.DataFrame(sample_rows)

        # ── pm_sample_batch (scope log) ───────────────────────────────────────
        pm_rows = []
        if not scope_df.empty and 'type' in scope_df.columns:
            for _, batch in scope_df[scope_df['type'] == 'pm_sample_batch'].iterrows():
                cols = batch.get('columns', [])
                base = int(batch.get('base_time_ns', 0))
                ci = {c: i for i, c in enumerate(cols)}
                for row in (batch.get('rows') or []):
                    metric_id = row[ci['metric_id']]
                    scope_id = row[ci['scope_name_id']]
                    pm_rows.append({
                        'type': 'pm_sample',
                        'session_id': batch.get('session_id'),
                        'ts_ns': base + row[ci['dt_ns']],
                        'device_id': row[ci['device_id']],
                        'metric_id': metric_id,
                        'metric_name': dict_maps['metric'].get(metric_id) if metric_id else None,
                        'value': row[ci['value']],
                        'scope_name': dict_maps['scope_name'].get(scope_id) if scope_id else None,
                    })
        pm_df = pd.DataFrame(pm_rows)

        # ── perf_metric_event / kernel_perf_metric_event (per-event counters) ─
        perf_df = pd.DataFrame()
        if not scope_df.empty and 'type' in scope_df.columns:
            perf_rows = scope_df[scope_df['type'] == 'perf_metric_event']
            if not perf_rows.empty:
                perf_df = perf_rows.copy()
                perf_df['kind'] = 'scope'
                if 'name' not in perf_df.columns and 'user_scope' in perf_df.columns:
                    perf_df['name'] = perf_df['user_scope']
        if not device_df.empty and 'type' in device_df.columns:
            kernel_perf_rows = device_df[device_df['type'] == 'kernel_perf_metric_event']
            if not kernel_perf_rows.empty:
                kernel_perf = kernel_perf_rows.copy()
                kernel_perf['kind'] = 'kernel'
                if 'name' not in kernel_perf.columns:
                    kernel_perf['name'] = kernel_perf.get('kernel_name')
                if 'name' in kernel_perf.columns and 'kernel_name' in kernel_perf.columns:
                    kernel_perf['name'] = kernel_perf['name'].fillna(kernel_perf['kernel_name'])
                if 'name' in kernel_perf.columns and 'range_name' in kernel_perf.columns:
                    kernel_perf['name'] = kernel_perf['name'].fillna(kernel_perf['range_name'])
                perf_df = pd.concat([perf_df, kernel_perf], ignore_index=True, sort=False)

        # ── device_metric_batch (system log) ──────────────────────────────────
        dm_rows = []
        if not system_df.empty and 'type' in system_df.columns:
            for _, batch in system_df[system_df['type'] == 'device_metric_batch'].iterrows():
                cols = batch.get('columns', [])
                base = int(batch.get('base_time_ns', 0))
                ci   = {c: i for i, c in enumerate(cols)}
                def _val(row, key, default=0):
                    idx = ci.get(key)
                    return row[idx] if idx is not None and idx < len(row) else default
                for row in (batch.get('rows') or []):
                    dm_rows.append({
                        'session_id': batch.get('session_id'),
                        'ts_ns':      base + _val(row, 'dt_ns', 0),
                        'device_id':  _val(row, 'device_id', 0),
                        'gpu_util':   _val(row, 'gpu_util', 0),
                        'mem_util':   _val(row, 'mem_util', 0),
                        'temp_c':     _val(row, 'temp_c', 0),
                        'power_mw':   _val(row, 'power_mw', 0),
                        'used_mib':   _val(row, 'used_mib', 0),
                        'clock_sm':   _val(row, 'clock_sm', None),
                    })
        device_metrics_df = pd.DataFrame(dm_rows)

        # ── host_metric_batch (system log) ────────────────────────────────────
        hm_rows = []
        if not system_df.empty and 'type' in system_df.columns:
            for _, batch in system_df[system_df['type'] == 'host_metric_batch'].iterrows():
                cols = batch.get('columns', [])
                base = int(batch.get('base_time_ns', 0))
                ci   = {c: i for i, c in enumerate(cols)}
                for row in (batch.get('rows') or []):
                    hm_rows.append({
                        'session_id':    batch.get('session_id'),
                        'ts_ns':         base + row[ci['dt_ns']],
                        'cpu_pct':       row[ci['cpu_pct_x100']] / 100.0,
                        'ram_used_mib':  row[ci['ram_used_mib']],
                        'ram_total_mib': row[ci['ram_total_mib']],
                    })
        host_metrics_df = pd.DataFrame(hm_rows)

        # ── scope_event_batch (scope log) — begin/end pairs ──────────────────
        scope_event_rows = []
        if not scope_df.empty and 'type' in scope_df.columns:
            for _, batch in scope_df[scope_df['type'] == 'scope_event_batch'].iterrows():
                cols = batch.get('columns', [])
                base = int(batch.get('base_time_ns', 0))
                ci   = {c: i for i, c in enumerate(cols)}
                for row in (batch.get('rows') or []):
                    sn_id = row[ci['name_id']]
                    scope_event_rows.append({
                        'session_id':        batch.get('session_id'),
                        'ts_ns':             base + row[ci['dt_ns']],
                        'scope_instance_id': row[ci['scope_instance_id']],
                        'name':              dict_maps['scope_name'].get(sn_id, f"scope_{sn_id}"),
                        'event_type':        row[ci['event_type']],   # 0=begin, 1=end
                        'depth':             row[ci['depth']],
                    })
        scope_events_df = pd.DataFrame(scope_event_rows)

        return (kernels_df, memcpy_df, pd.DataFrame(), scopes_df, perf_df, pm_df,
                device_metrics_df, host_metrics_df, scope_events_df)

    # ── Derived metrics ───────────────────────────────────────────────────────

    def _enrich_data(self):
        """Calculates derived metrics (Latency, Bandwidth, Duration)"""
        if not self.kernels.empty:
            k = self.kernels
            k['duration_ms'] = (k['end_ns'] - k['start_ns']) / 1e6
            # api_start/exit only present in old per-event format
            if {'api_start_ns', 'api_exit_ns'}.issubset(k.columns):
                k['cpu_overhead_ms'] = (k['api_exit_ns'] - k['api_start_ns']) / 1e6
                k['queue_latency_ms'] = ((k['start_ns'] - k['api_exit_ns']) / 1e6).clip(lower=0)
            self.kernels = k

        # Phase 1b: memcpy throughput
        if not self.memcpy.empty and {'bytes', 'start_ns', 'end_ns'}.issubset(self.memcpy.columns):
            m = self.memcpy
            duration_ns = (m['end_ns'] - m['start_ns']).replace(0, float('nan'))
            m['throughput_gbps'] = m['bytes'] / duration_ns  # bytes/ns == GB/s
            m['duration_ms'] = (m['end_ns'] - m['start_ns']) / 1e6
            self.memcpy = m

        if not self.perf.empty:
            if 'kind' not in self.perf.columns:
                if 'type' in self.perf.columns:
                    self.perf['kind'] = self.perf['type'].map({
                        'kernel_perf_metric_event': 'kernel',
                        'perf_metric_event': 'scope',
                    }).fillna('scope')
                else:
                    self.perf['kind'] = 'scope'
            if 'name' not in self.perf.columns:
                self.perf['name'] = None
            if 'kernel_name' in self.perf.columns:
                self.perf['name'] = self.perf['name'].fillna(self.perf['kernel_name'])
            if 'range_name' in self.perf.columns:
                self.perf['name'] = self.perf['name'].fillna(self.perf['range_name'])
            if 'user_scope' in self.perf.columns:
                self.perf['name'] = self.perf['name'].fillna(self.perf['user_scope'])
            for col in [
                'start_ns', 'end_ns',
                'sm_throughput_pct', 'l1_hit_rate_pct', 'l2_hit_rate_pct',
                'dram_read_bytes', 'dram_write_bytes', 'tensor_active_pct'
            ]:
                if col in self.perf.columns:
                    self.perf[col] = pd.to_numeric(self.perf[col], errors='coerce')
            if {'start_ns', 'end_ns'}.issubset(self.perf.columns):
                self.perf['duration_ms'] = (self.perf['end_ns'] - self.perf['start_ns']) / 1e6

    def _resolve_sample_kernel_names(self, samples: pd.DataFrame) -> tuple[dict, int]:
        """Resolve sample corr_id -> kernel name.

        Matching order:
        1) Exact corr_id match
        2) Fallback by function/device + time proximity (kernel interval or nearest midpoint)
        """
        if self.kernels.empty or 'corr_id' not in self.kernels.columns:
            return {}, 0

        required_kernel_cols = {'name', 'start_ns', 'end_ns'}
        if not required_kernel_cols.issubset(self.kernels.columns):
            return {}, 0

        kernels = self.kernels.copy()
        kernels = kernels.dropna(subset=['name', 'start_ns', 'end_ns'])
        if kernels.empty:
            return {}, 0

        kernels['corr_id'] = pd.to_numeric(kernels['corr_id'], errors='coerce')
        kernels['start_ns'] = pd.to_numeric(kernels['start_ns'], errors='coerce')
        kernels['end_ns'] = pd.to_numeric(kernels['end_ns'], errors='coerce')
        kernels['mid_ns'] = (kernels['start_ns'] + kernels['end_ns']) / 2.0
        if 'device_id' in kernels.columns:
            kernels['device_id_num'] = pd.to_numeric(kernels['device_id'], errors='coerce')
        else:
            kernels['device_id_num'] = pd.NA

        samples = samples.copy()
        if 'corr_id' in samples.columns:
            samples['corr_id'] = pd.to_numeric(samples['corr_id'], errors='coerce')
        if 'ts_ns' in samples.columns:
            samples['ts_ns'] = pd.to_numeric(samples['ts_ns'], errors='coerce')
        if 'device_id' in samples.columns:
            samples['device_id_num'] = pd.to_numeric(samples['device_id'], errors='coerce')
        else:
            samples['device_id_num'] = pd.NA

        corr_to_name = {}
        fallback_used = 0

        exact = (
            kernels.dropna(subset=['corr_id'])
            .drop_duplicates('corr_id')
            .set_index('corr_id')['name']
            .to_dict()
        )

        sample_keys = (
            samples.dropna(subset=['corr_id'])
            .groupby('corr_id', as_index=False)
            .agg(
                ts_ns=('ts_ns', 'median'),
                function_name=('function_name', 'first') if 'function_name' in samples.columns else ('corr_id', 'first'),
                device_id_num=('device_id_num', 'first'),
            )
        )

        for _, s in sample_keys.iterrows():
            corr_id = s['corr_id']
            if corr_id in exact:
                corr_to_name[corr_id] = exact[corr_id]
                continue

            candidates = kernels
            fn = s.get('function_name', None)
            if pd.notna(fn) and str(fn):
                by_name = candidates[candidates['name'] == str(fn)]
                if not by_name.empty:
                    candidates = by_name

            dev = s.get('device_id_num', pd.NA)
            if pd.notna(dev):
                by_dev = candidates[candidates['device_id_num'] == dev]
                if not by_dev.empty:
                    candidates = by_dev

            if candidates.empty:
                continue

            ts = s.get('ts_ns', float('nan'))
            if pd.notna(ts):
                containing = candidates[(candidates['start_ns'] <= ts) & (ts <= candidates['end_ns'])]
                choose_from = containing if not containing.empty else candidates
                deltas = (choose_from['mid_ns'] - ts).abs()
                idx = deltas.idxmin()
            else:
                idx = candidates['end_ns'].idxmax()

            name = candidates.loc[idx, 'name']
            if pd.notna(name):
                corr_to_name[corr_id] = str(name)
                fallback_used += 1

        return corr_to_name, fallback_used

    # ── Public analysis methods ───────────────────────────────────────────────

    def print_summary(self):
        """Prints an 'Executive Summary' of the session"""
        # Determine session duration
        if self.session_start_ns is not None and self.session_end_ns is not None:
            total_duration_ms = (self.session_end_ns - self.session_start_ns) / 1e6
        elif not self.kernels.empty:
            total_duration_ms = (self.kernels['end_ns'].max() - self.kernels['start_ns'].min()) / 1e6
        else:
            total_duration_ms = 0.0

        gpu_busy_time = self.kernels['duration_ms'].sum() if not self.kernels.empty else 0.0

        has_device_stats = False
        avg_gpu_util = 0.0
        peak_mem = 0
        avg_sm_clock = None

        # New batch format: device_metrics DataFrame
        if not self.device_metrics.empty and 'gpu_util' in self.device_metrics.columns:
            has_device_stats = True
            avg_gpu_util = self.device_metrics['gpu_util'].mean()
            peak_mem = int(self.device_metrics['used_mib'].max()) if 'used_mib' in self.device_metrics.columns else 0
            if 'clock_sm' in self.device_metrics.columns:
                sm_series = pd.to_numeric(self.device_metrics['clock_sm'], errors='coerce').dropna()
                if not sm_series.empty:
                    avg_sm_clock = sm_series.mean()

        # Legacy format: system DataFrame with nested devices list
        elif not self.system.empty and 'devices' in self.system.columns:
            def get_device_stat(devices, key, agg='mean'):
                if not isinstance(devices, list) or len(devices) == 0:
                    return 0
                stats = [d.get(key, 0) for d in devices if isinstance(d, dict)]
                if not stats: return 0
                return sum(stats) / len(stats) if agg == 'mean' else max(stats)

            non_empty = self.system[self.system['devices'].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            )]
            if not non_empty.empty:
                has_device_stats = True
                avg_gpu_util = non_empty['devices'].apply(
                    lambda x: get_device_stat(x, 'util_gpu_pct')).mean()
                peak_mem = non_empty['devices'].apply(
                    lambda x: get_device_stat(x, 'used_mib', 'max')).max()

        # Create Dashboard
        stats = Table(show_header=False, box=None)
        stats.add_row("Total Duration:", f"[bold cyan]{total_duration_ms/1000:.2f} s[/bold cyan]")
        if not self.kernels.empty:
            stats.add_row("Total Kernels:", f"[bold]{len(self.kernels)}[/bold]")
            stats.add_row("GPU Busy Time:", f"[green]{gpu_busy_time/1000:.2f} s[/green]")
        else:
            stats.add_row("Total Kernels:", "[dim]n/a (no GPU profiling)[/dim]")
        if not self.scope_events.empty:
            n_scopes = len(self.scope_events[self.scope_events['event_type'] == 0]) if 'event_type' in self.scope_events.columns else 0
            if n_scopes:
                stats.add_row("Scope Events:", f"[bold]{n_scopes}[/bold]")

        if has_device_stats:
            stats.add_row("Avg GPU Util:", f"[yellow]{avg_gpu_util:.1f}%[/yellow]")
            stats.add_row("Peak VRAM:", f"[red]{peak_mem} MiB[/red]")
            if avg_sm_clock is not None:
                stats.add_row("Avg SM Clock:", f"[yellow]{avg_sm_clock:.0f} MHz[/yellow]")
        else:
            if self.static_devices:
                for dev in self.static_devices:
                    name = dev.get('name', 'Unknown GPU')
                    sm_count = dev.get('multi_processor_count', '?')
                    stats.add_row("Device:", f"[yellow]{name}[/yellow]")
                    stats.add_row("SMs:", f"[yellow]{sm_count}[/yellow]")
            else:
                stats.add_row("Avg GPU Util:", f"[dim]n/a (NVML unavailable)[/dim]")
                stats.add_row("Peak VRAM:", f"[dim]n/a (NVML unavailable)[/dim]")

        # Use app_name from job_start/init, fall back to kernel column if present
        subtitle = self.app_name
        if subtitle is None and 'app' in self.kernels.columns and not self.kernels.empty:
            subtitle = self.kernels.iloc[0]['app']
        self.console.print(Panel(stats, title="[bold]GPUFlight Session Report[/bold]",
                                 subtitle=subtitle or ''))

    def inspect_hotspots(self, top_n=5, max_stack_depth=None):
        """Identify the most expensive kernels and show their stack traces"""
        if self.kernels.empty:
            self.console.print("[yellow]No kernel data to analyze hotspots.[/yellow]")
            return

        depth = max_stack_depth or self.max_stack_depth

        # Group by Kernel Name and Stack Trace
        group_cols = ['name']
        if 'stack_trace' in self.kernels.columns:
            group_cols.append('stack_trace')

        def safe_mode(x):
            return x.mode()[0] if not x.empty else ''

        # Build agg_dict — only include columns that are actually present
        k = self.kernels
        agg_dict = dict(
            count=('name', 'count'),
            total_time_ms=('duration_ms', 'sum'),
            avg_time_ms=('duration_ms', 'mean'),
            max_time_ms=('duration_ms', 'max'),
        )
        for col, alias, fn in [
            ('occupancy',        'avg_occupancy', 'mean'),
            ('grid',             'grid',          'first'),
            ('block',            'block',         'first'),
            ('dyn_shared_bytes', 'dyn_shared',    'first'),
            ('static_shared_bytes', 'static_shared', 'first'),
            ('num_regs',         'num_regs',      'first'),
            ('local_bytes',      'local_bytes',   'first'),
            ('const_bytes',      'const_bytes',   'first'),
        ]:
            if col in k.columns:
                agg_dict[alias] = (col, fn)

        for col, alias in [
            ('local_mem_per_thread_bytes', 'local_mem_per_thread'),
            ('local_mem_total_bytes',      'local_mem_total'),
        ]:
            if col in k.columns:
                agg_dict[alias] = (col, 'first')

        for col, alias in [
            ('reg_occupancy',    'reg_occ'),
            ('smem_occupancy',   'smem_occ'),
            ('warp_occupancy',   'warp_occ'),
            ('block_occupancy',  'block_occ'),
            ('limiting_resource','limiting'),
        ]:
            if col in k.columns:
                if col == 'limiting_resource':
                    agg_dict[alias] = (col, safe_mode)
                else:
                    agg_dict[alias] = (col, 'mean')

        summary = k.groupby(group_cols).agg(**agg_dict).sort_values('total_time_ms', ascending=False).head(top_n)

        table = Table(title=f"🔥 Top {top_n} Kernel Hotspots (Time Consuming)")
        table.add_column("Kernel Name / Stack Trace", style="cyan", no_wrap=False)
        table.add_column("Calls", justify="right")
        table.add_column("Total Time", justify="right", style="green")
        table.add_column("Occupancy", justify="right", style="magenta")
        table.add_column("Grid/Block", justify="center")
        table.add_column("Resources (Reg/SMem/LMem/CMem/Spill)", justify="left")

        for (name, *rest), row in summary.iterrows():
            stack_trace = rest[0] if rest else None

            display_content = f"[bold]{name}[/bold]"

            if stack_trace and isinstance(stack_trace, str) and stack_trace.strip():
                frames = stack_trace.split('|')
                frames = [f.strip() for f in frames if f.strip() and not f.strip().startswith('gpufl::')]
                if frames:
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

            occ_parts = []
            for key, label in [('reg_occ', 'reg'), ('smem_occ', 'smem'), ('warp_occ', 'warp'), ('block_occ', 'blk')]:
                if key in row.index and pd.notna(row[key]):
                    occ_parts.append(f"{label} {row[key]*100:.1f}%")
            occ_breakdown = " | ".join(occ_parts) if occ_parts else ""

            limiting = row.get('limiting', '') if 'limiting' in row.index else ''
            bottleneck_str = f"\n⚑ Bottleneck: {limiting}" if limiting else ""

            static_b       = row['static_shared']   if 'static_shared'   in row.index and pd.notna(row.get('static_shared'))   else 0
            dyn_b          = row['dyn_shared']       if 'dyn_shared'       in row.index and pd.notna(row.get('dyn_shared'))       else 0
            local_b        = row['local_bytes']      if 'local_bytes'      in row.index and pd.notna(row.get('local_bytes'))      else 0
            const_b        = row['const_bytes']      if 'const_bytes'      in row.index and pd.notna(row.get('const_bytes'))      else 0
            spill_per_thd  = row.get('local_mem_per_thread', 0) or 0
            spill_total_kb = (row.get('local_mem_total', 0) or 0) / 1024

            if spill_per_thd > 0:
                spill_str = f"\n[red]Spill {spill_per_thd} B/thread · {spill_total_kb:.1f} KB total[/red]"
            else:
                spill_str = ""

            num_regs = int(row['num_regs']) if 'num_regs' in row.index and pd.notna(row.get('num_regs')) else '?'
            occ_val  = f"{row['avg_occupancy']*100:.1f}%" if 'avg_occupancy' in row.index and pd.notna(row.get('avg_occupancy')) else "n/a"
            grid_val  = row['grid']  if 'grid'  in row.index and pd.notna(row.get('grid'))  else "n/a"
            block_val = row['block'] if 'block' in row.index and pd.notna(row.get('block')) else "n/a"

            # Shared memory: `static` is compile-time __shared__ arrays;
            # `dyn` is the third launch arg <<<grid,block,dyn_shared>>>.
            # Both live in the same physical SM shared-memory space and
            # together drive smem_occupancy — so we display the SUM as
            # `SMem` (matching what the occupancy % is computed against)
            # with the static/dyn breakdown in parentheses for users
            # tuning either piece. Previously this row showed two
            # separate `SMem`/`DMem` values which read as "static
            # shared" vs "device memory" (wrong: it's dyn-shared, not
            # device global memory) and didn't visually reconcile with
            # the smem occupancy % above.
            smem_total = (static_b or 0) + (dyn_b or 0)
            resource_str = (
                f"{num_regs} regs"
                + (f" ({occ_breakdown})" if occ_breakdown else "")
                + f"\nSMem {smem_total} B ({static_b} static + {dyn_b} dyn)"
                + f"\nLMem {local_b} B · CMem {const_b} B"
                + spill_str
                + bottleneck_str
            )

            table.add_row(
                display_content,
                str(row['count']),
                f"{row['total_time_ms']:.2f} ms",
                occ_val,
                f"[dim]Grid[/dim]  {grid_val}\n[dim]Block[/dim] {block_val}",
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
            self.console.print("[yellow]No PC sampling data found — enable PC sampling at session init.[/yellow]")
            return

        samples = self.scopes[self.scopes['type'] == 'profile_sample'].copy()
        if 'sample_kind' in samples.columns:
            samples = samples[samples['sample_kind'] == 'pc_sampling'].copy()
        elif 'metric_name' in samples.columns:
            samples = samples[samples['metric_name'].isna()].copy()
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

        # Join kernel names (corr_id first, then fallback by function/time)
        corr_to_name, fallback_used = self._resolve_sample_kernel_names(samples)
        if corr_to_name:
            names_df = pd.DataFrame(
                [(k, v) for k, v in corr_to_name.items()],
                columns=['corr_id', 'name']
            ).set_index('corr_id')
            pivot = pivot.join(names_df, how='left')
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
                cell = f"[yellow]{val:.1f}%[/yellow]" if val >= 20.0 else f"{val:.1f}%"
                stall_cells.append(cell)
            table.add_row(
                str(row['name']),
                str(int(row.get('total_samples', 0))),
                *stall_cells,
            )

        self.console.print(table)
        if fallback_used > 0:
            self.console.print(
                f"[yellow]Kernel name fallback used for {fallback_used} corr_id(s) "
                f"(function/time-based match).[/yellow]"
            )

    def inspect_profile_samples(self, top_n: int = 10):
        """Summarize profile_sample records from scope logs.

        Shows top stall reasons by total sample count and, when kernel
        correlation is available, top kernels by sampled stall pressure.
        """
        if self.scopes.empty or 'type' not in self.scopes.columns:
            self._print_profile_sample_hint()
            return

        all_samples = self.scopes[self.scopes['type'] == 'profile_sample'].copy()
        if all_samples.empty:
            self._print_profile_sample_hint()
            return

        if 'sample_kind' in all_samples.columns:
            pc_samples   = all_samples[all_samples['sample_kind'] == 'pc_sampling'].copy()
            sass_samples = all_samples[all_samples['sample_kind'] == 'sass_metric'].copy()
        else:
            if 'metric_name' in all_samples.columns:
                sass_samples = all_samples[all_samples['metric_name'].notna()].copy()
                pc_samples   = all_samples[all_samples['metric_name'].isna()].copy()
            else:
                pc_samples   = all_samples.copy()
                sass_samples = pd.DataFrame()

        if 'sample_count' not in pc_samples.columns:
            pc_samples['sample_count'] = 0
        pc_samples['sample_count'] = pd.to_numeric(pc_samples['sample_count'], errors='coerce').fillna(0)
        if 'reason_name' not in pc_samples.columns:
            if 'stall_reason' in pc_samples.columns:
                pc_samples['reason_name'] = (
                    "Stall_" + pd.to_numeric(pc_samples['stall_reason'], errors='coerce')
                    .fillna(-1).astype(int).astype(str)
                )
            else:
                pc_samples['reason_name'] = "unknown"
        if float(pc_samples['sample_count'].sum()) <= 0.0:
            pc_samples = pd.DataFrame()

        if not pc_samples.empty:
            by_reason = (
                pc_samples.groupby('reason_name', dropna=False)['sample_count']
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
            )

            reason_table = Table(title=f"PC Sampling Reasons — Top {top_n}")
            reason_table.add_column("Reason", style="cyan")
            reason_table.add_column("Samples", justify="right")
            total_samples = float(pc_samples['sample_count'].sum()) or 1.0
            reason_table.add_column("Share", justify="right")

            for reason, count in by_reason.items():
                label = str(reason) if pd.notna(reason) and str(reason) else "unknown"
                reason_table.add_row(label, str(int(count)), f"{(count/total_samples)*100:.1f}%")
            self.console.print(reason_table)

            if not self.kernels.empty and 'corr_id' in self.kernels.columns and 'corr_id' in pc_samples.columns:
                sample_corr_ids = set(pc_samples['corr_id'].dropna().tolist())
                kernel_corr_ids = set(self.kernels['corr_id'].dropna().tolist())
                overlap_count = len(sample_corr_ids & kernel_corr_ids)
                if sample_corr_ids and overlap_count == 0:
                    self.console.print(
                        "[yellow]No corr_id overlap between profile_sample and kernel_event. "
                        "Kernel names will appear as 'unknown' (likely kernel throttling).[/yellow]"
                    )

                corr_to_name, fallback_used = self._resolve_sample_kernel_names(pc_samples)
                kernel_samples = pc_samples.groupby('corr_id', as_index=False)['sample_count'].sum()
                if corr_to_name:
                    map_df = pd.DataFrame(
                        [(k, v) for k, v in corr_to_name.items()],
                        columns=['corr_id', 'name']
                    )
                    kernel_samples = kernel_samples.merge(map_df, on='corr_id', how='left')
                if 'name' not in kernel_samples.columns:
                    kernel_samples['name'] = 'unknown'
                kernel_samples['name'] = kernel_samples['name'].fillna('unknown')
                kernel_samples = kernel_samples.sort_values('sample_count', ascending=False).head(top_n)

                kernel_table = Table(title=f"PC Sampling Kernels — Top {top_n}")
                kernel_table.add_column("Kernel", style="cyan")
                kernel_table.add_column("Samples", justify="right")
                for _, row in kernel_samples.iterrows():
                    kernel_table.add_row(str(row['name']), str(int(row['sample_count'])))
                self.console.print(kernel_table)
                if fallback_used > 0:
                    self.console.print(
                        f"[yellow]Kernel name fallback used for {fallback_used} corr_id(s) "
                        f"(function/time-based match).[/yellow]"
                    )
        else:
            self.console.print("[yellow]No pc_sampling records found in profile_sample stream.[/yellow]")

        if not sass_samples.empty:
            sass = sass_samples.copy()
            if 'metric_value' in sass.columns:
                sass['metric_value'] = pd.to_numeric(sass['metric_value'], errors='coerce').fillna(0)
            else:
                sass['metric_value'] = 0
            if 'metric_name' not in sass.columns:
                sass['metric_name'] = "unknown_metric"
            if 'function_name' not in sass.columns:
                sass['function_name'] = "unknown"

            by_metric = (
                sass.groupby('metric_name', dropna=False)['metric_value']
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
            )
            metric_table = Table(title=f"SASS Metrics — Top {top_n}")
            metric_table.add_column("Metric", style="cyan")
            metric_table.add_column("Total Value", justify="right")
            for metric, value in by_metric.items():
                metric_table.add_row(str(metric), str(int(value)))
            self.console.print(metric_table)

            by_func = (
                sass.groupby('function_name', dropna=False)['metric_value']
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
            )
            func_table = Table(title=f"SASS Functions — Top {top_n}")
            func_table.add_column("Function", style="cyan")
            func_table.add_column("Metric Sum", justify="right")
            for fn, value in by_func.items():
                func_table.add_row(str(fn), str(int(value)))
            self.console.print(func_table)

            # Per-function warp + memory efficiency (mirrors text_report.cpp /
            # hint_engine.cpp so this matches the C++ report number-for-number).
            eff_pivot = (
                sass.groupby(['function_name', 'metric_name'])['metric_value']
                .sum().unstack(fill_value=0)
            )

            def _ev(series, name):
                return float(series[name]) if name in series.index else 0.0

            eff_table = Table(title="SASS Efficiency — per function")
            eff_table.add_column("Function", style="cyan")
            eff_table.add_column("Warp Eff", justify="right")
            eff_table.add_column("Mem Eff", justify="right")
            eff_table.add_column("Hints", style="yellow")
            order = eff_pivot.sum(axis=1).sort_values(ascending=False)
            any_eff = False
            for fn in order.index[:top_n]:
                r        = eff_pivot.loc[fn]
                warp     = _ev(r, "smsp__sass_inst_executed")
                thread   = _ev(r, "smsp__sass_thread_inst_executed")
                gsect    = _ev(r, "smsp__sass_sectors_mem_global")
                ideal    = _ev(r, "smsp__sass_sectors_mem_global_ideal")
                ideal_ld = _ev(r, "smsp__sass_sectors_mem_global_op_ld_ideal")
                ideal_st = _ev(r, "smsp__sass_sectors_mem_global_op_st_ideal")
                if warp <= 0 and thread <= 0 and gsect <= 0:
                    continue
                eff_ideal = ideal if ideal > 0 else (ideal_ld + ideal_st)
                warp_pct = (thread / warp / 32.0 * 100) if (warp > 0 and thread > 0) else None
                mem_pct  = (eff_ideal / gsect * 100) if (gsect > 0 and eff_ideal > 0) else None
                warp_s = f"{warp_pct:.1f}%" if warp_pct is not None else "-"
                mem_s  = (f"{mem_pct:.1f}%" if mem_pct is not None
                          else ("n/a" if gsect > 0 else "-"))
                tips = []
                if warp_pct is not None and warp_pct < 90:
                    tips.append("divergence")
                if mem_pct is not None and mem_pct < 50:
                    tips.append("coalescing")
                fn_s = str(fn).split("@", 1)[0]
                if len(fn_s) > 44:
                    fn_s = fn_s[:41] + "..."
                eff_table.add_row(fn_s, warp_s, mem_s, ", ".join(tips))
                any_eff = True
            if any_eff:
                self.console.print(eff_table)
        else:
            self.console.print("[yellow]No sass_metric records found in profile_sample stream.[/yellow]")

    def inspect_scopes(self):
        """Analyze time spent in user-defined Scopes.

        Uses kernel user_scope annotations when GPU kernels are available,
        otherwise falls back to scope_event_batch begin/end pairs.
        """
        # ── Path 1: kernel user_scope (available when kernel_detail present) ──
        if not self.kernels.empty and 'user_scope' in self.kernels.columns:
            scoped = self.kernels.dropna(subset=['user_scope'])
            if not scoped.empty:
                agg_dict: dict = {'kernels': ('name', 'count'), 'gpu_time_ms': ('duration_ms', 'sum')}
                if 'queue_latency_ms' in scoped.columns:
                    agg_dict['avg_queue_ms'] = ('queue_latency_ms', 'mean')
                if 'cpu_overhead_ms' in scoped.columns:
                    agg_dict['cpu_overhead_ms'] = ('cpu_overhead_ms', 'sum')

                scope_stats = scoped.groupby('user_scope').agg(**agg_dict).sort_index()

                table = Table(title="📂 Scope Analysis (Hierarchical, kernel-based)")
                table.add_column("Scope / Phase", style="bold white")
                table.add_column("GPU Time", style="green", justify="right")
                if 'avg_queue_ms' in scope_stats.columns:
                    table.add_column("Queue Latency", style="red", justify="right")
                if 'cpu_overhead_ms' in scope_stats.columns:
                    table.add_column("CPU Overhead", style="yellow", justify="right")

                for scope, row in scope_stats.iterrows():
                    formatted_scope = scope.replace("|", " [dim]>[/dim] ")
                    cells = [formatted_scope, f"{row['gpu_time_ms']:.2f} ms"]
                    if 'avg_queue_ms' in scope_stats.columns:
                        cells.append(f"{row['avg_queue_ms']:.3f} ms")
                    if 'cpu_overhead_ms' in scope_stats.columns:
                        cells.append(f"{row['cpu_overhead_ms']:.2f} ms")
                    table.add_row(*cells)

                self.console.print(table)
                return

        # ── Path 2: scope_event_batch begin/end pairs ─────────────────────────
        if self.scope_events.empty:
            self.console.print("[yellow]No scope data found.[/yellow]")
            return

        # Stack-based pairing: handles recycled scope_instance_ids across runs.
        # Events are sorted by ts_ns; each begin is matched with the next end
        # sharing the same scope_instance_id.
        events_sorted = self.scope_events.sort_values('ts_ns')
        open_scopes: dict = {}   # scope_instance_id → begin row
        pair_rows = []
        for _, row in events_sorted.iterrows():
            key = row['scope_instance_id']
            if row['event_type'] == 0:   # begin
                open_scopes[key] = row
            elif key in open_scopes:     # end with matching begin
                begin = open_scopes.pop(key)
                pair_rows.append({
                    'name':        begin['name'],
                    'begin_ts':    begin['ts_ns'],
                    'end_ts':      row['ts_ns'],
                    'duration_ms': (row['ts_ns'] - begin['ts_ns']) / 1e6,
                    'depth':       begin['depth'],
                })

        if not pair_rows:
            self.console.print("[yellow]No complete scope begin/end pairs found.[/yellow]")
            return

        paired = pd.DataFrame(pair_rows)

        scope_stats = (
            paired.groupby('name')
            .agg(count=('duration_ms', 'count'),
                 total_ms=('duration_ms', 'sum'),
                 avg_ms=('duration_ms', 'mean'),
                 max_ms=('duration_ms', 'max'))
            .sort_values('total_ms', ascending=False)
        )

        table = Table(title="📂 Scope Analysis (begin/end pairs)")
        table.add_column("Scope", style="bold white")
        table.add_column("Count", justify="right")
        table.add_column("Total", style="green", justify="right")
        table.add_column("Avg", justify="right")
        table.add_column("Max", justify="right")

        for name, row in scope_stats.iterrows():
            table.add_row(
                str(name),
                str(int(row['count'])),
                f"{row['total_ms']:.2f} ms",
                f"{row['avg_ms']:.2f} ms",
                f"{row['max_ms']:.2f} ms",
            )

        self.console.print(table)

    # ------------------------------------------------------------------
    # Empty-state hints — printed when one of the inspect_* methods has
    # nothing to show. The methods themselves are correct; the data
    # simply isn't in the logs because the session wasn't configured to
    # collect it (wrong engine, no scopes, build missing the relevant
    # CUPTI subsystem, etc.). The hint blocks below name the exact
    # `init()` kwarg and other preconditions so users can fix the
    # session config rather than debugging the analyzer.
    # ------------------------------------------------------------------
    def _print_profile_sample_hint(self):
        self.console.print(
            "[yellow]No profile sample data found.[/yellow]\n"
            "  To collect PC sampling / SASS metrics:\n"
            "    1. [bold]profiling_engine[/bold]: pass "
            "[cyan]ProfilingEngine.PcSampling[/cyan], "
            "[cyan]ProfilingEngine.SassMetrics[/cyan], or "
            "[cyan]ProfilingEngine.Deep[/cyan] to "
            "[cyan]gpufl.init()[/cyan].\n"
            "    2. [bold]Scopes required[/bold]: wrap GPU work in "
            "[cyan]with gpufl.Scope(\"name\"):[/cyan] blocks — sample "
            "buffers flush only on scope close.\n"
            "    3. [bold]Hardware[/bold]: PC sampling needs compute "
            "capability ≥ 7.0 (Volta+); SASS metrics need ≥ 7.5 "
            "(Turing+).\n"
            "    4. [bold]Build[/bold]: wheel must include CUPTI "
            "(NVIDIA backend, full CUDA toolkit at build time)."
        )

    def _print_perf_metric_hint(self):
        self.console.print(
            "[yellow]No perf metric counter records found in logs.[/yellow]\n"
            "  To collect hardware-counter perf metrics:\n"
            "    1. [bold]profiling_engine[/bold]: pass "
            "[cyan]ProfilingEngine.RangeProfiler[/cyan] or "
            "[cyan]ProfilingEngine.RangeProfilerKernelReplay[/cyan] to "
            "[cyan]gpufl.init()[/cyan].\n"
            "    2. [bold]Scopes[/bold]: RangeProfiler emits on scope close; wrap GPU work in "
            "[cyan]with gpufl.Scope(\"name\"):[/cyan] blocks — perf "
            "metrics emit on scope close. Kernel replay emits per replayed kernel range.\n"
            "    3. [bold]Mutually exclusive with PC sampling[/bold]: "
            "RangeProfiler and PcSampling both use hardware perf "
            "counters; pick one per session.\n"
            "    4. [bold]Build[/bold]: wheel must have "
            "[cyan]GPUFL_HAS_PERFWORKS=1[/cyan] — requires the full "
            "CUPTI/Perfworks SDK (nvperf_host + nvperf_target) at "
            "CMake configure time.\n"
            "    5. [bold]Hardware[/bold]: Volta+ for SM throughput / "
            "DRAM bytes; Turing+ for tensor-core utilization."
        )

    def _print_pm_sampling_hint(self):
        self.console.print(
            "[yellow]No pm_sample_batch records found in scope log.[/yellow]\n"
            "  To collect PM sampling hardware timeline data:\n"
            "    1. [bold]profiling_engine[/bold]: pass "
            "[cyan]ProfilingEngine.PmSampling[/cyan] to "
            "[cyan]gpufl.init()[/cyan].\n"
            "    2. [bold]Scopes recommended[/bold]: wrap GPU work in "
            "[cyan]with gpufl.Scope(\"name\"):[/cyan] blocks so "
            "samples can be attributed to workload phases.\n"
            "    3. [bold]Build[/bold]: wheel must have "
            "[cyan]GPUFL_HAS_PERFWORKS=1[/cyan] and a CUDA toolkit with "
            "[cyan]cupti_pmsampling.h[/cyan]."
        )

    def inspect_pm_sampling(self, top_n: int = 10):
        """Summarize PM sampling hardware timeline rows."""
        pm = getattr(self, 'pm_samples', pd.DataFrame())
        if pm.empty:
            self._print_pm_sampling_hint()
            return

        p = pm.copy()
        p['value'] = pd.to_numeric(p['value'], errors='coerce')
        grouped = (p.groupby('metric_name', dropna=False)['value']
                   .agg(['count', 'mean', 'max'])
                   .reset_index()
                   .sort_values('max', ascending=False)
                   .head(top_n))

        table = Table(title="PM Sampling Hardware Timeline")
        table.add_column("Metric", style="cyan")
        table.add_column("Samples", justify="right")
        table.add_column("Avg", justify="right")
        table.add_column("Peak", justify="right")
        for _, row in grouped.iterrows():
            table.add_row(
                str(row.get('metric_name') or 'unknown'),
                str(int(row['count'])),
                f"{row['mean']:.3f}" if pd.notna(row['mean']) else "n/a",
                f"{row['max']:.3f}" if pd.notna(row['max']) else "n/a",
            )
        self.console.print(table)

    def inspect_perf_metrics(self, top_n: int = 10):
        """Summarize perf_metric_event records from scope logs."""
        if self.perf.empty:
            self._print_perf_metric_hint()
            return

        p = self.perf.copy()
        for col in ['sm_throughput_pct', 'l1_hit_rate_pct', 'l2_hit_rate_pct', 'tensor_active_pct']:
            if col in p.columns:
                # Backend uses -1.0 sentinel for unavailable counters.
                p.loc[p[col] < 0, col] = float('nan')
        for col in ['dram_read_bytes', 'dram_write_bytes']:
            if col in p.columns:
                p[col] = pd.to_numeric(p[col], errors='coerce')
                p.loc[p[col] < 0, col] = float('nan')

        def avg_if_exists(col: str):
            return p[col].dropna().mean() if col in p.columns else float('nan')

        def fmt_pct(v):
            return f"{v:.2f}" if pd.notna(v) else "n/a"

        summary = Table(title="Perf Metrics Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Average", justify="right")
        summary.add_row("SM Throughput (%)", fmt_pct(avg_if_exists('sm_throughput_pct')))
        summary.add_row("L1 Hit Rate (%)", fmt_pct(avg_if_exists('l1_hit_rate_pct')))
        summary.add_row("L2 Hit Rate (%)", fmt_pct(avg_if_exists('l2_hit_rate_pct')))
        summary.add_row("Tensor Active (%)", fmt_pct(avg_if_exists('tensor_active_pct')))
        if 'dram_read_bytes' in p.columns:
            summary.add_row("DRAM Read (avg)", _fmt_bytes(p['dram_read_bytes'].dropna().mean()))
        if 'dram_write_bytes' in p.columns:
            summary.add_row("DRAM Write (avg)", _fmt_bytes(p['dram_write_bytes'].dropna().mean()))
        self.console.print(summary)

        if 'name' not in p.columns:
            return

        for col in [
            'sm_throughput_pct', 'l1_hit_rate_pct', 'l2_hit_rate_pct',
            'tensor_active_pct', 'dram_read_bytes', 'dram_write_bytes'
        ]:
            if col not in p.columns:
                p[col] = float('nan')

        if 'kind' not in p.columns:
            p['kind'] = 'scope'

        agg = p.groupby(['kind', 'name'], dropna=False).agg(
            count=('name', 'count'),
            sm=('sm_throughput_pct', 'mean'),
            l1=('l1_hit_rate_pct', 'mean'),
            l2=('l2_hit_rate_pct', 'mean'),
            tensor=('tensor_active_pct', 'mean'),
            dram_r=('dram_read_bytes', 'mean'),
            dram_w=('dram_write_bytes', 'mean'),
        ).sort_values('count', ascending=False).head(top_n)

        table = Table(title=f"Perf Metrics by Target - Top {top_n}")
        table.add_column("Target", style="cyan")
        table.add_column("Name", style="cyan")
        table.add_column("Events", justify="right")
        table.add_column("SM%", justify="right")
        table.add_column("L1%", justify="right")
        table.add_column("L2%", justify="right")
        table.add_column("Tensor%", justify="right")
        table.add_column("DRAM Read", justify="right")
        table.add_column("DRAM Write", justify="right")

        for (kind, target_name), row in agg.iterrows():
            table.add_row(
                str(kind),
                str(target_name),
                str(int(row['count'])),
                fmt_pct(row['sm']),
                fmt_pct(row['l1']),
                fmt_pct(row['l2']),
                fmt_pct(row['tensor']),
                _fmt_bytes(row['dram_r']) if pd.notna(row['dram_r']) else "n/a",
                _fmt_bytes(row['dram_w']) if pd.notna(row['dram_w']) else "n/a",
            )
        self.console.print(table)
