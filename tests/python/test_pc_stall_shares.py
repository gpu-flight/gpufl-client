import io
import json
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from gpufl.analyzer import GpuFlightSession
from gpufl.report import generate_report

# Reason names and indices as the PC Sampling API enumerates them on an
# RTX 5060 (CUDA 13.3). Sample counts are synthetic. CUPTI counts each sample
# once under its warp state and again under the _not_issued twin when the
# scheduler issued nothing that cycle, so twin <= state for every PC.
_P = "smsp__pcsamp_warps_issue_stalled_"
_METRICS = {
    "1": _P + "long_scoreboard",
    "2": _P + "long_scoreboard_not_issued",
    "3": _P + "wait",
    "4": _P + "wait_not_issued",
    "5": _P + "selected",
    "6": _P + "not_selected",
}
_COLUMNS = ["dt_ns", "corr_id", "device_id", "function_id", "pc_offset",
            "metric_id", "metric_value", "stall_reason", "sample_kind",
            "scope_name_id", "source_file_id", "source_line"]


def _row(corr_id, function_id, pc, metric_id, samples, stall_index):
    return [0, corr_id, 0, function_id, pc, metric_id, samples, stall_index,
            0, 0, 0, 0]


# memBound (corr 4): 100 samples, 90 not-issued.
# issueBound (corr 5): 120 samples, 10 not-issued.
_ROWS = [
    _row(4, 1, 192, 1, 90, 12),
    _row(4, 1, 192, 2, 85, 13),
    _row(4, 1, 128, 3, 6, 34),
    _row(4, 1, 128, 4, 5, 35),
    _row(4, 1, 64, 5, 4, 26),
    _row(5, 2, 64, 5, 70, 26),
    _row(5, 2, 128, 3, 30, 34),
    _row(5, 2, 128, 4, 10, 35),
    _row(5, 2, 32, 6, 20, 24),
]


def _write_session(tmp_path, prefix="pcs"):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    device = [
        {"version": 1, "type": "job_start", "session_id": "pcs-session",
         "app": "pcs", "pid": 1, "ts_ns": 100, "host": {}, "devices": []},
        {"version": 1, "type": "dictionary_update",
         "session_id": "pcs-session",
         "kernel_dict": {"1": "memBound", "2": "issueBound"},
         "scope_name_dict": {}, "function_dict": {}, "metric_dict": {}},
        {"version": 1, "type": "kernel_event_batch",
         "session_id": "pcs-session", "batch_id": 1, "base_time_ns": 1000,
         "columns": ["dt_ns", "kernel_id", "stream_id", "duration_ns",
                     "corr_id", "dyn_shared", "num_regs", "has_details"],
         "rows": [[0, 1, 0, 100, 4, 0, 16, 0],
                  [200, 2, 0, 100, 5, 0, 16, 0]]},
        {"type": "shutdown", "session_id": "pcs-session", "app": "pcs",
         "pid": 1, "ts_ns": 5000},
    ]
    scope = [
        {"version": 1, "type": "dictionary_update",
         "session_id": "pcs-session",
         "function_dict": {"1": "memBound@", "2": "issueBound@"},
         "metric_dict": _METRICS},
        {"version": 1, "type": "profile_sample_batch",
         "session_id": "pcs-session", "batch_id": 1, "base_time_ns": 2000,
         "columns": _COLUMNS, "rows": _ROWS},
    ]
    for channel, events in (("device", device), ("scope", scope)):
        with open(log_dir / f"{prefix}.{channel}.log", "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
    return log_dir, prefix


def _session(tmp_path):
    log_dir, prefix = _write_session(tmp_path)
    session = GpuFlightSession(log_dir, log_prefix=prefix)
    buf = io.StringIO()
    session.console = Console(file=buf, width=200, color_system=None)
    return session, buf


def _lines(text, label):
    return [line for line in text.splitlines() if label in line]


def test_pc_reason_names_come_from_the_sampling_api_name(tmp_path):
    session, _ = _session(tmp_path)
    pc = session.scopes[session.scopes["type"] == "profile_sample"]

    assert set(pc["reason_name"]) == {
        "long_scoreboard", "wait", "selected", "not_selected"}
    assert int(pc.loc[pc["not_issued"], "sample_count"].sum()) == 100
    assert int(pc.loc[~pc["not_issued"], "sample_count"].sum()) == 220


def test_stall_shares_exclude_not_issued_samples(tmp_path):
    session, buf = _session(tmp_path)

    session.inspect_stalls()
    out = buf.getvalue()

    mem = _lines(out, "memBound")
    issue = _lines(out, "issueBound")
    assert mem and issue, out
    assert " 100 " in mem[0] and "90.0%" in mem[0], out
    assert " 120 " in issue[0] and "58.3%" in issue[0], out
    # Ranked by samples, not by samples plus their not-issued twins.
    assert out.index("issueBound") < out.index("memBound"), out


def test_not_issued_breakdown_uses_its_own_denominator(tmp_path):
    session, buf = _session(tmp_path)

    session.inspect_stalls(not_issued=True)
    out = buf.getvalue()

    mem = _lines(out, "memBound")
    assert mem and " 90 " in mem[0] and "94.4%" in mem[0], out


def test_reason_table_reports_both_families_separately(tmp_path):
    session, buf = _session(tmp_path)

    session.inspect_profile_samples()
    out = buf.getvalue()

    ls = _lines(out, "long_scoreboard")
    assert ls, out
    # 90 of 220 samples; 85 of 100 not-issued samples.
    assert "40.9%" in ls[0] and "85.0%" in ls[0], out
    kernels = _lines(out, "issueBound")
    assert kernels and " 120 " in kernels[-1], out


def test_text_report_stall_distribution_excludes_not_issued(tmp_path):
    log_dir, prefix = _write_session(tmp_path)

    report = generate_report(str(log_dir), log_prefix=prefix)

    ls = _lines(report, "long_scoreboard")
    assert ls and "40.9%" in ls[0] and "85.0%" in ls[0], report
    ranked = _lines(report, " samples")
    assert len(ranked) == 2, report
    assert "issueBound" in ranked[0] and "120 samples" in ranked[0], report
    assert "memBound" in ranked[1] and "100 samples" in ranked[1], report
