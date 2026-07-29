#!/usr/bin/env python3
"""Workload-mode benchmark: evaluator cost on a fixed-work decode loop.

Five configurations, identical target and arguments; only the rule differs.
Which one fires is decided by the threshold, never by reshaping the workload.

  bare          no gpufl at all - the floor
  no_evaluator  gpufl trace, PM prepared+dormant (--deep-after far future),
                NO rule: the profiling baseline the evaluator adds onto
  rule_missing  rule on a counter that never registers: evaluator polls a
                Missing source all run
  rule_armed    rule on the live counter, threshold below any real rate:
                full metric pipeline, condition never true
  rule_fires    threshold above the steady rate: condition true from warmup,
                exactly one PM deep window fires mid-run

Paired randomized blocks: each block runs all five in a shuffled order, and
ratios are computed within the block before aggregating.
"""
import csv
import gzip
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys

HOME = os.path.expanduser("~")
GPUFL = HOME + "/sources/gpufl-client/build/daemon/launcher/gpufl"
TARGET = "/tmp/wl/target"
OUTROOT = "/tmp/wl/runs"
CSV_PATH = "/tmp/wl/results.csv"
BLOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ITERS = sys.argv[2] if len(sys.argv) > 2 else "12000"
TOKENS = "32"
RULE_LIVE = "custom.bench.tokens_rate"
DEEP = ["--deep-for", "2s", "--deep-cooldown", "600s"]


def config_cmd(name, outdir):
    base = ["/usr/bin/time", "-v"]
    tgt = [TARGET, ITERS, TOKENS, "400000"]
    if name == "bare":
        return base + tgt
    launch = base + [GPUFL, "trace", "-o", outdir]
    if name == "no_evaluator":
        return launch + ["--deep-after", "100000s"] + DEEP + ["--"] + tgt
    if name == "rule_missing":
        return launch + ["--deep-when", "custom.bench.missing_rate<1 for 1s"] + DEEP + ["--"] + tgt
    if name == "rule_armed":
        return launch + ["--deep-when", RULE_LIVE + "<1 for 1s"] + DEEP + ["--"] + tgt
    if name == "rule_fires":
        return launch + ["--deep-when", RULE_LIVE + "<999999999 for 1s"] + DEEP + ["--"] + tgt
    raise ValueError(name)


CONFIGS = ["bare", "no_evaluator", "rule_missing", "rule_armed", "rule_fires"]


def parse_time_v(err):
    user = sys_t = rss_kb = None
    for line in err.splitlines():
        if "User time (seconds):" in line:
            user = float(line.split(":")[-1])
        elif "System time (seconds):" in line:
            sys_t = float(line.split(":")[-1])
        elif "Maximum resident set size" in line:
            rss_kb = int(line.split(":")[-1])
    return user, sys_t, rss_kb


def scan_trace(outdir):
    windows = 0
    outcome = ""
    if not os.path.isdir(outdir):
        return windows, outcome
    for root, _dirs, files in os.walk(outdir):
        for f in files:
            if not f.endswith(".log.gz"):
                continue
            try:
                with gzip.open(os.path.join(root, f), "rt", errors="replace") as fh:
                    for line in fh:
                        if '"type":"deep_window_event"' in line:
                            windows += 1
                        elif '"type":"deep_window_rule_summary"' in line:
                            m = re.search(r'"outcome":"([a-z_]+)"', line)
                            if m:
                                outcome = m.group(1)
            except OSError:
                pass
    return windows, outcome


def one_run(name, block):
    outdir = "%s/%s_b%d" % (OUTROOT, name, block)
    shutil.rmtree(outdir, ignore_errors=True)
    cmd = config_cmd(name, outdir)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"iters_per_sec=([0-9.]+)", proc.stdout)
    if proc.returncode != 0 or not m:
        print("RUN FAILED", name, "rc=", proc.returncode, file=sys.stderr)
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    ips = float(m.group(1))
    wall = float(re.search(r"secs=([0-9.]+)", proc.stdout).group(1))
    user, sys_t, rss_kb = parse_time_v(proc.stderr)
    windows, outcome = scan_trace(outdir)
    shutil.rmtree(outdir, ignore_errors=True)
    return {
        "block": block, "config": name, "iters_per_sec": ips, "wall_s": wall,
        "cpu_s": round((user or 0) + (sys_t or 0), 3),
        "maxrss_mb": round((rss_kb or 0) / 1024.0, 1),
        "windows": windows, "outcome": outcome,
    }


def governor():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def boot_ci(values, iters=10000):
    n = len(values)
    meds = sorted(statistics.median(random.choices(values, k=n))
                  for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


def main():
    random.seed(20260728)
    os.makedirs(OUTROOT, exist_ok=True)
    print("governor:", governor(), flush=True)
    rows = []
    for block in range(BLOCKS):
        order = CONFIGS[:]
        random.shuffle(order)
        print("block %d order: %s" % (block, " ".join(order)), flush=True)
        for name in order:
            row = one_run(name, block)
            rows.append(row)
            print("  %-13s %9.1f it/s  cpu %6.2fs  rss %7.1fMB  win %d %s"
                  % (row["config"], row["iters_per_sec"], row["cpu_s"],
                     row["maxrss_mb"], row["windows"], row["outcome"]),
                  flush=True)

    # ── gates ───────────────────────────────────────────────────────────
    failures = []
    expected_rows = BLOCKS * len(CONFIGS)
    if len(rows) != expected_rows:
        failures.append("row count %d != %d" % (len(rows), expected_rows))

    # windows / outcome each configuration MUST show. no_evaluator and bare
    # have no rule, so no rule summary may exist at all.
    expect = {
        "bare":         (0, ""),
        "no_evaluator": (0, ""),
        "rule_missing": (0, "never_true"),
        "rule_armed":   (0, "never_true"),
        "rule_fires":   (1, "fired"),
    }
    for r in rows:
        want_win, want_outcome = expect[r["config"]]
        if r["windows"] != want_win or r["outcome"] != want_outcome:
            failures.append(
                "block %d %s: windows=%d outcome=%r (want %d %r)" %
                (r["block"], r["config"], r["windows"], r["outcome"],
                 want_win, want_outcome))

    if failures:
        for f in failures:
            print("GATE FAILED:", f, file=sys.stderr)
        sys.exit(1)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r)

    print("\n== medians over %d blocks (iters=%s) ==" % (BLOCKS, ITERS))
    print("%-13s %10s %8s %9s %7s %s" %
          ("config", "it/s", "cpu_s", "rss_MB", "windows", "outcome"))
    for name in CONFIGS:
        rs = by.get(name, [])
        if not rs:
            continue
        print("%-13s %10.1f %8.2f %9.1f %7d %s" % (
            name,
            statistics.median(x["iters_per_sec"] for x in rs),
            statistics.median(x["cpu_s"] for x in rs),
            statistics.median(x["maxrss_mb"] for x in rs),
            max(x["windows"] for x in rs),
            rs[0]["outcome"]))

    print("\n== paired throughput ratios (per block, vs no_evaluator) ==")
    base = {r["block"]: r["iters_per_sec"] for r in by.get("no_evaluator", [])}
    for name in CONFIGS:
        if name == "no_evaluator":
            continue
        ratios = [r["iters_per_sec"] / base[r["block"]]
                  for r in by.get(name, []) if r["block"] in base]
        if not ratios:
            continue
        med = statistics.median(ratios)
        lo, hi = boot_ci(ratios)
        print("  %-13s x%.4f [%.4f, %.4f]" % (name, med, lo, hi))

    print("\ngovernor after:", governor())
    print("WL_DRIVER_DONE all_gates_passed rows=%d" % len(rows))


if __name__ == "__main__":
    main()
