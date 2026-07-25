#!/usr/bin/env bash
# End-to-end verification for the bounded deep-profiling window.
#
# Written for a Linux box with working GPU performance-counter access (the
# RTX 3090 dev machine). The Windows 5060 cannot finish this: its PC sampling
# stops producing hardware samples and needs elevation plus a reboot, so the
# PC and SASS legs there are unverifiable.
#
# Three things get checked, in increasing order of what they prove:
#
#   A. EMBED    - an app that calls gpufl::deepWindow() itself, per engine.
#   B. INJECT   - `gpufl trace --deep-after/--deep-for` against a target with
#                 ZERO gpufl calls. This is the case the feature exists for,
#                 since a job whose source you can't edit can never call the
#                 API. Also the only leg that exercises the deferred engine
#                 start, which is where the clock-anchor bug lived.
#   C. OVERHEAD - what an armed-but-idle session costs. Deep capture is only
#                 worth carrying through a long run if idle is close to free,
#                 and that number has never been measured.
#
# Usage:
#   scripts/deep_window_e2e.sh [--build-dir DIR] [--out DIR] [--seconds N]
#                              [--skip-overhead] [--engines "A B C"]
#
# Requires: a build with GPUFL_ENABLE_NVIDIA=ON, BUILD_GPUFL_EXAMPLE=ON,
# BUILD_GPUFL_LAUNCHER=ON, BUILD_GPUFL_INJECT=ON; nvcc on PATH; python3.

set -uo pipefail

BUILD_DIR="build"
OUT_DIR=""
SECONDS_PER_RUN=12
SKIP_OVERHEAD=0
ENGINES="PmSampling PcSampling SassMetrics RangeProfilerKernelReplay"
REPS=3
DEBUG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --out)       OUT_DIR="$2"; shift 2 ;;
        --seconds)   SECONDS_PER_RUN="$2"; shift 2 ;;
        --engines)   ENGINES="$2"; shift 2 ;;
        --reps)      REPS="$2"; shift 2 ;;
        --skip-overhead) SKIP_OVERHEAD=1; shift ;;
        # Turns on GPUFL_DEBUG in every run: the engine diagnostics are the
        # only way to tell "armed but the hardware gave nothing" apart from
        # "never armed".
        --debug)     DEBUG=1; shift ;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
[[ -n "$OUT_DIR" ]] || OUT_DIR="$REPO_DIR/deep_window_e2e_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

CHECK="$SCRIPT_DIR/deep_window_check.py"
REPORT="$OUT_DIR/report.md"
[[ "$DEBUG" -eq 1 ]] && export GPUFL_DEBUG=1

# Build dirs differ between single- and multi-config generators.
# Absolute, because the embed leg runs from inside its own output directory
# (the demo writes its logs relative to cwd) and a relative path would not
# survive the cd.
find_bin() {
    local name="$1" p
    for p in "$BUILD_DIR/$2/$name" "$BUILD_DIR/$2/Release/$name" \
             "$BUILD_DIR/$name" "$BUILD_DIR/Release/$name"; do
        [[ -x "$p" ]] && { (cd "$(dirname "$p")" && printf '%s/%s\n' "$PWD" "$name"); return 0; }
    done
    return 1
}

DEMO="$(find_bin deep_window_demo example/cuda)" || {
    echo "deep_window_demo not found under $BUILD_DIR - build it first:" >&2
    echo "  cmake -S . -B $BUILD_DIR -DGPUFL_ENABLE_NVIDIA=ON -DBUILD_GPUFL_EXAMPLE=ON \\" >&2
    echo "        -DBUILD_GPUFL_LAUNCHER=ON -DBUILD_GPUFL_INJECT=ON" >&2
    echo "  cmake --build $BUILD_DIR -j" >&2
    exit 2
}
GPUFL="$(find_bin gpufl daemon/launcher)" || {
    echo "gpufl launcher not found under $BUILD_DIR (BUILD_GPUFL_LAUNCHER=ON?)" >&2
    exit 2
}

TARGET="$OUT_DIR/deep_window_target"
echo "[e2e] compiling the gpufl-unaware target"
nvcc -O2 -lineinfo -o "$TARGET" "$SCRIPT_DIR/deep_window_target.cu" \
    >"$OUT_DIR/nvcc.log" 2>&1 || { cat "$OUT_DIR/nvcc.log"; exit 2; }

{
    echo "# Deep window E2E"
    echo
    echo "- host: \`$(hostname)\`"
    echo "- gpu: \`$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)\`"
    echo "- build: \`$BUILD_DIR\`"
    echo "- run seconds: $SECONDS_PER_RUN"
    echo
} > "$REPORT"

PASS=0
FAIL=0
note() { echo "$@" | tee -a "$REPORT"; }
record() {  # name, exit code
    if [[ "$2" -eq 0 ]]; then PASS=$((PASS+1)); note "- **PASS** $1";
    else FAIL=$((FAIL+1)); note "- **FAIL** $1"; fi
}

# ── A. embed ────────────────────────────────────────────────────────────────
note "## A. Embedded API (app calls gpufl::deepWindow)"
note
for eng in $ENGINES; do
    d="$OUT_DIR/embed_$eng"; mkdir -p "$d"
    echo "[e2e] embed: $eng"
    # A faster card gets through the default loop before the window expires,
    # which closes it at session stop and proves nothing. Scale with
    # GPUFL_DEMO_ITERATIONS if the check reports that.
    ( cd "$d" && GPUFL_PROFILING_ENGINE="$eng" \
        GPUFL_DEMO_ITERATIONS="${GPUFL_DEMO_ITERATIONS:-3000}" "$DEMO" ) \
        >"$d/stdout.txt" 2>&1
    demo_rc=$?
    python3 "$CHECK" "$d/deep_window" >"$d/check.txt" 2>&1
    chk_rc=$?
    sed 's/^/    /' "$d/check.txt" >> "$REPORT"
    [[ $demo_rc -eq 0 && $chk_rc -eq 0 ]]
    record "embed $eng" $?
done
note

# ── B. injection ────────────────────────────────────────────────────────────
# The load-bearing leg: the target has no gpufl calls, so the only possible
# trigger is the launcher's. Also the only path that runs the deferred engine
# start, where the wall-clock anchor was being dropped.
note "## B. Injection (\`gpufl trace\`, target has zero gpufl calls)"
note
for eng in $ENGINES; do
    d="$OUT_DIR/inject_$eng"; mkdir -p "$d"
    echo "[e2e] inject: $eng"
    "$GPUFL" trace -o "$d/trace" --passes "$eng" \
        --deep-after 3s --deep-for 2s -- "$TARGET" "$SECONDS_PER_RUN" \
        >"$d/stdout.txt" 2>&1
    trace_rc=$?
    python3 "$CHECK" "$d/trace" >"$d/check.txt" 2>&1
    chk_rc=$?
    sed 's/^/    /' "$d/check.txt" >> "$REPORT"
    [[ $trace_rc -eq 0 && $chk_rc -eq 0 ]]
    record "inject $eng" $?
done

# Launch-bound leg: the bound that actually suits the replay engines, where a
# second of wall time buys far less work.
d="$OUT_DIR/inject_launch_bound"; mkdir -p "$d"
echo "[e2e] inject: SassMetrics bounded by launches"
"$GPUFL" trace -o "$d/trace" --passes SassMetrics \
    --deep-after 3s --deep-launches 200 -- "$TARGET" "$SECONDS_PER_RUN" \
    >"$d/stdout.txt" 2>&1
trace_rc=$?
python3 "$CHECK" "$d/trace" >"$d/check.txt" 2>&1
chk_rc=$?
sed 's/^/    /' "$d/check.txt" >> "$REPORT"
[[ $trace_rc -eq 0 && $chk_rc -eq 0 ]]
record "inject SassMetrics --deep-launches 200" $?
note

# ── C. armed-but-idle overhead ──────────────────────────────────────────────
# Conditions are interleaved rather than run in blocks so thermal drift hits
# all of them equally.
if [[ "$SKIP_OVERHEAD" -eq 0 ]]; then
    note "## C. Overhead (iterations/sec, median of $REPS)"
    note
    declare -A RESULTS
    declare -i probe_n=0
    run_probe() {  # label, then argv for the run
        local label="$1"; shift
        probe_n+=1
        local log="$OUT_DIR/ovh_${probe_n}_$(tr -c 'a-zA-Z0-9' '_' <<<"$label").log"
        local out
        # Keep the log: a probe that yields no throughput line is a run that
        # failed, and discarding stderr makes that indistinguishable from a
        # parse bug.
        out="$("$@" 2>"$log" | tee -a "$log" \
               | grep -o 'ITERS_PER_SEC=[0-9.]*' | tail -1)"
        [[ -n "$out" ]] || echo "[e2e] probe '$label' produced no throughput line; see $log" >&2
        RESULTS["$label"]+="${out#ITERS_PER_SEC=} "
    }
    for _ in $(seq 1 "$REPS"); do
        run_probe "no gpufl" "$TARGET" "$SECONDS_PER_RUN"
        run_probe "trace (light tier)" \
            "$GPUFL" trace -o "$OUT_DIR/ovh_light" --passes Trace -- "$TARGET" "$SECONDS_PER_RUN"
        # Window-only arming with no window ever opened - the idle cost. Set
        # via env because --deep-* always opens one.
        export GPUFL_DEEP_ARM=window
        run_probe "PM armed-but-idle" \
            "$GPUFL" trace -o "$OUT_DIR/ovh_idle" --passes PmSampling -- "$TARGET" "$SECONDS_PER_RUN"
        unset GPUFL_DEEP_ARM
        run_probe "PM always armed" \
            "$GPUFL" trace -o "$OUT_DIR/ovh_always" --passes PmSampling -- "$TARGET" "$SECONDS_PER_RUN"
    done

    median() { tr ' ' '\n' <<<"$1" | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
    base="$(median "${RESULTS["no gpufl"]:-}")"
    note "| condition | iters/sec | vs no gpufl |"
    note "|---|---|---|"
    for label in "no gpufl" "trace (light tier)" "PM armed-but-idle" "PM always armed"; do
        m="$(median "${RESULTS[$label]:-}")"
        if [[ -n "$m" && -n "$base" && "$base" != "0" ]]; then
            pct="$(awk -v a="$m" -v b="$base" 'BEGIN{printf "%+.1f%%", (a/b-1)*100}')"
        else pct="n/a"; fi
        note "| $label | ${m:-n/a} | $pct |"
    done
    note
    note "\`PM armed-but-idle\` is the number that decides whether deep capture"
    note "can ride along in a long run. \`PM always armed\` is what the window avoids."
    note
fi

note "## Summary"
note
note "passed: $PASS, failed: $FAIL"
echo
echo "[e2e] report: $REPORT"
[[ "$FAIL" -eq 0 ]]
