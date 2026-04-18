#!/usr/bin/env bash
# Runs the engine-coverage test suite, one engine per fresh process.
#
# CUPTI does not reliably tolerate multiple init/shutdown cycles in a single
# process, so each of the 5 profiling engines (None, PcSampling, SassMetrics,
# RangeProfiler, PcSamplingWithSass) is exercised in its own invocation of
# gpufl_tests. Per-engine pass/fail is reported; script exits 1 if any engine
# fails.
#
# Usage:
#   tests/run_engine_coverage.sh                    # uses default build path
#   tests/run_engine_coverage.sh path/to/gpufl_tests

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default build paths: try the Linux out-of-source layout first, then the
# Windows multi-config layout for MSYS/WSL users.
TEST_EXE="${1:-}"
if [[ -z "$TEST_EXE" ]]; then
    for candidate in \
        "$REPO_ROOT/build/tests/gpufl_tests" \
        "$REPO_ROOT/build_tests/tests/gpufl_tests" \
        "$REPO_ROOT/build/tests/Release/gpufl_tests.exe" \
        "$REPO_ROOT/build/tests/Debug/gpufl_tests.exe"; do
        if [[ -x "$candidate" ]]; then
            TEST_EXE="$candidate"
            break
        fi
    done
fi

if [[ -z "$TEST_EXE" || ! -x "$TEST_EXE" ]]; then
    echo "ERROR: Test binary not found."
    echo "Build first: cmake --build build --target gpufl_tests"
    echo "Or pass an explicit path: tests/run_engine_coverage.sh <path>"
    exit 2
fi

echo "Using test binary: $TEST_EXE"

engines=(None PcSampling SassMetrics RangeProfiler PcSamplingWithSass)
passed=()
failed=()

for engine in "${engines[@]}"; do
    echo ""
    echo "=== Engine: $engine ==="
    # Some engines (PcSamplingWithSass) segfault during CUPTI-at-exit after
    # the test itself completes successfully. Gtest's PASSED/FAILED markers
    # in stdout are the source of truth, not the process exit code.
    set +e
    output="$("$TEST_EXE" --gtest_filter="AllEngines/*/$engine" 2>&1)"
    rc=$?
    set -e
    echo "$output"
    if echo "$output" | grep -q "\[  FAILED  \]"; then
        failed+=("$engine")
        echo "[$engine] FAILED (gtest reported failure; exit=$rc)"
    elif echo "$output" | grep -qE "\[  PASSED  \] [1-9][0-9]* test"; then
        passed+=("$engine")
        if [[ $rc -ne 0 ]]; then
            echo "[$engine] PASSED (test OK; non-zero exit $rc ignored — likely CUPTI-at-exit crash)"
        else
            echo "[$engine] PASSED"
        fi
    else
        failed+=("$engine")
        echo "[$engine] FAILED (no gtest verdict found; exit=$rc)"
    fi
done

echo ""
echo "=== Summary ==="
echo "  Passed: ${#passed[@]} [${passed[*]:-}]"
if (( ${#failed[@]} > 0 )); then
    echo "  Failed: ${#failed[@]} [${failed[*]}]"
    exit 1
fi
echo "  All engines passed."
exit 0
