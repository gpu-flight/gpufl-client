<#
.SYNOPSIS
    Runs the engine-coverage test suite, one engine per fresh process.

.DESCRIPTION
    CUPTI does not reliably tolerate multiple init/shutdown cycles in a single
    process, so each of the 5 profiling engines (None, PcSampling, SassMetrics,
    RangeProfiler, PcSamplingWithSass) is exercised in its own invocation of
    gpufl_tests.exe. Per-engine pass/fail is reported; script exits 1 if any
    engine fails.

.PARAMETER TestExe
    Path to gpufl_tests.exe. Default: build/tests/Release/gpufl_tests.exe
    relative to the repo root.

.EXAMPLE
    ./tests/run_engine_coverage.ps1
    ./tests/run_engine_coverage.ps1 -TestExe build/tests/Debug/gpufl_tests.exe
#>
param(
    [string]$TestExe = ""
)

$ErrorActionPreference = "Stop"

# Resolve default path relative to the script's repo root.
if (-not $TestExe) {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    $TestExe = Join-Path $repoRoot "build/tests/Release/gpufl_tests.exe"
}

if (-not (Test-Path $TestExe)) {
    Write-Error "Test binary not found: $TestExe`nBuild with: cmake --build build --config Release --target gpufl_tests"
    exit 2
}

$engines = @("None", "PcSampling", "SassMetrics", "RangeProfiler", "PcSamplingWithSass")
$failed = @()
$passed = @()
$skipped = @()

foreach ($engine in $engines) {
    Write-Host ""
    Write-Host "=== Engine: $engine ===" -ForegroundColor Cyan

    $filter = "AllEngines/*/$engine"
    # Some engines (PcSamplingWithSass) segfault during CUPTI-at-exit after
    # the test itself completes successfully. Gtest's PASSED/FAILED markers
    # in stdout are the source of truth, not the process exit code.
    # Lower error-action so stderr lines from the native binary don't get
    # treated as PowerShell errors and abort the script.
    $prevEA = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $TestExe "--gtest_filter=$filter" 2>&1 | Out-String
        $rc = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEA
    }
    Write-Host $output

    if ($output -match "\[  FAILED  \]") {
        $failed += $engine
        Write-Host "[$engine] FAILED (gtest reported failure; exit=$rc)" -ForegroundColor Red
    } elseif ($output -match "\[  PASSED  \] [1-9]\d* test") {
        $passed += $engine
        if ($rc -ne 0) {
            Write-Host "[$engine] PASSED (test OK; non-zero exit $rc ignored - likely CUPTI-at-exit crash)" -ForegroundColor Yellow
        } else {
            Write-Host "[$engine] PASSED" -ForegroundColor Green
        }
    } else {
        $failed += $engine
        Write-Host "[$engine] FAILED (no gtest verdict found; exit=$rc)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host ("  Passed: {0} [{1}]" -f $passed.Count, ($passed -join ", "))
if ($failed.Count -gt 0) {
    Write-Host ("  Failed: {0} [{1}]" -f $failed.Count, ($failed -join ", ")) -ForegroundColor Red
    exit 1
}
Write-Host "  All engines passed." -ForegroundColor Green
exit 0
