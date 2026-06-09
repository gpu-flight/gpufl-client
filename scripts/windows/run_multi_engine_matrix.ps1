param(
    [string]$ExePath,
    [int]$TimeoutSeconds = 120,
    [switch]$IncludeSass,
    [string[]]$Combos
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $ExePath) {
    $ExePath = Join-Path $RepoRoot "build\example\cuda\Release\multi_engine_demo.exe"
}
$ExePath = (Resolve-Path $ExePath).Path
$WorkDir = Split-Path $ExePath

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "This PowerShell is not elevated. PcSampling/PmSampling/RangeProfiler may report CUPTI_ERROR_INSUFFICIENT_PRIVILEGES."
}

if (-not $Combos -or $Combos.Count -eq 0) {
    $Combos = @(
        "Trace",
        "PcSampling",
        "RangeProfiler",
        "Trace,PcSampling",
        "Trace,PmSampling",
        "Trace,RangeProfiler",
        "Trace,PcSampling,PmSampling",
        "Trace,PcSampling,RangeProfiler",
        "Trace,PmSampling,RangeProfiler",
        "Trace,PcSampling,PmSampling,RangeProfiler"
    )
}
if ($IncludeSass) {
    $Combos += "Trace,SassMetrics"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $RepoRoot "runs\multi_engine_matrix_$stamp"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$summary = New-Object System.Collections.Generic.List[object]

foreach ($combo in $Combos) {
    $safe = $combo -replace "[^A-Za-z0-9]+", "_"
    $out = Join-Path $RunRoot "$safe.out.txt"
    $err = Join-Path $RunRoot "$safe.err.txt"

    Write-Host ""
    Write-Host "=== RUN $combo ==="

    $env:GPUFL_ENGINE_COMBO = $combo
    $proc = Start-Process -FilePath $ExePath `
        -WorkingDirectory $WorkDir `
        -PassThru `
        -RedirectStandardOutput $out `
        -RedirectStandardError $err

    $finished = $proc.WaitForExit($TimeoutSeconds * 1000)
    if (-not $finished) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $exitCode = "TIMEOUT"
        $timedOut = $true
    } else {
        $proc.Refresh()
        $exitCode = $proc.ExitCode
        $timedOut = $false
    }

    $outText = if (Test-Path $out) { Get-Content -Raw -LiteralPath $out } else { "" }
    $errText = if (Test-Path $err) { Get-Content -Raw -LiteralPath $err } else { "" }
    $allLines = ($outText + "`n" + $errText) -split "`r?`n"

    $matrixLines = @($allLines | Where-Object { $_ -match "\[Composite\]\[matrix\]\s+\S+Engine\s+armed=" })
    $notableLines = @($allLines | Where-Object {
        $_ -match "CUPTI_ERROR|INSUFFICIENT|UNKNOWN|NOT_INITIALIZED|Failed|ERROR|deadlock|exception|timeout"
    } | Select-Object -First 20)

    Write-Host "exit=$exitCode"
    if ($matrixLines.Count -gt 0) {
        $matrixLines | ForEach-Object { Write-Host $_ }
    }
    if ($notableLines.Count -gt 0) {
        Write-Host "notable:"
        $notableLines | ForEach-Object { Write-Host $_ }
    }

    $summary.Add([pscustomobject]@{
        combo = $combo
        exit = $exitCode
        timed_out = $timedOut
        matrix = ($matrixLines -join " | ")
        notable = ($notableLines -join " | ")
        stdout = $out
        stderr = $err
    })
}

Remove-Item Env:\GPUFL_ENGINE_COMBO -ErrorAction SilentlyContinue

$summaryCsv = Join-Path $RunRoot "summary.csv"
$summaryMd = Join-Path $RunRoot "summary.md"
$summary | Export-Csv -NoTypeInformation -Path $summaryCsv

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# GPUFlight multi-engine compatibility run")
$md.Add("")
$md.Add("- Date: $(Get-Date -Format s)")
$md.Add("- Elevated PowerShell: $isAdmin")
$md.Add("- Executable: $ExePath")
$md.Add("- Timeout seconds: $TimeoutSeconds")
$md.Add("")
$md.Add("| Combo | Exit | Timeout | Matrix | Notable |")
$md.Add("|---|---:|---:|---|---|")
foreach ($row in $summary) {
    $matrix = ($row.matrix -replace "\|", "\|") -replace "`r?`n", " "
    $notable = ($row.notable -replace "\|", "\|") -replace "`r?`n", " "
    $md.Add("| $($row.combo) | $($row.exit) | $($row.timed_out) | $matrix | $notable |")
}
$md | Set-Content -Encoding UTF8 -Path $summaryMd

Write-Host ""
Write-Host "Summary CSV: $summaryCsv"
Write-Host "Summary MD : $summaryMd"
$summary | Format-Table -AutoSize
