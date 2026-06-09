param(
    [string]$ExePath,
    [int]$TimeoutSeconds = 120,
    [string[]]$Combos,
    [switch]$IncludeSass
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $ExePath) {
    $ExePath = Join-Path $RepoRoot "build\example\cuda\Release\multi_engine_demo.exe"
}
$ExePath = (Resolve-Path $ExePath).Path
$WorkDir = Split-Path $ExePath

if (-not $Combos -or $Combos.Count -eq 0) {
    $Combos = @(
        "Trace",
        "PcSampling",
        "PmSampling",
        "RangeProfiler",
        "Trace,PmSampling",
        "Trace,PcSampling",
        "PmSampling,PcSampling",
        "Trace,RangeProfiler"
    )
}
if ($IncludeSass) {
    $Combos += @(
        "SassMetrics",
        "Trace,SassMetrics"
    )
}

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "This PowerShell is not elevated. Some CUPTI engines may be skipped or produce no data."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $RepoRoot "runs\capability_matrix_$stamp"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Get-CapabilityLines {
    param(
        [string]$Root,
        [datetime]$Since
    )
    if (-not (Test-Path $Root)) { return @() }
    $files = Get-ChildItem -Path $Root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '\.(log|txt)(\.gz)?$' } |
        Where-Object { $_.LastWriteTime -ge $Since }
    $hits = New-Object System.Collections.Generic.List[object]
    foreach ($file in $files) {
        if ($file.Name.EndsWith(".gz")) {
            $fs = [System.IO.File]::OpenRead($file.FullName)
            try {
                $gz = New-Object System.IO.Compression.GZipStream($fs, [System.IO.Compression.CompressionMode]::Decompress)
                try {
                    $reader = New-Object System.IO.StreamReader($gz)
                    try {
                        while (($line = $reader.ReadLine()) -ne $null) {
                            if ($line.Contains('"type":"capture_capabilities"')) {
                                $hits.Add([pscustomobject]@{
                                    path = $file.FullName
                                    line = $line
                                })
                            }
                        }
                    } finally {
                        $reader.Dispose()
                    }
                } finally {
                    $gz.Dispose()
                }
            } finally {
                $fs.Dispose()
            }
        } else {
            Select-String -LiteralPath $file.FullName -Pattern '"type":"capture_capabilities"' -SimpleMatch |
                ForEach-Object {
                    $hits.Add([pscustomobject]@{
                        path = $file.FullName
                        line = $_.Line
                    })
                }
        }
    }
    return $hits.ToArray()
}

function Summarize-Capabilities {
    param([string]$JsonLine)
    try {
        $obj = $JsonLine | ConvertFrom-Json
    } catch {
        return @([pscustomobject]@{
            feature = "parse_error"
            requested = $false
            status = "parse_error"
            mode = ""
            reason_code = ""
            message = $_.Exception.Message
        })
    }
    return @($obj.capabilities | ForEach-Object {
        [pscustomobject]@{
            feature = $_.feature
            requested = [bool]$_.requested
            status = $_.status
            mode = $_.mode
            reason_code = $_.reason_code
            message = $_.message
        }
    })
}

$rows = New-Object System.Collections.Generic.List[object]

foreach ($combo in $Combos) {
    $safe = $combo -replace "[^A-Za-z0-9]+", "_"
    $out = Join-Path $RunRoot "$safe.out.txt"
    $err = Join-Path $RunRoot "$safe.err.txt"
    $startedAt = Get-Date

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

    $logRoot = Join-Path $WorkDir "multi_engine"
    $capLines = Get-CapabilityLines -Root $logRoot -Since $startedAt
    if ($capLines.Count -eq 0) {
        Write-Warning "No capture_capabilities event found for $combo under $logRoot"
        $rows.Add([pscustomobject]@{
            combo = $combo
            exit = $exitCode
            timed_out = $timedOut
            feature = "(missing)"
            requested = ""
            status = "missing"
            mode = ""
            reason_code = ""
            message = "No capture_capabilities event found"
            log_path = ""
        })
        continue
    }

    $latest = $capLines | Sort-Object path | Select-Object -Last 1
    $caps = Summarize-Capabilities -JsonLine $latest.line
    foreach ($cap in $caps) {
        $rows.Add([pscustomobject]@{
            combo = $combo
            exit = $exitCode
            timed_out = $timedOut
            feature = $cap.feature
            requested = $cap.requested
            status = $cap.status
            mode = $cap.mode
            reason_code = $cap.reason_code
            message = $cap.message
            log_path = $latest.path
        })
    }

    $interesting = $caps | Where-Object {
        $_.feature -in @(
            "kernel_events",
            "kernel_names",
            "kernel_details",
            "memcpy_activity",
            "sync_activity",
            "nvtx_markers",
            "graph_activity",
            "pc_sampling",
            "sass_metrics",
            "pm_sampling",
            "range_counters",
            "cubin_disassembly",
            "source_correlation"
        )
    }
    $interesting | Format-Table feature, requested, status, mode, reason_code -AutoSize
}

Remove-Item Env:\GPUFL_ENGINE_COMBO -ErrorAction SilentlyContinue

$summaryCsv = Join-Path $RunRoot "capabilities.csv"
$summaryMd = Join-Path $RunRoot "capabilities.md"
$rows | Export-Csv -NoTypeInformation -Path $summaryCsv

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# GPUFlight capability matrix")
$md.Add("")
$md.Add("- Date: $(Get-Date -Format s)")
$md.Add("- Elevated PowerShell: $isAdmin")
$md.Add("- Executable: $ExePath")
$md.Add("- Timeout seconds: $TimeoutSeconds")
$md.Add("")
$md.Add("| Combo | Feature | Requested | Status | Mode | Reason |")
$md.Add("|---|---|---:|---|---|---|")
foreach ($row in $rows) {
    $reason = ($row.reason_code -replace "\|", "\|")
    $md.Add("| $($row.combo) | $($row.feature) | $($row.requested) | $($row.status) | $($row.mode) | $reason |")
}
$md | Set-Content -Encoding UTF8 -Path $summaryMd

Write-Host ""
Write-Host "Summary CSV: $summaryCsv"
Write-Host "Summary MD : $summaryMd"
