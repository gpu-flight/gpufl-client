param(
    [ValidateSet("install", "wheel", "trace")]
    [string]$Mode = "install",

    [string]$Python = $env:PYTHON,

    [string]$CudaPath = $env:CUDA_PATH,

    [string]$WheelDir,

    [switch]$NoVcVars
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = "python"
}
if ([string]::IsNullOrWhiteSpace($WheelDir)) {
    $WheelDir = Join-Path $RootDir "dist"
}

function Resolve-CudaPath {
    param([string]$RequestedPath, [bool]$Explicit)

    # Prefer the NEWEST installed toolkit: CUPTI's profiler/counter
    # subsystem must be at least as new as the display driver, or
    # cuptiProfilerInitialize fails (CUPTI_ERROR_NOT_INITIALIZED) and
    # cuptiPCSamplingGetNumStallReasons "succeeds" with ZERO stall reasons —
    # the PcSampling pass then silently collects nothing (verified live on
    # driver 592.01: CUPTI 13.2 = 0 stall reasons, CUPTI 13.3 = 36).
    #
    # The curated list outranks ambient CUDA_PATH/CUDA_HOME unless the
    # caller passed -CudaPath explicitly, so a stale env var can't silently
    # downgrade the toolkit.
    $curated = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    )
    $ambient = @($RequestedPath, $env:CUDA_HOME)
    $candidates = $(if ($Explicit) { @($RequestedPath) + $curated + @($env:CUDA_HOME) }
                    else { $curated + $ambient }) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    foreach ($candidate in $candidates) {
        $nvcc = Join-Path $candidate "bin\nvcc.exe"
        if (Test-Path -LiteralPath $nvcc) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "CUDA nvcc was not found. Pass -CudaPath or set CUDA_PATH/CUDA_HOME."
}

function Set-UniquePath {
    param([string[]]$Prepend = @())

    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $entries = @($Prepend + ($env:Path -split ";")) |
        ForEach-Object { $_.Trim() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    $unique = foreach ($entry in $entries) {
        $key = $entry.TrimEnd("\")
        if ($seen.Add($key)) {
            $entry
        }
    }

    $env:Path = $unique -join ";"
}

function Test-MsvcEnvReady {
    return (
        $env:VSCMD_ARG_TGT_ARCH -eq "x64" -and
        -not [string]::IsNullOrWhiteSpace($env:VCINSTALLDIR) -and
        $null -ne (Get-Command cl.exe -ErrorAction SilentlyContinue)
    )
}

function Get-PythonBuildInfo {
    param([string]$PythonExe)

    $code = @(
        "import sys, sysconfig",
        "print(sys.executable)",
        "print(sys.version.split()[0])",
        "print(sys.implementation.cache_tag.replace('cpython-', 'cp'))",
        "print(sysconfig.get_platform().replace('-', '_'))"
    ) -join "; "

    $output = & $PythonExe -c $code
    if ($LASTEXITCODE -ne 0 -or $output.Count -lt 4) {
        throw "Unable to query Python compatibility tag from '$PythonExe'."
    }

    return [PSCustomObject]@{
        Executable = $output[0]
        Version = $output[1]
        AbiTag = $output[2]
        PlatformTag = $output[3]
        WheelTag = "$($output[2])-$($output[2])-$($output[3])"
    }
}

function Import-VcVars64 {
    if (Test-MsvcEnvReady) {
        Write-Host "MSVC x64 environment already loaded."
        return
    }

    $programFiles = ${env:ProgramFiles}
    $candidates = @(
        "$programFiles\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "$programFiles\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "$programFiles\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "$programFiles\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )

    $vcvars = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $vcvars) {
        Write-Warning "vcvars64.bat was not found. Continuing with the current shell environment."
        return
    }

    Write-Host "Importing MSVC environment from: $vcvars"
    $vcvarsOutput = cmd /d /s /c "call `"$vcvars`" >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) {
        throw "vcvars64.bat failed. If this shell has been used for repeated builds, open a fresh PowerShell and retry."
    }

    $vcvarsOutput | ForEach-Object {
        $idx = $_.IndexOf("=")
        if ($idx -gt 0) {
            $name = $_.Substring(0, $idx)
            $value = $_.Substring($idx + 1)
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

$CudaPath = Resolve-CudaPath -RequestedPath $CudaPath `
    -Explicit ($PSBoundParameters.ContainsKey('CudaPath'))
Set-UniquePath
if (-not $NoVcVars) {
    Import-VcVars64
}

$env:CUDA_PATH = $CudaPath
$env:CUDA_HOME = $CudaPath
$env:CUDACXX = Join-Path $CudaPath "bin\nvcc.exe"
$env:CMAKE_GENERATOR = "Visual Studio 17 2022"
$env:CMAKE_GENERATOR_PLATFORM = "x64"
$env:CMAKE_GENERATOR_TOOLSET = "cuda=$CudaPath"
Set-UniquePath -Prepend @("$CudaPath\bin", "$CudaPath\extras\CUPTI\lib64")

$PythonInfo = Get-PythonBuildInfo -PythonExe $Python

$commonConfig = @(
    "-C", "cmake.define.BUILD_PYTHON=ON",
    "-C", "cmake.define.BUILD_GPUFL_EXAMPLE=OFF",
    "-C", "cmake.define.BUILD_TESTING=OFF",
    "-C", "cmake.define.PYBIND11_FINDPYTHON=ON",
    "-C", "cmake.define.GPUFL_ENABLE_NVIDIA=ON",
    "-C", "cmake.define.GPUFL_ENABLE_AMD=OFF",
    "-C", "cmake.define.CUDAToolkit_ROOT=$CudaPath",
    "-C", "cmake.define.CMAKE_CUDA_COMPILER=$env:CUDACXX"
)

Write-Host "GPUFlight build"
Write-Host "  mode:      $Mode"
Write-Host "  python:    $($PythonInfo.Executable)"
Write-Host "  py version: $($PythonInfo.Version)"
Write-Host "  wheel tag: $($PythonInfo.WheelTag)"
Write-Host "  cuda path: $CudaPath"

if ($Mode -eq "wheel") {
    New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
    & $Python -m pip wheel $RootDir -w $WheelDir --no-deps -v @commonConfig
    if ($LASTEXITCODE -ne 0) { throw "pip wheel failed" }
} elseif ($Mode -eq "trace") {
    # Native injection-mode tooling: the `gpufl` launcher (gpufl trace/upload/
    # version) + gpufl_inject.dll. These are NOT part of the Python wheel - a
    # plain CMake build into build-windows\. CUDAToolkit_ROOT is pinned to the
    # same $CudaPath used everywhere here, so the inject DLL links and the
    # copied cupti64*.dll come from ONE toolkit (avoids the version skew that
    # otherwise breaks the DLL load at injection time).
    $buildDir = Join-Path $RootDir "build-windows"
    $traceConfig = @(
        "-DGPUFL_ENABLE_NVIDIA=ON",
        "-DGPUFL_ENABLE_AMD=OFF",
        "-DBUILD_PYTHON=OFF",
        "-DBUILD_TESTING=OFF",
        "-DBUILD_GPUFL_EXAMPLE=OFF",
        "-DBUILD_GPUFL_LAUNCHER=ON",
        "-DBUILD_GPUFL_INJECT=ON",
        "-DCUDAToolkit_ROOT=$CudaPath",
        "-DCMAKE_CUDA_COMPILER=$env:CUDACXX"
    )
    # cmake + FetchContent (git) write normal progress to stderr (e.g.
    # "Cloning into 'zlib-src'..."). Under the script's ErrorActionPreference
    # = "Stop" those stderr lines become terminating errors and abort the
    # build, so relax it for the native cmake calls and gate on $LASTEXITCODE.
    $ErrorActionPreference = "Continue"
    # Uses the VS generator/platform/toolset from the env vars set above.
    cmake -S $RootDir -B $buildDir @traceConfig
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
    # Building gpufl_launcher also builds gpufl_inject (dependency), colocates
    # the DLL next to gpufl.exe, and copies cupti64*/nvperf_* beside it.
    cmake --build $buildDir --config Release --target gpufl_launcher -j
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }
    $ErrorActionPreference = "Stop"
    $exe = Join-Path $buildDir "daemon\launcher\Release\gpufl.exe"
    Write-Host ""
    Write-Host "Built native trace tooling:"
    Write-Host "  launcher:    $exe"
    Write-Host "  (gpufl_inject.dll + cupti64*/nvperf_* are colocated next to it)"
    Write-Host ""
    Write-Host "Run:  & `"$exe`" trace --passes=Trace -- <python.exe> <script.py>"
    Write-Host "      & `"$exe`" trace --passes=PcSampling -- <python.exe> <script.py>"
    Write-Host "      (PcSampling needs an elevated shell for stall-reason access;"
    Write-Host "       unprivileged runs report 'skipped / no_stall_reasons')"
} else {
    & $Python -m pip install $RootDir -v @commonConfig
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}
