param(
    [ValidateSet("install", "wheel")]
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
    param([string]$RequestedPath)

    $candidates = @(
        $RequestedPath,
        $env:CUDA_HOME,
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    foreach ($candidate in $candidates) {
        $nvcc = Join-Path $candidate "bin\nvcc.exe"
        if (Test-Path -LiteralPath $nvcc) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "CUDA nvcc was not found. Pass -CudaPath or set CUDA_PATH/CUDA_HOME."
}

function Import-VcVars64 {
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
    cmd /d /s /c "call `"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        $idx = $_.IndexOf("=")
        if ($idx -gt 0) {
            $name = $_.Substring(0, $idx)
            $value = $_.Substring($idx + 1)
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

$CudaPath = Resolve-CudaPath -RequestedPath $CudaPath
if (-not $NoVcVars) {
    Import-VcVars64
}

$env:CUDA_PATH = $CudaPath
$env:CUDA_HOME = $CudaPath
$env:CUDACXX = Join-Path $CudaPath "bin\nvcc.exe"
$env:CMAKE_GENERATOR = "Visual Studio 17 2022"
$env:CMAKE_GENERATOR_PLATFORM = "x64"
$env:CMAKE_GENERATOR_TOOLSET = "cuda=$CudaPath"
$env:Path = "$CudaPath\bin;$CudaPath\extras\CUPTI\lib64;$env:Path"

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
Write-Host "  python:    $Python"
Write-Host "  cuda path: $CudaPath"

if ($Mode -eq "wheel") {
    New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
    & $Python -m pip wheel $RootDir -w $WheelDir --no-deps -v @commonConfig
} else {
    & $Python -m pip install $RootDir -v @commonConfig
}
