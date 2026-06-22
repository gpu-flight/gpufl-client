#!/usr/bin/env bash
set -euo pipefail

gpufl_install_script_dir() {
    local source_path="${BASH_SOURCE[0]}"
    local script_dir
    script_dir="$(cd "$(dirname "${source_path}")" && pwd)"
    printf '%s\n' "${script_dir}"
}

gpufl_install_repo_root() {
    local script_dir
    script_dir="$(gpufl_install_script_dir)"
    cd "${script_dir}/../.." && pwd
}

gpufl_install_info() {
    printf '[gpufl-install] %s\n' "$*"
}

gpufl_install_warn() {
    printf '[gpufl-install:warning] %s\n' "$*" >&2
}

gpufl_install_die() {
    printf '[gpufl-install:error] %s\n' "$*" >&2
    exit 1
}

gpufl_install_usage() {
    cat <<'EOF'
Usage:
  scripts/install/install-linux.sh --backend nvidia [options]
  scripts/install/install-linux.sh --backend amd [options]
  scripts/install/install-linux-nvidia.sh [options]
  scripts/install/install-linux-amd.sh [options]

Options:
  --backend nvidia|amd        Backend to install. Required for install-linux.sh.
  --prefix PATH              Install prefix. Default: $GPUFL_INSTALL_PREFIX or ~/.local.
  --build-dir PATH           Build directory. Default: build/install-linux-<backend>.
  --config TYPE              CMake build type. Default: $GPUFL_BUILD_TYPE or Release.
  --cuda-root PATH           CUDA toolkit root for NVIDIA installs.
  --rocm-root PATH           ROCm root for AMD installs. Default: $ROCM_PATH or /opt/rocm.
  --with-python              Build/install Python bindings too.
  --with-tests               Build tests as part of the install build.
  --with-examples            Build examples as part of the install build.
  --clean                    Remove the selected build directory before configuring.
  --dry-run                  Print commands without executing them.
  --skip-verify              Skip post-install checks.
  -h, --help                 Show this help.

NVIDIA install:
  Installs gpufl, gpufl-monitor, and libgpufl_inject.so.

AMD install:
  Installs gpufl-monitor. The injection-mode gpufl trace launcher is NVIDIA-only
  in the current Linux implementation.
EOF
}

gpufl_install_bool() {
    if [[ "${1}" == "1" ]]; then
        printf 'ON'
    else
        printf 'OFF'
    fi
}

gpufl_install_print_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

gpufl_install_run() {
    gpufl_install_print_cmd "$@"
    if [[ "${GPUFL_INSTALL_DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

gpufl_install_find_cuda_root() {
    local requested="${1:-}"
    local candidate
    local candidates=()

    if [[ -n "${requested}" ]]; then
        candidates+=("${requested}")
    fi
    if [[ -n "${CUDA_ROOT:-}" ]]; then
        candidates+=("${CUDA_ROOT}")
    fi
    if [[ -n "${CUDA_HOME:-}" ]]; then
        candidates+=("${CUDA_HOME}")
    fi
    if [[ -n "${CUDA_PATH:-}" ]]; then
        candidates+=("${CUDA_PATH}")
    fi

    candidates+=(
        /usr/local/cuda
        /usr/local/cuda-13.2
        /usr/local/cuda-13.1
        /usr/local/cuda-13.0
        /usr/local/cuda-12.8
        /usr/local/cuda-12.6
        /usr/local/cuda-12.4
    )

    for candidate in "${candidates[@]}"; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            cd "${candidate}" && pwd
            return 0
        fi
    done

    return 1
}

gpufl_install_reject_backend_arg() {
    local arg
    for arg in "$@"; do
        case "${arg}" in
            --backend|--backend=*)
                gpufl_install_die "backend is fixed by this wrapper; use install-linux.sh if you need --backend"
                ;;
        esac
    done
}

gpufl_install_verify_nvidia() {
    local prefix="$1"
    local gpufl_bin="${prefix}/bin/gpufl"
    local inject_lib="${prefix}/lib/libgpufl_inject.so"

    [[ -x "${gpufl_bin}" ]] || gpufl_install_die "expected executable not found: ${gpufl_bin}"
    [[ -f "${inject_lib}" ]] || gpufl_install_die "expected injection library not found: ${inject_lib}"

    if ! "${gpufl_bin}" version >/dev/null 2>&1; then
        gpufl_install_warn "'${gpufl_bin} version' failed; binary was installed but should be checked manually"
    fi

    if ! "${gpufl_bin}" trace --help >/dev/null 2>&1; then
        gpufl_install_warn "'${gpufl_bin} trace --help' failed; injection mode should be checked manually"
    fi

    if command -v ldd >/dev/null 2>&1; then
        if ldd "${inject_lib}" 2>/dev/null | grep -q "not found"; then
            gpufl_install_warn "ldd reports missing dependencies for ${inject_lib}"
            ldd "${inject_lib}" || true
        fi
    fi
}

gpufl_install_verify_amd() {
    local prefix="$1"
    local monitor_bin="${prefix}/bin/gpufl-monitor"

    [[ -x "${monitor_bin}" ]] || gpufl_install_die "expected executable not found: ${monitor_bin}"
}

gpufl_install_main() {
    local backend=""
    local prefix="${GPUFL_INSTALL_PREFIX:-${HOME}/.local}"
    local build_dir=""
    local config="${GPUFL_BUILD_TYPE:-Release}"
    local cuda_root="${CUDA_ROOT:-${CUDA_HOME:-${CUDA_PATH:-}}}"
    local rocm_root="${ROCM_PATH:-/opt/rocm}"
    local with_python=0
    local with_tests=0
    local with_examples=0
    local clean=0
    local skip_verify=0
    local generator="${GPUFL_CMAKE_GENERATOR:-}"
    local repo_root
    local arg

    while [[ $# -gt 0 ]]; do
        arg="$1"
        case "${arg}" in
            --backend)
                shift
                [[ $# -gt 0 ]] || gpufl_install_die "--backend requires a value"
                backend="$1"
                ;;
            --backend=*)
                backend="${arg#--backend=}"
                ;;
            --prefix)
                shift
                [[ $# -gt 0 ]] || gpufl_install_die "--prefix requires a value"
                prefix="$1"
                ;;
            --prefix=*)
                prefix="${arg#--prefix=}"
                ;;
            --build-dir)
                shift
                [[ $# -gt 0 ]] || gpufl_install_die "--build-dir requires a value"
                build_dir="$1"
                ;;
            --build-dir=*)
                build_dir="${arg#--build-dir=}"
                ;;
            --config)
                shift
                [[ $# -gt 0 ]] || gpufl_install_die "--config requires a value"
                config="$1"
                ;;
            --config=*)
                config="${arg#--config=}"
                ;;
            --cuda-root)
                shift
                [[ $# -gt 0 ]] || gpufl_install_die "--cuda-root requires a value"
                cuda_root="$1"
                ;;
            --cuda-root=*)
                cuda_root="${arg#--cuda-root=}"
                ;;
            --rocm-root)
                shift
                [[ $# -gt 0 ]] || gpufl_install_die "--rocm-root requires a value"
                rocm_root="$1"
                ;;
            --rocm-root=*)
                rocm_root="${arg#--rocm-root=}"
                ;;
            --with-python)
                with_python=1
                ;;
            --with-tests)
                with_tests=1
                ;;
            --with-examples)
                with_examples=1
                ;;
            --clean)
                clean=1
                ;;
            --dry-run)
                GPUFL_INSTALL_DRY_RUN=1
                export GPUFL_INSTALL_DRY_RUN
                ;;
            --skip-verify)
                skip_verify=1
                ;;
            -h|--help)
                gpufl_install_usage
                return 0
                ;;
            *)
                gpufl_install_usage >&2
                gpufl_install_die "unknown option: ${arg}"
                ;;
        esac
        shift
    done

    case "${backend}" in
        nvidia|amd)
            ;;
        "")
            gpufl_install_usage >&2
            gpufl_install_die "--backend is required"
            ;;
        *)
            gpufl_install_die "unsupported backend '${backend}', expected nvidia or amd"
            ;;
    esac

    repo_root="$(gpufl_install_repo_root)"
    if [[ -z "${build_dir}" ]]; then
        build_dir="${repo_root}/build/install-linux-${backend}"
    fi

    if [[ -z "${generator}" ]] && command -v ninja >/dev/null 2>&1; then
        generator="Ninja"
    fi

    gpufl_install_info "repo: ${repo_root}"
    gpufl_install_info "backend: ${backend}"
    gpufl_install_info "prefix: ${prefix}"
    gpufl_install_info "build-dir: ${build_dir}"
    gpufl_install_info "config: ${config}"

    if [[ "${backend}" == "nvidia" ]]; then
        if ! cuda_root="$(gpufl_install_find_cuda_root "${cuda_root}")"; then
            gpufl_install_die "CUDA toolkit not found. Pass --cuda-root or set CUDA_ROOT/CUDA_HOME/CUDA_PATH."
        fi
        export CUDA_ROOT="${cuda_root}"
        export CUDA_HOME="${cuda_root}"
        export CUDA_PATH="${cuda_root}"
        export CUDACXX="${cuda_root}/bin/nvcc"
        export PATH="${cuda_root}/bin:${PATH}"
        export LD_LIBRARY_PATH="${cuda_root}/lib64:${cuda_root}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
        gpufl_install_info "CUDA: ${cuda_root}"
    else
        if [[ -d "${rocm_root}" ]]; then
            export ROCM_PATH="${rocm_root}"
            export PATH="${rocm_root}/bin:${PATH}"
            export LD_LIBRARY_PATH="${rocm_root}/lib:${LD_LIBRARY_PATH:-}"
            gpufl_install_info "ROCm: ${rocm_root}"
        else
            gpufl_install_warn "ROCm root not found at ${rocm_root}; CMake will configure with whatever ROCm components are discoverable"
        fi
    fi

    if [[ "${clean}" == "1" ]]; then
        gpufl_install_run cmake -E rm -rf "${build_dir}"
    fi

    local cmake_args=(
        -S "${repo_root}"
        -B "${build_dir}"
        -DCMAKE_BUILD_TYPE="${config}"
        -DCMAKE_INSTALL_PREFIX="${prefix}"
        -DCMAKE_INSTALL_BINDIR=bin
        -DCMAKE_INSTALL_LIBDIR=lib
        -DBUILD_GPUFL_MONITOR=ON
        -DBUILD_GPUFL_EXAMPLE="$(gpufl_install_bool "${with_examples}")"
        -DBUILD_TESTING="$(gpufl_install_bool "${with_tests}")"
        -DBUILD_PYTHON="$(gpufl_install_bool "${with_python}")"
    )

    if [[ -n "${generator}" ]]; then
        cmake_args=(-G "${generator}" "${cmake_args[@]}")
    fi

    if [[ "${backend}" == "nvidia" ]]; then
        cmake_args+=(
            -DGPUFL_ENABLE_NVIDIA=ON
            -DGPUFL_ENABLE_AMD=OFF
            -DBUILD_GPUFL_LAUNCHER=ON
            -DBUILD_GPUFL_INJECT=ON
            -DCUDAToolkit_ROOT="${cuda_root}"
            -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc"
        )
    else
        cmake_args+=(
            -DGPUFL_ENABLE_NVIDIA=OFF
            -DGPUFL_ENABLE_AMD=ON
            -DGPUFL_ENABLE_AMD_ROCPROFILER=ON
            -DBUILD_GPUFL_LAUNCHER=OFF
            -DBUILD_GPUFL_INJECT=OFF
        )
        if [[ -d "${rocm_root}" ]]; then
            cmake_args+=(-DCMAKE_PREFIX_PATH="${rocm_root}${CMAKE_PREFIX_PATH:+;${CMAKE_PREFIX_PATH}}")
        fi
    fi

    gpufl_install_run cmake "${cmake_args[@]}"

    local build_args=(--build "${build_dir}" --config "${config}")
    if [[ "${with_python}" == "1" || "${with_tests}" == "1" || "${with_examples}" == "1" ]]; then
        gpufl_install_run cmake "${build_args[@]}"
    elif [[ "${backend}" == "nvidia" ]]; then
        gpufl_install_run cmake "${build_args[@]}" --target gpufl_launcher gpufl_inject gpufl-monitor
    else
        gpufl_install_run cmake "${build_args[@]}" --target gpufl-monitor
    fi

    gpufl_install_run cmake --install "${build_dir}" --config "${config}"

    if [[ "${GPUFL_INSTALL_DRY_RUN:-0}" != "1" && "${skip_verify}" != "1" ]]; then
        if [[ "${backend}" == "nvidia" ]]; then
            gpufl_install_verify_nvidia "${prefix}"
            gpufl_install_info "installed NVIDIA gpufl CLI: ${prefix}/bin/gpufl"
            gpufl_install_info "installed injection library: ${prefix}/lib/libgpufl_inject.so"
        else
            gpufl_install_verify_amd "${prefix}"
            gpufl_install_info "installed AMD monitor binary: ${prefix}/bin/gpufl-monitor"
        fi
    fi

    gpufl_install_info "done"
    gpufl_install_info "add this to PATH if needed: export PATH=\"${prefix}/bin:\$PATH\""
}
