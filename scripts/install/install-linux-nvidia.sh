#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

gpufl_install_reject_backend_arg "$@"
gpufl_install_main --backend nvidia "$@"
