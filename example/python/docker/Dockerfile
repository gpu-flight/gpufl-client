# syntax=docker/dockerfile:1
# GPUFlight Jupyter playground.
#
# Installs the PUBLISHED gpufl wheel from PyPI — the same artifact end users
# get from `pip install gpufl`. As of 1.0.0 the Linux wheel bundles its own
# CUDA runtime (libcudart), CUPTI (libcupti) and OpenSSL, so there is NO
# source build and NO system CUDA toolkit needed. The only thing required
# from the host is the NVIDIA driver, injected at runtime by the NVIDIA
# Container Toolkit.
#
#   Build:  docker build -t gpufl-jupyter:1.1.0 .
#   Run:    docker run --gpus all -p 8888:8888 gpufl-jupyter:1.1.0
#
# A '-base' image is enough: we don't compile (so no '-devel'), and we don't
# need the runtime's CUDA libs either — numba-cuda[cu13] ships its own
# NVVM/NVRTC wheels and the gpufl wheel bundles cudart/cupti. '-base' still
# sets NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES so the GPU is
# visible. (If numba/gpufl ever need a system CUDA lib, bump back to
# '-runtime' — one-line change.)
FROM nvidia/cuda:13.1.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Point any process in this image at the system CA bundle. The gpufl wheel
# statically links its own OpenSSL, whose compile-time OPENSSLDIR was set
# in the manylinux build container and does NOT match Ubuntu's
# /etc/ssl/certs/ layout. Without these env vars, the embedded OpenSSL
# fails to find any trusted roots and gpufl::uploadLogs reports
# `httplib::Error=10` (SSLServerVerification). Both names are read by
# OpenSSL's X509_get_default_cert_{file,dir}_env() at runtime and override
# the baked-in OPENSSLDIR; harmless for any process that doesn't use them.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_DIR=/etc/ssl/certs

# Ubuntu 24.04 ships Python 3.12, matching the gpufl cp312 wheel.
#
# `ca-certificates` is required for the gpufl wheel's bundled OpenSSL to
# verify TLS server certs when uploading to https://api.gpuflight.com.
# Without it, gpufl::uploadLogs fails with `httplib::Error=10`
# (SSLServerVerification): the handshake succeeds but there are no trusted
# root CAs on disk to validate the server cert chain against. The
# `nvidia/cuda:*-base-*` images intentionally ship without ca-certificates,
# so we have to add it explicitly.
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# venv to avoid PEP 668 issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# NOTE: PyTorch is intentionally NOT installed here. It pulls ~5–7 GB of its
# own CUDA wheels (cublas/cudnn/nccl/…) that duplicate the base image and
# aren't needed for the Numba / pure-CUDA examples this image is built for.
# For a torch-integration playground, add back:
#   RUN --mount=type=cache,target=/root/.cache/pip \
#       pip install --retries 5 --timeout 120 torch --index-url https://download.pytorch.org/whl/cu130
# (and import torch BEFORE gpufl to avoid the CUPTI clash).

# gpufl from PyPI, pinned for a reproducible image (bump for new releases).
# numba-cuda[cu13] supplies the CUDA-13 NVVM/NVRTC compiler — plain `numba`
# ships an older CUDA-12 target that reports "no CUDA device" on CUDA 13.
ARG GPUFL_VERSION=1.1.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 120 "gpufl[analyzer,viz]==${GPUFL_VERSION}" "numba-cuda[cu13]"

# Jupyter for interactive sessions
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 120 jupyterlab

# Working directory for notebooks
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab — no token for easy local access
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
