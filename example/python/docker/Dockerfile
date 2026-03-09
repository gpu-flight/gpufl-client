FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (Ubuntu 24.04 ships Python 3.12)
# NOTE: cmake/ninja come from pip (build-system.requires needs >=3.31, apt has 3.28)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create venv to avoid PEP 668 issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 13.1 support
RUN pip install torch --index-url https://download.pytorch.org/whl/cu130

# Build gpufl from source so it picks up NVML from the CUDA devel image
ARG GPUFL_VERSION=main
RUN git clone --depth 1 --branch ${GPUFL_VERSION} https://github.com/gpu-flight/gpufl-client.git /tmp/gpufl-client \
    && CMAKE_ARGS="-DBUILD_TESTING=OFF" pip install -v "/tmp/gpufl-client[analyzer,viz]" \
    && rm -rf /tmp/gpufl-client

# Install Jupyter
RUN pip install jupyterlab

# Working directory for notebooks
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab — no token for easy local access
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
