FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11
ARG PYTHONPATH=""
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace:${PYTHONPATH}
WORKDIR /workspace

# ---------------------------------------------------------------------
# 1. Install base system dependencies and build tools
# ---------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-distutils python3-pip python${PYTHON_VERSION}-dev \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx libvulkan-dev \
      zip unzip wget git-lfs cmake ffmpeg netcat dnsutils vim less sudo htop man tmux tzdata \
      ninja-build pkg-config iproute2\
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# 2. Build and install CycloneDDS from source (releases/0.10.x branch)
# ---------------------------------------------------------------------
RUN set -eux; \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /root/cyclonedds; \
    cd /root/cyclonedds; \
    mkdir -p build install && cd build; \
    cmake .. -DCMAKE_INSTALL_PREFIX=/root/cyclonedds/install; \
    cmake --build . --target install; \
    ldconfig

# Ensure CycloneDDS install is readable by all users (fix permission denied for non-root)
RUN chmod -R a+rX /root/cyclonedds/install

ENV CYCLONEDDS_HOME=/root/cyclonedds/install
ENV CMAKE_PREFIX_PATH=/root/cyclonedds/install
ENV PIP_DEFAULT_TIMEOUT=300
ENV NINJA_JOBS=3

# ---------------------------------------------------------------------
# 3. Create a non-root user
# ---------------------------------------------------------------------
RUN useradd -m -s /bin/bash deepansh && echo "deepansh ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/deepansh

# ---------------------------------------------------------------------
# 4. Copy minimal dependency specs
# ---------------------------------------------------------------------
COPY pyproject.toml uv.lock requirements.txt* /workspace/

# ---------------------------------------------------------------------
# 5. Create venv and bootstrap packaging tools (use the target python)
# ---------------------------------------------------------------------
RUN python${PYTHON_VERSION} -m venv /workspace/.venv \
 && /workspace/.venv/bin/python -m pip install --upgrade pip setuptools wheel

ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH

# ---------------------------------------------------------------------
# 6. Install torch and core dependencies in stages
# ---------------------------------------------------------------------
# Install torch first (without torchvision to reduce complexity)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    pip install --upgrade pip setuptools wheel; \
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Install torchvision and triton separately
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128; \
    pip install pytorch-triton

# Install requirements in multiple small batches to avoid resolution depth issues
# Batch 1: Core scientific computing packages (no complex dependencies)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    pip install --no-cache-dir \
        numpy==2.3.4 scipy==1.16.2 pandas==2.3.3 matplotlib==3.10.7 \
        pillow==12.0.0 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86

# Batch 2: ML/AI core libraries (install with specific torch already present)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    pip install --no-cache-dir \
        transformers==4.51.3 tokenizers==0.21.4 diffusers==0.35.2 \
        safetensors==0.6.2 huggingface-hub==0.35.3 timm==1.0.20

# Batch 3: Kornia and related packages (these depend on torch)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    pip install --no-cache-dir kornia==0.8.1 kornia-rs==0.1.9

# Batch 4: Acceleration and distributed training
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    pip install --no-cache-dir "accelerate>=0.26.0" peft==0.17.1

# Batch 5: Rest of the requirements (filter out already installed packages)
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    if [ -f /workspace/requirements.txt ]; then \
        pip install --no-cache-dir -r /workspace/requirements.txt; \
    fi

# ---------------------------------------------------------------------
# 7. Copy local editable packages and rest of repo
# ---------------------------------------------------------------------
COPY --chown=deepansh:deepansh flash-attention /workspace/flash-attention
COPY --chown=deepansh:deepansh unitree_sdk2_python /workspace/unitree_sdk2_python
COPY --chown=deepansh:deepansh . /workspace

# Some builds run as root during RUN; avoid git "dubious ownership" warnings by making
# local package directories temporarily root-owned for root-run installs.
RUN chown -R root:root /workspace/flash-attention /workspace/unitree_sdk2_python

# ---------------------------------------------------------------------
# 8. Install local editable packages using the venv and no build isolation
# ---------------------------------------------------------------------
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    export PIP_NO_BUILD_ISOLATION=1; \
    pip install --upgrade pip setuptools wheel; \
    pip uninstall -y transformer-engine; \
    MAX_JOBS=3 pip install --no-build-isolation -e /workspace/flash-attention;
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    set -eux; \
    pip install --no-build-isolation -e /workspace/unitree_sdk2_python; \
    pip install --no-build-isolation -e /workspace --no-deps; \
    pip install "accelerate>=0.26.0"; \
    pip install logging_mp

COPY gr00t /workspace/gr00t
COPY Makefile /workspace/Makefile
RUN pip install -e .

# ---------------------------------------------------------------------
# 9. Restore ownership of the repo to the non-root runtime user
# ---------------------------------------------------------------------
RUN chown -R deepansh:deepansh /workspace

# ---------------------------------------------------------------------
# 10. Metadata and runtime setup
# ---------------------------------------------------------------------
LABEL org.opencontainers.image.title="Isaac-GR00T" \
      org.opencontainers.image.version="0.1" \
      org.opencontainers.image.authors="NVIDIA"

USER deepansh
CMD ["bash"]
