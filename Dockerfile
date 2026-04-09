FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/workspace/torch_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install Python 3.12 from deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git ca-certificates curl build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /workspace/vibevoice

# Install PyTorch with CUDA 12.1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
COPY requirements.txt /workspace/vibevoice/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install VibeVoice
RUN pip install git+https://github.com/vibevoice-community/VibeVoice.git

# Install runpod
RUN pip install runpod>=1.6.0

# Pre-download model at build time
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('vibevoice/VibeVoice-7B')"

# Copy handler files
COPY handler.py /workspace/vibevoice/handler.py
COPY inference.py /workspace/vibevoice/inference.py
COPY config.py /workspace/vibevoice/config.py



COPY startup.py /workspace/vibevoice/startup.py

CMD ["python3", "/workspace/vibevoice/startup.py"]
