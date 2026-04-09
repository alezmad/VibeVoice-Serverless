FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/runpod-volume/vibevoice/torch_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git ca-certificates curl build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /workspace/vibevoice

COPY requirements.txt /workspace/vibevoice/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY bootstrap.sh /workspace/vibevoice/bootstrap.sh
COPY handler.py /workspace/vibevoice/handler.py
COPY inference.py /workspace/vibevoice/inference.py
COPY config.py /workspace/vibevoice/config.py

CMD ["bash", "/workspace/vibevoice/bootstrap.sh"]
