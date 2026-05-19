# Custom build: FROM CUDA base → install Python + vLLM qua pip.
#
# Vì sao build thay vì dùng vllm/vllm-openai pre-built:
#   - Kiểm soát CUDA version chính xác (cu129)
#   - Có thể nâng/hạ vLLM, transformers, torch độc lập
#   - Image gọn hơn (chỉ deps cần thiết)
#
# Yêu cầu RunPod: driver hỗ trợ CUDA 12.9 (đã filter trong endpoint config).
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ───────────────────────────────────────────────
# Python 3.12 + tools
# ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ───────────────────────────────────────────────
# vLLM stack
#   - vllm tự bundle torch tương thích CUDA
#   - transformers mới để hỗ trợ Qwen3.5, Gemma 4, DeepSeek-R1
# ───────────────────────────────────────────────
RUN pip install --upgrade pip wheel setuptools && \
    pip install \
        "vllm>=0.21.0" \
        "transformers>=4.55" \
        "runpod>=1.7.0" \
        "hf_transfer>=0.1.8"

WORKDIR /app
COPY handler.py /app/

CMD ["python", "-u", "handler.py"]
