# vLLM official image — pin CUDA 12.9
#
# Lưu ý:
#   - `:latest` HIỆN TẠI vẫn build với CUDA 13.0 → RunPod node driver < 580 sẽ reject.
#   - v0.20.1+ chỉ phát hành dạng cu129 (CUDA 12.9), driver >= 555 chạy được.
#   - Nếu RunPod node vẫn báo lỗi cuda mismatch, fallback xuống cu124 bằng cách
#     đổi FROM thành: vllm/vllm-openai:v0.20.0-cu129-ubuntu2404
#     Hoặc tự build từ NVIDIA base: FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM vllm/vllm-openai:v0.21.0-cu129-ubuntu2404

# Bỏ entrypoint mặc định của vLLM (chạy OpenAI server) để dùng handler riêng
ENTRYPOINT []

WORKDIR /app

# Cài runpod SDK
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py /app/

# Optional: preload model vào image (bỏ comment nếu muốn cold start nhanh hơn)
# ARG MODEL_NAME=Qwen/Qwen3.5-9B-Instruct
# RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_NAME}')"

CMD ["python3", "-u", "handler.py"]
