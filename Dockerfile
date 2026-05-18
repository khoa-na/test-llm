# vLLM official image — pin CUDA 12.9 vì RunPod GPU node chưa hỗ trợ CUDA 13
# `:latest` hiện build với CUDA 13.0 → nvidia-container-cli reject trên host driver < 580
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
