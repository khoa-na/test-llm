# vLLM official image — pin v0.21.0 với CUDA 12.9.
#
# Image label NVIDIA_REQUIRE_CUDA=cuda>=12.9 → cần driver >= 555 (CUDA 12.9 support).
# Đa số GPU node RunPod đều có driver >= 555, chỉ một số node cũ < 555 sẽ fail.
# Nếu node hiện tại fail, đổi GPU type khác trong cấu hình endpoint.
FROM vllm/vllm-openai:v0.21.0-cu129-ubuntu2404

# Bỏ entrypoint mặc định của vLLM (chạy OpenAI server) để dùng handler riêng
ENTRYPOINT []

WORKDIR /app

# Cài runpod SDK + hf_transfer (tăng tốc download HF model)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py /app/

CMD ["python3", "-u", "handler.py"]
