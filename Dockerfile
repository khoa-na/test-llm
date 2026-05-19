# vLLM official image — pin v0.21.0 với CUDA 12.9.
#
# Endpoint RunPod đã filter min CUDA = 12.9 nên image này được GPU node accept.
# Build nhanh (~2-3 phút) vì base image có sẵn full vLLM + torch + flash-attn.
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
