# vLLM official image — pin CUDA 12.4 cho RunPod GPU node có driver cũ.
#
# Lịch sử CUDA của vLLM official image:
#   - v0.21.0+         → CUDA 12.9 (cu129)            cần driver >= 555
#   - v0.14.0–v0.20.x  → CUDA 13.0 (cu130)            cần driver >= 580
#   - v0.13.0 trở xuống → CUDA 12.4 (cu124, mặc định)  cần driver >= 550
#
# RunPod nodes của bạn báo "cuda>=12.9 unsatisfied" → driver hỗ trợ tối đa
# CUDA 12.4 hoặc 12.6. Dùng v0.13.0 là lựa chọn ổn định nhất.
#
# Note: vLLM v0.13.0 (Dec 2025) chưa biết Qwen3.5 (Mar 2026), nhưng
# với trust_remote_code=True, transformers tự download modeling_qwen3.py
# từ HF repo → vẫn load được.
FROM vllm/vllm-openai:v0.13.0

# Bỏ entrypoint mặc định của vLLM (chạy OpenAI server) để dùng handler riêng
ENTRYPOINT []

WORKDIR /app

# Cài runpod SDK + hf_transfer (tăng tốc download)
# Update transformers lên bản mới hơn để hỗ trợ Qwen3.5 / Gemma 4 native
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade "transformers>=4.50"

# Copy handler
COPY handler.py /app/

CMD ["python3", "-u", "handler.py"]
