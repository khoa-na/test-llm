# vLLM official image — pin v0.8.5 (CUDA 12.4).
#
# Khảo sát thực tế image labels (NVIDIA_REQUIRE_CUDA):
#   v0.21.0    → cuda>=12.9   (driver >= 575)
#   v0.13.0    → cuda>=12.9   (driver >= 575)
#   v0.11.0    → cuda>=12.8   (driver >= 570)
#   v0.10.2    → cuda>=12.8
#   v0.9.2     → cuda>=12.8
#   v0.8.5     → cuda>=12.4   (driver >= 550)  ✅ phù hợp RunPod
#   v0.7.0     → cuda>=12.1
#
# RunPod GPU node driver < 575 → cần image yêu cầu CUDA <= 12.6.
# v0.8.5 là bản cao nhất với CUDA 12.4, ra mắt 04/2025 — đã hỗ trợ Qwen3
# (release cùng tháng). Qwen3.5/Gemma 4 sẽ dùng trust_remote_code +
# transformers upgrade trong RUN bên dưới.
FROM vllm/vllm-openai:v0.8.5

# Bỏ entrypoint mặc định của vLLM (chạy OpenAI server) để dùng handler riêng
ENTRYPOINT []

WORKDIR /app

# Cài runpod SDK + hf_transfer
# Upgrade transformers để load được Qwen3.5 (03/2026), Gemma 4 (04/2026),
# DeepSeek-R1-0528 (05/2025) — các model release sau khi base image build.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade "transformers>=4.50"

# Copy handler
COPY handler.py /app/

CMD ["python3", "-u", "handler.py"]
