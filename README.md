# test-llm — RunPod Serverless Worker

Custom vLLM serverless worker để chạy LLM cho chatbot thư ký (Qwen3-8B, Qwen3.5-9B, Gemma 4-E4B, DeepSeek-R1-Qwen3-8B). Deploy trực tiếp từ GitHub repo lên RunPod Serverless.

## Cấu trúc

```
.
├── handler.py          # Entry point — vLLM worker
├── Dockerfile          # Image build cho RunPod
├── requirements.txt    # runpod SDK
├── .dockerignore       # Exclude file khỏi build context
├── .env.example        # Worker config + client config
├── test_client.py      # Gọi endpoint test từ local
└── Chatbot_ThuKy_UseCases_EvalCriteria_v3.md   # Spec
```

## Deploy lên RunPod Serverless

### Cách 1: Deploy từ GitHub repo (recommend)

1. Push repo này lên GitHub
2. Vào https://www.runpod.io/console/serverless → **New Endpoint**
3. Chọn **"Import Git Repository"**
4. Authorize GitHub → chọn repo `khoa-na/test-llm`, branch `main`
5. RunPod tự build từ `Dockerfile`
6. Cấu hình:
   - **Container Disk**: 30 GB
   - **GPU Types**: tick nhiều loại (4090, A5000, A6000, L40S)
   - **Max Workers**: 3
   - **Idle Timeout**: 60s
   - **Execution Timeout**: 600s
   - **FlashBoot**: ON
7. **Environment Variables**:
   ```
   MODEL_NAME=Qwen/Qwen3.5-9B-Instruct
   MAX_MODEL_LEN=32768
   GPU_MEMORY_UTILIZATION=0.92
   TRUST_REMOTE_CODE=true
   ```
8. Deploy → đợi build ~10-15 phút lần đầu

### Cách 2: Build local + push Docker image

```bash
docker build -t <dockerhub-user>/test-llm-worker:qwen3.5-9b .
docker push <dockerhub-user>/test-llm-worker:qwen3.5-9b
```

Rồi tạo endpoint với image này.

## Test từ máy local

```bash
# 1. Cài deps
pip install requests python-dotenv

# 2. Set .env
cp .env.example .env
# Sửa: RUNPOD_API_KEY=rpa_xxx, RUNPOD_ENDPOINT_ID=xxx

# 3. Gọi thử
python test_client.py "Tôi có lịch họp gì hôm nay?"
```

## Test handler local (không cần GPU?)

Handler import vLLM → cần GPU CUDA để load model. Không test local được trừ khi có GPU.

→ Cứ deploy lên RunPod test trực tiếp.

## Deploy nhiều model

Mỗi model = 1 endpoint riêng, deploy cùng repo nhưng đổi `MODEL_NAME` trong env:

| Endpoint | MODEL_NAME | GPU recommend |
|---|---|---|
| qwen3.5-9b (default) | `Qwen/Qwen3.5-9B-Instruct` | 24GB |
| qwen3-8b | `Qwen/Qwen3-8B` | 24GB |
| gemma4-e4b | `google/gemma-4-e4b-it` | 16GB (cần HF_TOKEN) |
| deepseek-r1-qwen3-8b | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 24GB |

## API

### Endpoint URL
```
https://api.runpod.ai/v2/<endpoint-id>/run         # async
https://api.runpod.ai/v2/<endpoint-id>/runsync     # sync (timeout 90s)
https://api.runpod.ai/v2/<endpoint-id>/status/<job-id>
```

### Request body
```json
{
  "input": {
    "messages": [{"role": "user", "content": "..."}],
    "sampling_params": {
      "temperature": 0.7,
      "top_p": 0.95,
      "max_tokens": 1024
    },
    "thinking_mode": false
  }
}
```

### Response (status COMPLETED)
```json
{
  "output": {
    "text": "...",
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "finish_reason": "stop"
  }
}
```
