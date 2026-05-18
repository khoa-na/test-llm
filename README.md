# test-llm — Modal Serverless

Serverless vLLM worker chạy trên **Modal** cho chatbot thư ký (Qwen3.5-9B-Instruct mặc định). Khác với RunPod, Modal không cần Docker — chỉ 1 file Python + decorator.

## Cấu trúc

```
.
├── modal_app.py        # Main: image build + LLMServer class + HTTP endpoint
├── modal_client.py     # Gọi endpoint từ local (SDK hoặc HTTP)
├── requirements.txt    # modal, requests, python-dotenv (chỉ cho client)
├── .env.example        # MODEL_NAME, GPU_TYPE, MODAL_ENDPOINT_URL
└── Chatbot_ThuKy_*.md  # Spec doc
```

## Setup (1 lần)

```bash
# Cài Modal CLI vào venv
.venv/bin/pip install modal

# Auth — mở browser đăng nhập
.venv/bin/modal token new
```

Đăng ký tài khoản tại https://modal.com — có **$30 free credit/tháng**, đủ test 100+ batch.

## Deploy

```bash
.venv/bin/modal deploy modal_app.py
```

Output cuối sẽ có 2 thứ:
- **App URL**: dashboard quản lý
- **Endpoint URL** (cho HTTP): copy vào `.env` → `MODAL_ENDPOINT_URL=...`

## Test

### Cách 1: Local entrypoint (không cần deploy)
```bash
.venv/bin/modal run modal_app.py
.venv/bin/modal run modal_app.py --question "Tôi có lịch họp gì hôm nay?"
```
Modal spawn container 1 lần, chạy, tự tắt.

### Cách 2: Qua Modal SDK (sau khi deploy)
```bash
.venv/bin/python modal_client.py "2+2=?"
```

### Cách 3: HTTP (đã deploy + set MODAL_ENDPOINT_URL trong .env)
```bash
.venv/bin/python modal_client.py "2+2=?" http
```

## Đổi model

Sửa `modal_app.py` line `MODEL_NAME`, deploy lại. Hoặc tạo app khác với app name khác:

| Model | MODEL_NAME | GPU |
|---|---|---|
| **Qwen3.5-9B (default)** | `Qwen/Qwen3.5-9B-Instruct` | A10G 24GB |
| Qwen3-8B | `Qwen/Qwen3-8B` | A10G 24GB |
| Gemma 4-E4B | `google/gemma-4-e4b-it` | A10G 24GB (cần HF_TOKEN) |
| DeepSeek-R1-Qwen3-8B | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | A10G 24GB |

## Cost ước tính (A10G)

- ~$0.000306/s khi active
- Cold start (1 lần đầu): ~30-60s download model → ~$0.02
- Cold start subsequent (model cached): ~10-20s → ~$0.005
- Mỗi request inference: 3-10s → ~$0.001-0.003
- Idle (scaledown 120s sau request cuối): $0

Test toàn bộ 11 use case × 4 model × 2 thinking mode: **~$1-2**.

## Workflow dev

```bash
# 1. Sửa code
vi modal_app.py

# 2. Deploy lại (Modal tự rebuild diff image)
.venv/bin/modal deploy modal_app.py

# 3. Test
.venv/bin/python modal_client.py "câu hỏi mới"
```

## Tắt/xóa endpoint

```bash
.venv/bin/modal app stop test-llm-chatbot-thuky
# hoặc xóa hẳn:
.venv/bin/modal app remove test-llm-chatbot-thuky
```

## Tham khảo

- Modal docs: https://modal.com/docs
- vLLM example từ Modal: https://modal.com/docs/examples/vllm_inference
