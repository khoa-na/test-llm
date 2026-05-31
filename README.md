# test-llm — Chatbot Thư Ký (Modal Serverless)

Serverless **vLLM** worker chạy trên **Modal** cho chatbot thư ký doanh nghiệp tiếng Việt. Ngoài phần serving, repo còn có:

- **Hybrid RAG** server-side — kết hợp truy xuất *structured* (lịch / task / email qua SQLite) và *semantic* (tài liệu `.md` qua embedding **bge-m3**), định tuyến intent bằng **embedding router**.
- **Tool-calling `python_exec`** — model tự gọi để tính toán số học / ngày tháng *deterministic*, chạy cô lập trong **Modal Sandbox** (không mạng).
- **Pipeline đánh giá LLM-as-Judge** (Gemini) trên 3 bộ test + báo cáo so sánh nhiều model.

## Cấu trúc

```
modal_app.py            # LLMServer (vLLM) + RAG augment + agentic tool loop + HTTP endpoint
retrieval.py            # Hybrid retriever: SQLite structured + bge-m3 semantic + embedding router
rag_corpus/             # Seed data: calendar.json, tasks.json, emails.json, docs/*.md
modal_client.py         # Gọi endpoint từ local (SDK / HTTP)
eval/
  run_eval_judge.py     # Orchestrator: generate -> judge -> markdown report
  config|target|judge|criteria|report.py
  test_cases_{chat,rag,rag_live}.yaml
  eval_retrieval.py     # đo recall@k của retrieval (tách khỏi generation)
  rag_diagnose.py       # tune ngưỡng embedding router
  comparisons/          # báo cáo so sánh model (3-way, 2-way)
.env.example            # config mẫu
```

## Model đã test

Xem báo cáo đầy đủ trong `eval/comparisons/`.

| Model | `MODEL_NAME` | Ghi chú |
|---|---|---|
| Qwen3.5-9B (default) | `Qwen/Qwen3.5-9B` | toàn diện, tool-calling tốt |
| DeepSeek-R1-0528-Qwen3-8B | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | reasoning, hơi verbose |
| Gemma 4-E4B | `google/gemma-4-e4b-it` | gated (cần HF token); chạy text-only |

Tổng hợp 97 case (chat 63 + rag_live 34), judge `gemini-3.1-flash-lite`:
**DeepSeek 94.7 ≈ Gemma 94.2 > Qwen 92.1** (overall). Cả 3 đều yếu ở trích đa-dữ-kiện từ biên bản (`RL31`).

## Setup (1 lần)

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt    # Windows  (Linux/macOS: .venv/bin/pip)
.venv\Scripts\modal token new                    # đăng nhập Modal (mở browser)
```

Copy `.env.example` → `.env` rồi điền:
- `GEMINI_API_KEY` — cho LLM-as-Judge (lấy tại https://aistudio.google.com/apikey).
- `MODEL_NAME`, `MODAL_GPU` (mặc định `L40S`), `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`.

Model **gated** (Gemma): tạo Modal secret tên `huggingface` chứa `HF_TOKEN` (tài khoản đã accept license trên Hugging Face).

## Deploy

```bash
.venv\Scripts\modal deploy modal_app.py
```

Output có **Endpoint URL** (HTTP) → copy vào `.env` → `MODAL_ENDPOINT_URL=...` nếu muốn gọi qua HTTP.

## Test nhanh

```bash
# Local entrypoint (spawn container 1 lần, chạy, tự tắt)
.venv\Scripts\modal run modal_app.py --question "lịch hôm nay có gì"

# Qua client (sau khi deploy)
.venv\Scripts\python modal_client.py "2+2=?"          # SDK
.venv\Scripts\python modal_client.py "2+2=?" http     # HTTP (cần MODAL_ENDPOINT_URL)
```

## Đánh giá (LLM-as-Judge)

```bash
.venv\Scripts\python eval/run_eval_judge.py --set chat        # 63 case hội thoại (data inline)
.venv\Scripts\python eval/run_eval_judge.py --set rag_live    # 34 case RAG (retrieval THẬT server-side)
.venv\Scripts\python eval/eval_retrieval.py                   # recall@k của retrieval (rẻ, không generate)
```

**So sánh model khác:** sửa `MODEL_NAME` trong `.env` → `modal deploy modal_app.py` → chạy lại eval. Report ghi ra `eval/results/judge/` (gắn slug model). Có thể chạy `generate` và `judge` riêng (xem `--help`).

Unit test retrieval (không cần GPU, dùng FakeEmbedder):

```bash
.venv\Scripts\python test_retrieval.py
```

## Config chính (`.env`)

| Nhóm | Biến |
|---|---|
| Serving | `MODEL_NAME`, `MODAL_GPU`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `DTYPE`, `MAX_NUM_SEQS` |
| Tools | `USE_TOOLS` |
| RAG | `USE_RETRIEVAL`, `EMBED_MODEL` (bge-m3), `RAG_TOP_K`, `RAG_MIN_SCORE`, `RAG_ROUTE_MIN_SCORE`, `RAG_REFERENCE_DATE` |
| Judge / Eval | `JUDGE_MODEL`, `JUDGE_TEMPERATURE`, `JUDGE_SEED`, `JUDGE_RPM`, `EVAL_MODE`, `EVAL_TEMPERATURE`, `EVAL_MAX_TOKENS` |

## Cost (Modal, ước tính)

GPU tính theo giây khi active; `scaledown_window=5` → idle ~5s là tắt → $0. Cold start subsequent (model đã cache trong Volume) ~10-20s. Mỗi request inference vài giây.

## Tham khảo

- Modal docs: https://modal.com/docs
- vLLM trên Modal: https://modal.com/docs/examples/vllm_inference
