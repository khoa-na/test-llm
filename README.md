# Vietnamese AI Secretary — LLM on Modal

A serverless **vLLM** worker on **Modal** powering a Vietnamese enterprise secretary chatbot. Beyond serving, the repo ships:

- **Server-side hybrid RAG** — combines *structured* retrieval (calendar / tasks / emails via in-memory SQLite) with *semantic* retrieval (`.md` documents via **bge-m3** embeddings), with intent routing done by an **embedding router**.
- **`python_exec` tool-calling** — the model calls it to run deterministic arithmetic / date math, executed in an isolated **Modal Sandbox** (no network).
- **An LLM-as-Judge evaluation harness** (Gemini) over three test sets, with multi-model comparison reports.

## Layout

```
modal_app.py            # LLMServer (vLLM) + RAG augmentation + agentic tool loop + HTTP endpoint
retrieval.py            # Hybrid retriever: SQLite structured + bge-m3 semantic + embedding router
rag_corpus/             # Seed data: calendar.json, tasks.json, emails.json, docs/*.md
modal_client.py         # Call the endpoint from your machine (SDK / HTTP)
eval/
  run_eval_judge.py     # Orchestrator: generate -> judge -> markdown report
  config|target|judge|criteria|report.py
  test_cases_{chat,rag,rag_live}.yaml
  eval_retrieval.py     # measure retrieval recall@k (decoupled from generation)
  rag_diagnose.py       # tune the embedding-router threshold
  comparisons/          # model comparison reports (3-way, 2-way)
.env.example            # config template
```

## Models evaluated

Full reports live in `eval/comparisons/`.

| Model | `MODEL_NAME` | Notes |
|---|---|---|
| Qwen3.5-9B (default) | `Qwen/Qwen3.5-9B` | well-rounded, strong tool-calling |
| DeepSeek-R1-0528-Qwen3-8B | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | reasoning model, somewhat verbose |
| Gemma 4-E4B | `google/gemma-4-e4b-it` | gated (needs an HF token); runs text-only |

Across 97 cases (chat 63 + rag_live 34), judged by `gemini-3.1-flash-lite`:
**DeepSeek 94.7 ~= Gemma 94.2 > Qwen 92.1** (overall score). All three struggle most with multi-fact extraction from meeting minutes (`RL31`).

## Setup (one time)

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt    # Windows  (Linux/macOS: .venv/bin/pip)
.venv\Scripts\modal token new                    # log in to Modal (opens a browser)
```

Copy `.env.example` to `.env` and fill in:

- `GEMINI_API_KEY` — for the LLM-as-Judge (get one at https://aistudio.google.com/apikey).
- `MODEL_NAME`, `MODAL_GPU` (default `L40S`), `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`.

For a **gated** model (Gemma): create a Modal secret named `huggingface` holding `HF_TOKEN` (from an account that has accepted the license on Hugging Face).

## Deploy

```bash
.venv\Scripts\modal deploy modal_app.py
```

The output includes an **Endpoint URL** (HTTP). Copy it into `.env` as `MODAL_ENDPOINT_URL=...` if you want to call the service over HTTP.

## Quick test

```bash
# Local entrypoint (spawns one container, runs, then shuts down)
.venv\Scripts\modal run modal_app.py --question "what is on my calendar today"

# Via the client (after deploy)
.venv\Scripts\python modal_client.py "2+2=?"          # SDK
.venv\Scripts\python modal_client.py "2+2=?" http     # HTTP (requires MODAL_ENDPOINT_URL)
```

## Evaluation (LLM-as-Judge)

```bash
.venv\Scripts\python eval/run_eval_judge.py --set chat        # 63 conversational cases (inline data)
.venv\Scripts\python eval/run_eval_judge.py --set rag_live    # 34 RAG cases (real server-side retrieval)
.venv\Scripts\python eval/eval_retrieval.py                   # retrieval recall@k (cheap, no generation)
```

**Comparing another model:** change `MODEL_NAME` in `.env`, run `modal deploy modal_app.py`, then re-run the eval. Reports are written to `eval/results/judge/` (named with a model slug). The `generate` and `judge` stages can be run separately (see `--help`).

Retrieval unit tests (no GPU, uses a FakeEmbedder):

```bash
.venv\Scripts\python test_retrieval.py
```

## Key configuration (`.env`)

| Group | Variables |
|---|---|
| Serving | `MODEL_NAME`, `MODAL_GPU`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `DTYPE`, `MAX_NUM_SEQS` |
| Tools | `USE_TOOLS` |
| RAG | `USE_RETRIEVAL`, `EMBED_MODEL` (bge-m3), `RAG_TOP_K`, `RAG_MIN_SCORE`, `RAG_ROUTE_MIN_SCORE`, `RAG_REFERENCE_DATE` |
| Judge / Eval | `JUDGE_MODEL`, `JUDGE_TEMPERATURE`, `JUDGE_SEED`, `JUDGE_RPM`, `EVAL_MODE`, `EVAL_TEMPERATURE`, `EVAL_MAX_TOKENS` |

## Cost (Modal, rough)

GPU is billed per second while active; with `scaledown_window=5` the container shuts down about 5 seconds after the last request, dropping to $0. Subsequent cold starts (model already cached in a Volume) take roughly 10-20 seconds, and each inference request takes a few seconds.

## References

- Modal docs: https://modal.com/docs
- vLLM on Modal: https://modal.com/docs/examples/vllm_inference
