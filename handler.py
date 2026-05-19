"""
RunPod Serverless handler — vLLM worker cho chatbot thư ký.

Cold start: lazy init model trong handler đầu tiên.

Input schema:
    {
      "input": {
        "messages": [{"role":"user","content":"..."}],   # hoặc "prompt": "..."
        "sampling_params": {
          "temperature": 0.7,
          "top_p": 0.95,
          "max_tokens": 1024
        },
        "thinking_mode": false
      }
    }
"""
import os

# ───────────────────────────────────────────────
# vLLM stability env — set TRƯỚC khi import vllm
# ───────────────────────────────────────────────
# Default dùng V0 engine cho ổn định hơn V1 (V1 hay fail subprocess init trên RunPod)
os.environ.setdefault("VLLM_USE_V1", "0")
# Worker multiproc method — KHÔNG dùng "fork" (CUDA fork conflict)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Tăng timeout init engine (RunPod cold start chậm)
os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "600")

# ───────────────────────────────────────────────
# HuggingFace download
# ───────────────────────────────────────────────
# Bật hf_transfer cho tốc độ download model nhanh hơn 5-10x (cần thư viện đi kèm).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
# RunPod inject secret HF_TOKEN tự động. HF SDK đọc cả HF_TOKEN lẫn HUGGING_FACE_HUB_TOKEN,
# nhưng vLLM/transformers ưu tiên HUGGING_FACE_HUB_TOKEN → mirror sang biến đó nếu chưa có.
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    print(f"[handler] HF_TOKEN detected: {_hf_token[:6]}...{_hf_token[-4:]}", flush=True)
else:
    print("[handler] No HF_TOKEN found — chỉ download được model public.", flush=True)

import runpod
from vllm import LLM, SamplingParams

# ───────────────────────────────────────────────
# Config từ env
# ───────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B-Instruct")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))           # giảm để tránh OOM KV cache
GPU_MEM_UTIL = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.88"))  # giảm để có headroom
DTYPE = os.getenv("DTYPE", "bfloat16")                            # explicit bf16, tránh auto detect lỗi
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "true").lower() == "true"  # bỏ CUDA graph → ổn định hơn
QUANTIZATION = os.getenv("QUANTIZATION", "").strip() or None      # None | awq | gptq | fp8

# Lazy globals
llm = None
tokenizer = None


def init_model():
    global llm, tokenizer
    if llm is None:
        print(f"[handler] Loading {MODEL_NAME}", flush=True)
        print(f"[handler]   max_model_len={MAX_MODEL_LEN} gpu_mem_util={GPU_MEM_UTIL}", flush=True)
        print(f"[handler]   dtype={DTYPE} enforce_eager={ENFORCE_EAGER} quant={QUANTIZATION}", flush=True)
        kwargs = dict(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            dtype=DTYPE,
            trust_remote_code=TRUST_REMOTE_CODE,
            enforce_eager=ENFORCE_EAGER,
            disable_log_stats=True,
        )
        if QUANTIZATION:
            kwargs["quantization"] = QUANTIZATION
        llm = LLM(**kwargs)
        tokenizer = llm.get_tokenizer()
        print("[handler] Model ready.", flush=True)


def handler(event):
    init_model()

    job_input = event.get("input", {}) or {}
    messages = job_input.get("messages")
    prompt = job_input.get("prompt", "")
    sp_dict = job_input.get("sampling_params") or {}
    thinking_mode = bool(job_input.get("thinking_mode", False))

    if messages:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_mode,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    elif prompt:
        text = prompt
    else:
        return {"error": "Thiếu 'messages' hoặc 'prompt' trong input."}

    sampling_params = SamplingParams(
        temperature=sp_dict.get("temperature", 0.7),
        top_p=sp_dict.get("top_p", 0.95),
        max_tokens=sp_dict.get("max_tokens", 1024),
        stop=sp_dict.get("stop"),
    )

    outputs = llm.generate([text], sampling_params)
    out = outputs[0].outputs[0]
    return {
        "text": out.text,
        "prompt_tokens": len(outputs[0].prompt_token_ids),
        "completion_tokens": len(out.token_ids),
        "finish_reason": out.finish_reason,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
