"""
RunPod Serverless handler — vLLM worker cho chatbot thư ký.

Cold start: lazy init model trong handler đầu tiên (tránh CUDA fork issue).

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

Output:
    {
      "text": "...",
      "prompt_tokens": N,
      "completion_tokens": N,
      "finish_reason": "stop"
    }
"""
import os
import multiprocessing as mp

# PHẢI set trước khi import vllm/torch — tránh "Cannot re-initialize CUDA in forked subprocess"
mp.set_start_method("spawn", force=True)

import runpod
from vllm import LLM, SamplingParams

# ───────────────────────────────────────────────
# Config từ env
# ───────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B-Instruct")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))
GPU_MEM_UTIL = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.92"))
DTYPE = os.getenv("DTYPE", "auto")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"

# Lazy globals — init bên trong handler để không touch CUDA ở import time
llm = None
tokenizer = None


def init_model():
    """Load model 1 lần duy nhất khi handler đầu tiên chạy."""
    global llm, tokenizer
    if llm is None:
        print(f"[handler] Loading model {MODEL_NAME} ...")
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            dtype=DTYPE,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        tokenizer = llm.get_tokenizer()
        print("[handler] Model ready.")


def handler(event):
    init_model()

    job_input = event.get("input", {}) or {}
    messages = job_input.get("messages")
    prompt = job_input.get("prompt", "")
    sp_dict = job_input.get("sampling_params") or {}
    thinking_mode = bool(job_input.get("thinking_mode", False))

    # Build prompt string
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
