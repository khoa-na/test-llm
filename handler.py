"""
RunPod Serverless handler — vLLM worker cho chatbot thư ký.

Cold start: load model 1 lần.
Mỗi job: nhận prompt/messages → sinh response.

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
import runpod
from vllm import LLM, SamplingParams

# ───────────────────────────────────────────────
# Cold-start: load model 1 lần
# ───────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B-Instruct")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))
GPU_MEM_UTIL = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.92"))
DTYPE = os.getenv("DTYPE", "auto")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"

print(f"[handler] Loading model {MODEL_NAME} ...")
llm = LLM(
    model=MODEL_NAME,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEM_UTIL,
    dtype=DTYPE,
    trust_remote_code=TRUST_REMOTE_CODE,
)
tokenizer = llm.get_tokenizer()
print(f"[handler] Model ready.")


# ───────────────────────────────────────────────
# Job handler
# ───────────────────────────────────────────────
def handler(event):
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
            # tokenizer chưa support enable_thinking
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


# ───────────────────────────────────────────────
# Start serverless loop
# ───────────────────────────────────────────────
runpod.serverless.start({"handler": handler})
