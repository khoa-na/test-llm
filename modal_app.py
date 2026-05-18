"""
Modal serverless deployment cho chatbot thư ký.

Deploy:
    modal deploy modal_app.py

Test 1 lần (chạy local entrypoint):
    modal run modal_app.py

Gọi từ client khác:
    Xem `modal_client.py`
"""
import os
import modal

# ───────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B-Instruct")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
GPU_MEM_UTIL = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.92"))
DTYPE = os.getenv("DTYPE", "auto")
GPU_TYPE = os.getenv("MODAL_GPU", "A10G")  # A10G 24GB | A100-40GB | H100

# ───────────────────────────────────────────────
# Container image (Modal tự build trên cloud)
# ───────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.8.5",
        "transformers>=4.45",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # download model nhanh hơn 5-10x
        "VLLM_USE_V1": "1",
    })
)

# Volume cache HuggingFace — model weights chỉ download 1 lần
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("test-llm-chatbot-thuky")


# ───────────────────────────────────────────────
# LLM Server class
# ───────────────────────────────────────────────
@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/.cache/huggingface": hf_cache},
    scaledown_window=120,  # idle 120s → tắt → $0
    timeout=600,
    min_containers=0,
)
class LLMServer:
    @modal.enter()
    def load(self):
        """Cold start: load model 1 lần."""
        from vllm import LLM
        print(f"[modal] Loading {MODEL_NAME} on {GPU_TYPE} ...")
        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            dtype=DTYPE,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("[modal] Model ready.")

    @modal.method()
    def generate(
        self,
        messages: list | None = None,
        prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        thinking_mode: bool = False,
    ) -> dict:
        from vllm import SamplingParams

        if messages:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking_mode,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        elif prompt:
            text = prompt
        else:
            return {"error": "Thiếu 'messages' hoặc 'prompt'."}

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate([text], params)
        out = outputs[0].outputs[0]
        return {
            "text": out.text,
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(out.token_ids),
            "finish_reason": out.finish_reason,
        }


# ───────────────────────────────────────────────
# HTTP endpoint (OpenAI-compat-ish)
# ───────────────────────────────────────────────
@app.function(image=image)
@modal.fastapi_endpoint(method="POST", docs=True)
def chat(item: dict) -> dict:
    """
    POST endpoint cho client HTTP.

    Body:
        {
          "messages": [{"role":"user","content":"..."}],
          "max_tokens": 200,
          "temperature": 0.7,
          "thinking_mode": false
        }
    """
    return LLMServer().generate.remote(
        messages=item.get("messages"),
        prompt=item.get("prompt", ""),
        temperature=item.get("temperature", 0.7),
        top_p=item.get("top_p", 0.95),
        max_tokens=item.get("max_tokens", 1024),
        thinking_mode=item.get("thinking_mode", False),
    )


# ───────────────────────────────────────────────
# Local entrypoint test
# ───────────────────────────────────────────────
@app.local_entrypoint()
def main(question: str = "2+2=?"):
    """Chạy: modal run modal_app.py --question 'Tôi có lịch họp gì hôm nay?'"""
    result = LLMServer().generate.remote(
        messages=[{"role": "user", "content": question}],
        max_tokens=200,
    )
    print("\n=== RESULT ===")
    print(result.get("text"))
    print(f"\ntokens: in={result.get('prompt_tokens')} out={result.get('completion_tokens')}")
