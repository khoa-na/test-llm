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
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))           # A10G 24GB chật, để 4096 cho an toàn
GPU_MEM_UTIL = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")) # chừa headroom KV cache
DTYPE = os.getenv("DTYPE", "bfloat16")
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "true").lower() == "true"  # skip CUDA graphs → save ~2GB
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "4"))                # batch nhỏ cho A10G
GPU_TYPE = os.getenv("MODAL_GPU", "A10G")  # A10G 24GB | A100-40GB | H100

# ───────────────────────────────────────────────
# Container image (Modal tự build trên cloud)
# ───────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.21.0-cu129-ubuntu2404",
        add_python=None,  # image đã có python 3.12 (nhưng chỉ binary python3)
        # vLLM image có ENTRYPOINT ["vllm"] → chặn Modal container entrypoint. Clear nó.
        setup_dockerfile_commands=["ENTRYPOINT []"],
    )
    # Modal gọi `python -m pip ...` — image chỉ có python3 → symlink trước.
    .run_commands("ln -sf $(which python3) /usr/local/bin/python")
    # Modal không chạy entrypoint của image — Modal Function tự spawn python process
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.26",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # download model nhanh 5-10x
        # V1 engine là default từ vLLM 0.20+, không cần set
        "VLLM_NO_USAGE_STATS": "1",
    })
)

# Volumes cache:
#   - hf-cache: model weights (download 1 lần từ HF)
#   - vllm-cache: torch.compile + CUDA graph artifacts (compile 1 lần)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("test-llm-chatbot-thuky")


# ───────────────────────────────────────────────
# LLM Server class
# ───────────────────────────────────────────────
@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_name("huggingface")],  # inject HF_TOKEN env var
    enable_memory_snapshot=True,  # CPU snapshot — skip imports nặng ở cold start sau
    scaledown_window=120,          # idle 120s → tắt → $0
    timeout=600,
    min_containers=0,
)
class LLMServer:
    @modal.enter(snap=True)
    def load_imports(self):
        """Pre-snapshot (CPU only): chạy imports nặng. KHÔNG được đụng GPU/CUDA ở đây."""
        print(f"[modal] [snap] Importing vllm/transformers ...", flush=True)
        import vllm  # noqa: F401
        import transformers  # noqa: F401
        # Tuyệt đối không gọi torch.cuda.is_available() — sẽ init CUDA và phá snapshot.
        print("[modal] [snap] Imports done — ready to snapshot.", flush=True)

    @modal.enter(snap=False)
    def load_model(self):
        """Post-snapshot (GPU available): load weights lên VRAM."""
        from vllm import LLM
        # Mirror HF_TOKEN sang HUGGING_FACE_HUB_TOKEN (huggingface_hub check cả 2).
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            print(f"[modal] HF_TOKEN detected: {hf_token[:6]}...{hf_token[-4:]}", flush=True)
        else:
            print("[modal] No HF_TOKEN — chỉ download được model public.", flush=True)
        print(f"[modal] Loading {MODEL_NAME} on {GPU_TYPE} ...", flush=True)
        print(f"[modal]   max_model_len={MAX_MODEL_LEN} gpu_mem={GPU_MEM_UTIL} "
              f"enforce_eager={ENFORCE_EAGER} max_num_seqs={MAX_NUM_SEQS}", flush=True)
        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_num_seqs=MAX_NUM_SEQS,
            dtype=DTYPE,
            enforce_eager=ENFORCE_EAGER,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("[modal] Model ready.", flush=True)

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
