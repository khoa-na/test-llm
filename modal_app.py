"""
Modal serverless deployment cho chatbot thư ký.

Deploy:
    modal deploy modal_app.py

Test 1 lần (chạy local entrypoint):
    modal run modal_app.py

Gọi từ client khác:
    Xem `modal_client.py`
"""
import json
import os
import re
from pathlib import Path
from dotenv import dotenv_values, load_dotenv
import modal

load_dotenv()
env_path = Path(__file__).parent / ".env"
env_config = {
    **dotenv_values(env_path),
    **os.environ
}

# ───────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────
MODEL_NAME = env_config.get("MODEL_NAME", "Qwen/Qwen3.5-9B")
MAX_MODEL_LEN = int(env_config.get("MAX_MODEL_LEN", "4096"))           # A10G 24GB chật, để 4096 cho an toàn
GPU_MEM_UTIL = float(env_config.get("GPU_MEMORY_UTILIZATION", "0.90")) # chừa headroom KV cache
DTYPE = env_config.get("DTYPE", "bfloat16")
ENFORCE_EAGER = env_config.get("ENFORCE_EAGER", "true").lower() == "true"  # skip CUDA graphs → save ~2GB
MAX_NUM_SEQS = int(env_config.get("MAX_NUM_SEQS", "4"))                # batch nhỏ cho A10G
GPU_TYPE = env_config.get("MODAL_GPU", "A10G")  # A10G 24GB | A100-40GB | H100

# ───────────────────────────────────────────────
# Tool-calling config
# ───────────────────────────────────────────────
# Bật python_exec tool cho mọi generate có messages. Model tự quyết định khi nào gọi
# (qua Qwen3 chat-template tools=[...]). Set USE_TOOLS=false để tắt và quay về behaviour cũ.
USE_TOOLS_DEFAULT = env_config.get("USE_TOOLS", "true").lower() == "true"

# Tool schema (Qwen3 hermes-compatible: tokenizer chèn vào system prompt, model emit
# <tool_call>{"name":...,"arguments":{...}}</tool_call>).
PYTHON_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Chạy Python code trong môi trường isolated để tính toán deterministic. "
            "DÙNG cho mọi phép tính số học, ngày tháng, đếm/thống kê — không tự nhẩm. "
            "Code BẮT BUỘC print() kết quả ra stdout. "
            "Stdlib có sẵn: datetime, math, calendar, statistics, collections, re."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code; in kết quả bằng print().",
                }
            },
            "required": ["code"],
        },
    },
}

# Qwen3 emit `<tool_call>\n{json}\n</tool_call>`. DOTALL để \n match.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
MAX_TOOL_ITERS = 4            # bao nhiêu vòng tool tối đa / một generate request
SANDBOX_EXEC_TIMEOUT = 8       # giây / một lần exec code
SANDBOX_LIFETIME = 1800        # giây — sandbox tự kill sau idle

# ───────────────────────────────────────────────
# Container image cho LLM server (vLLM)
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

# ───────────────────────────────────────────────
# Sandbox image — môi trường isolated để chạy python_exec.
# Chỉ stdlib + python-dateutil (đủ cho mọi tính toán ngày/giờ thường gặp).
# Không có network (block_network=True ở Sandbox.create), không có gì ngoài Python.
# ───────────────────────────────────────────────
sandbox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("python-dateutil")
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
    scaledown_window=5,            # idle 5s → tắt GPU ngay → $0 (đánh đổi: request sau chịu cold start)
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
        hf_token = env_config.get("HF_TOKEN") or env_config.get("HUGGING_FACE_HUB_TOKEN")
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
            # Tắt vision/audio encoder của Qwen3.5 (mặc định multimodal).
            # Use case của mình text-only → save ~50-70s engine init + ~3-4GB VRAM cho KV cache.
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._sandbox = None  # lazy-init khi có tool call đầu tiên
        print("[modal] Model ready.", flush=True)

    @modal.exit()
    def cleanup(self):
        """Terminate sandbox khi container scale xuống — Modal Sandbox không tự cleanup theo parent."""
        sb = getattr(self, "_sandbox", None)
        if sb is not None:
            try:
                sb.terminate()
                print("[modal] [sandbox] terminated", flush=True)
            except Exception as e:
                print(f"[modal] [sandbox] cleanup error: {e}", flush=True)

    # ───────────────────────────────────────────────
    # Sandbox helpers
    # ───────────────────────────────────────────────
    def _get_sandbox(self):
        """Lazy-init một sandbox dài hơi (sleep infinity); reuse cho mọi exec trong instance."""
        sb = getattr(self, "_sandbox", None)
        if sb is not None:
            # Re-create nếu sandbox đã chết (timeout, hoặc Modal kill).
            try:
                # poll: nếu sandbox còn sống thì returncode is None
                if sb.returncode is None:
                    return sb
            except Exception:
                pass
            sb = None
        from modal import Sandbox
        sb = Sandbox.create(
            "sleep", "infinity",          # keep-alive, ta exec ad-hoc qua sb.exec()
            image=sandbox_image,
            app=app,
            timeout=SANDBOX_LIFETIME,
            block_network=True,            # KHÔNG cho code gọi internet
        )
        self._sandbox = sb
        print("[modal] [sandbox] created", flush=True)
        return sb

    def _exec_in_sandbox(self, code: str) -> str:
        """Chạy code trong sandbox. Trả về stdout (kèm stderr nếu có)."""
        try:
            sb = self._get_sandbox()
            p = sb.exec("python3", "-c", code, timeout=SANDBOX_EXEC_TIMEOUT)
            stdout = p.stdout.read() or ""
            stderr = p.stderr.read() or ""
            # đảm bảo process kết thúc
            try:
                p.wait()
            except Exception:
                pass
        except Exception as e:
            return f"[sandbox_error] {e}"
        out = stdout.rstrip()
        if stderr.strip():
            out = (out + "\n[stderr]\n" + stderr.rstrip()).strip()
        return out or "(no output)"

    # ───────────────────────────────────────────────
    # Chat-template & tool-call parsing
    # ───────────────────────────────────────────────
    def _render(self, messages, tools, thinking_mode):
        # Thử full (tools + enable_thinking) → bỏ enable_thinking → bỏ luôn tools.
        attempts = [
            dict(tools=tools, enable_thinking=thinking_mode),
            dict(tools=tools),
            dict(enable_thinking=thinking_mode),
            dict(),
        ]
        last_err = None
        for kw in attempts:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kw,
                )
            except TypeError as e:
                last_err = e
                continue
        # Không attempt nào pass — re-raise lỗi gốc để debug.
        raise last_err  # noqa: TRY200

    @staticmethod
    def _parse_tool_calls(text: str):
        """Tìm các block `<tool_call>{json}</tool_call>` trong output. Trả (calls, residual_text)."""
        calls = []
        for m in TOOL_CALL_RE.finditer(text):
            raw = m.group(1).strip()
            try:
                calls.append(json.loads(raw))
            except json.JSONDecodeError:
                # JSON hỏng (bị cắt do max_tokens, hoặc model emit kèm rác). Bỏ qua.
                pass
        residual = TOOL_CALL_RE.sub("", text).strip()
        return calls, residual

    # ───────────────────────────────────────────────
    # Public method
    # ───────────────────────────────────────────────
    @modal.method()
    def generate(
        self,
        messages: list | None = None,
        prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        thinking_mode: bool = False,
        use_tools: bool | None = None,
    ) -> dict:
        from vllm import SamplingParams

        # ───── Path cũ: raw prompt (không messages) ─────
        if prompt and not messages:
            params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            outputs = self.llm.generate([prompt], params)
            out = outputs[0].outputs[0]
            return {
                "text": out.text,
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(out.token_ids),
                "finish_reason": out.finish_reason,
            }

        if not messages:
            return {"error": "Thiếu 'messages' hoặc 'prompt'."}

        # ───── Path mới: messages + agentic tool loop ─────
        enable_tools = USE_TOOLS_DEFAULT if use_tools is None else bool(use_tools)
        tools = [PYTHON_EXEC_TOOL] if enable_tools else None

        params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        tool_log: list[dict] = []
        convo = list(messages)
        finish_reason = ""

        for iteration in range(MAX_TOOL_ITERS + 1):
            text = self._render(convo, tools, thinking_mode)
            outputs = self.llm.generate([text], params)
            out = outputs[0].outputs[0]
            total_prompt_tokens += len(outputs[0].prompt_token_ids)
            total_completion_tokens += len(out.token_ids)
            finish_reason = out.finish_reason or finish_reason

            # Không bật tool → 1 vòng, trả luôn.
            if not tools:
                return {
                    "text": out.text,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "finish_reason": finish_reason,
                }

            calls, residual = self._parse_tool_calls(out.text)
            # Hết tool call (hoặc hit cap) → đáp án cuối.
            if not calls or iteration == MAX_TOOL_ITERS:
                return {
                    "text": out.text,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "finish_reason": finish_reason,
                    "tool_calls": tool_log,
                    "iterations": iteration + 1,
                }

            # Có tool call → append assistant message + chạy tools + append tool messages.
            convo.append({
                "role": "assistant",
                "content": residual,
                "tool_calls": [
                    {
                        "id": f"call_{iteration}_{i}",
                        "type": "function",
                        "function": {
                            "name": c.get("name", ""),
                            "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False),
                        },
                    }
                    for i, c in enumerate(calls)
                ],
            })

            for i, c in enumerate(calls):
                name = c.get("name", "")
                args = c.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if name == "python_exec":
                    code = args.get("code", "") if isinstance(args, dict) else ""
                    result = self._exec_in_sandbox(code) if code else "[error] empty code"
                else:
                    result = f"[error] unknown tool: {name}"

                tool_log.append({
                    "iter": iteration,
                    "name": name,
                    "code": (args.get("code", "") if isinstance(args, dict) else "")[:500],
                    "result": result[:500],
                })
                # Cap tool result để không phá max_model_len.
                convo.append({
                    "role": "tool",
                    "tool_call_id": f"call_{iteration}_{i}",
                    "content": result[:2000],
                })

        # Unreachable theo logic trên (vòng `iteration == MAX_TOOL_ITERS` đã return).
        return {"text": "", "error": "tool_loop_exceeded"}


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
          "thinking_mode": false,
          "use_tools": true
        }
    """
    return LLMServer().generate.remote(
        messages=item.get("messages"),
        prompt=item.get("prompt", ""),
        temperature=item.get("temperature", 0.7),
        top_p=item.get("top_p", 0.95),
        max_tokens=item.get("max_tokens", 1024),
        thinking_mode=item.get("thinking_mode", False),
        use_tools=item.get("use_tools"),
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
    if result.get("tool_calls"):
        print(f"tool_calls: {len(result['tool_calls'])} (iters={result.get('iterations')})")
        for tc in result["tool_calls"]:
            print(f"  - {tc['name']}({tc['code'][:80]}...) → {tc['result'][:120]}")
