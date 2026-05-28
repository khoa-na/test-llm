"""Cấu hình + CLI args cho eval pipeline.

Một chỗ duy nhất chứa: env loading, parse args, constants, encoding setup.
Các module khác chỉ `from config import ...`.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# Windows console mặc định cp1252 → crash khi print Unicode tiếng Việt. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

load_dotenv()
_ENV_PATH = Path(__file__).parent.parent / ".env"
env_config = {
    **dotenv_values(_ENV_PATH),
    **os.environ,
}


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="run_eval_judge",
        description="LLM-as-Judge evaluation runner (Modal target + Gemini/Qwen judge).",
    )
    p.add_argument(
        "stage", nargs="?", default="all", choices=["generate", "judge", "all"],
        help="Stage: generate (chỉ sinh response), judge (chỉ chấm), all (mặc định).",
    )
    p.add_argument(
        "mode", nargs="?", default=None, choices=["sdk", "http"],
        help="Modal transport: sdk (default) hoặc http. Override EVAL_MODE.",
    )
    p.add_argument(
        "judge_model", nargs="?", default=None,
        help="Tên judge model. Override JUDGE_MODEL.",
    )
    p.add_argument(
        "--set", dest="test_set", default=None, choices=["chat", "rag"],
        help="Bộ test: 'chat' (test_cases_chat.yaml) hoặc 'rag' (test_cases_rag.yaml). "
             "Override EVAL_SET (mặc định 'chat').",
    )
    p.add_argument(
        "--id", dest="case_ids", default=None,
        help="Chỉ chạy các case có ID này (phân tách bằng dấu phẩy) — để test riêng từng cái. "
             "Vd: --id T01.v1.rag  hoặc  --id EMO.v1,EMO.v2. Output ghi ra file *__subset.* riêng.",
    )
    p.add_argument(
        "--variant", dest="variant", default="all", choices=["base", "rag", "all"],
        help="Lọc trong bộ test theo type: 'base' = bản CHƯA có RAG (refuse baseline), "
             "'rag' = bản CÓ RAG (type rag_with_data, data đã truy xuất), 'all' = cả hai (mặc định). "
             "Output tách tên file theo variant để không đè nhau.",
    )
    # Hỗ trợ cú pháp cũ: thứ tự args có thể là bất kỳ — gom lại rồi reorder.
    # Vd `python run_eval_judge.py judge sdk gemini-2.5-flash` vẫn parse đúng.
    return p.parse_args(argv)


_args = _parse_args(sys.argv[1:])

STAGE: str = _args.stage
MODE: str = _args.mode or env_config.get("EVAL_MODE", "sdk")
if MODE not in ("sdk", "http"):
    print(f"⚠️ EVAL_MODE='{MODE}' không hỗ trợ. Fallback về 'sdk' (Modal SDK).")
    MODE = "sdk"

JUDGE_MODEL: str = _args.judge_model or env_config.get("JUDGE_MODEL", "gemini-3.1-flash-lite")

# Bộ test: chat (chỉ cần hội thoại) hoặc rag (cần data truy xuất). Mỗi bộ 1 file riêng.
TEST_SET: str = _args.test_set or env_config.get("EVAL_SET", "chat")
if TEST_SET not in ("chat", "rag"):
    print(f"⚠️ EVAL_SET='{TEST_SET}' không hỗ trợ. Fallback về 'chat'.")
    TEST_SET = "chat"

MAX_TOKENS: int = int(env_config.get("EVAL_MAX_TOKENS", "512"))
TARGET_TEMPERATURE: float = float(env_config.get("EVAL_TEMPERATURE", "0.2"))
# Bật python_exec tool cho target. None = theo USE_TOOLS_DEFAULT của Modal server.
_use_tools_raw = env_config.get("USE_TOOLS")
USE_TOOLS: bool | None = (
    None if _use_tools_raw is None else _use_tools_raw.strip().lower() == "true"
)
GENERATE_WORKERS: int = max(1, int(env_config.get("EVAL_GENERATE_WORKERS", "1")))
JUDGE_TEMPERATURE: float = float(env_config.get("JUDGE_TEMPERATURE", "0.0"))
JUDGE_SEED: int = int(env_config.get("JUDGE_SEED", "42"))
JUDGE_RPM: int = int(env_config.get("JUDGE_RPM", "15"))

# Đọc cùng key MODEL_NAME như modal_app.py để đồng bộ.
TARGET_MODEL_NAME: str = env_config.get("MODEL_NAME", "Qwen/Qwen3.5-9B")

# Lọc theo case ID (chạy riêng 1 vài case). None = chạy cả bộ.
CASE_IDS: list[str] | None = (
    [s.strip() for s in _args.case_ids.split(",") if s.strip()]
    if _args.case_ids else None
)

# Lọc theo variant: base (chưa RAG) | rag (có RAG) | all.
VARIANT: str = _args.variant

# Slug an toàn cho filename: "Qwen/Qwen3.5-9B-Instruct" → "Qwen_Qwen3.5-9B-Instruct"
_MODEL_SLUG = TARGET_MODEL_NAME.replace("/", "_").replace("\\", "_").replace(":", "_")

# Suffix output để các lần chạy lọc khác nhau KHÔNG đè kết quả của nhau.
_SUFFIX = (f"__{VARIANT}" if VARIANT != "all" else "") + ("__subset" if CASE_IDS else "")

_EVAL_DIR = Path(__file__).parent

# Output gom vào folder riêng: results/generate (model gen) + results/judge (judge report).
RESULTS_DIR: Path = _EVAL_DIR / "results"
GEN_DIR: Path = RESULTS_DIR / "generate"
JUDGE_DIR: Path = RESULTS_DIR / "judge"
GEN_DIR.mkdir(parents=True, exist_ok=True)
JUDGE_DIR.mkdir(parents=True, exist_ok=True)

# Tên file gắn slug model + bộ test để dễ phân biệt (Qwen vs Llama, chat vs rag).
TEST_CASES_PATH: Path = _EVAL_DIR / f"test_cases_{TEST_SET}.yaml"
OUTPUTS_JSON_PATH: Path = GEN_DIR / f"eval_outputs__{TEST_SET}__{_MODEL_SLUG}{_SUFFIX}.json"
REPORT_PATH: Path = JUDGE_DIR / f"eval_report_judge__{TEST_SET}__{_MODEL_SLUG}{_SUFFIX}.md"

GEMINI_API_KEY: str | None = env_config.get("GEMINI_API_KEY") or env_config.get("GOOGLE_API_KEY")
DASHSCOPE_API_KEY: str | None = env_config.get("DASHSCOPE_API_KEY")
MODAL_ENDPOINT_URL: str = env_config.get("MODAL_ENDPOINT_URL", "")

# Ngưỡng pass / production-ready trên thang 0-100.
PASS_THRESHOLD: int = 60
PRODUCTION_THRESHOLD: int = 75
