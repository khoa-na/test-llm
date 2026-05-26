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

MAX_TOKENS: int = int(env_config.get("EVAL_MAX_TOKENS", "512"))
GENERATE_WORKERS: int = max(1, int(env_config.get("EVAL_GENERATE_WORKERS", "1")))
JUDGE_TEMPERATURE: float = float(env_config.get("JUDGE_TEMPERATURE", "0.0"))
JUDGE_RPM: int = int(env_config.get("JUDGE_RPM", "15"))

# Đọc cùng key MODEL_NAME như modal_app.py để đồng bộ.
TARGET_MODEL_NAME: str = env_config.get("MODEL_NAME", "Qwen/Qwen3.5-9B")

# Slug an toàn cho filename: "Qwen/Qwen3.5-9B-Instruct" → "Qwen_Qwen3.5-9B-Instruct"
_MODEL_SLUG = TARGET_MODEL_NAME.replace("/", "_").replace("\\", "_").replace(":", "_")

# File output gắn slug model để dễ phân biệt khi đổi model (Qwen vs Llama vs ...).
TEST_CASES_PATH: Path = Path(__file__).parent / "test_cases.yaml"
OUTPUTS_JSON_PATH: Path = Path(__file__).parent / f"eval_outputs__{_MODEL_SLUG}.json"
REPORT_PATH: Path = Path(__file__).parent / f"eval_report_judge__{_MODEL_SLUG}.md"

GEMINI_API_KEY: str | None = env_config.get("GEMINI_API_KEY") or env_config.get("GOOGLE_API_KEY")
DASHSCOPE_API_KEY: str | None = env_config.get("DASHSCOPE_API_KEY")
MODAL_ENDPOINT_URL: str = env_config.get("MODAL_ENDPOINT_URL", "")

# Ngưỡng pass / production-ready trên thang 0-100.
PASS_THRESHOLD: int = 60
PRODUCTION_THRESHOLD: int = 75
