"""
Client gọi Modal endpoint từ máy local.

Cách 1: Gọi qua Modal Function (cần `pip install modal`)
    python modal_client.py

Cách 2: Gọi qua HTTP endpoint (chỉ cần `requests`)
    Set MODAL_ENDPOINT_URL trong .env hoặc sửa biến URL ở dưới.
"""
import os
import sys
import time
import requests
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

load_dotenv()
env_path = Path(__file__).parent / ".env"
env_config = {
    **dotenv_values(env_path),
    **os.environ
}


def call_via_modal_sdk(question: str):
    """Gọi method trực tiếp qua modal SDK."""
    import modal
    cls = modal.Cls.from_name("test-llm-chatbot-thuky", "LLMServer")
    t0 = time.time()
    result = cls().generate.remote(
        messages=[{"role": "user", "content": question}],
        max_tokens=200,
    )
    print(f"[modal-sdk] {time.time()-t0:.1f}s")
    print(result)


def call_via_http(question: str):
    """Gọi qua HTTP endpoint (web URL)."""
    url = env_config.get("MODAL_ENDPOINT_URL", "")
    if not url:
        print("❌ Set MODAL_ENDPOINT_URL trong .env (URL hiện ra sau khi modal deploy)")
        return
    t0 = time.time()
    r = requests.post(
        url,
        json={
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 200,
        },
        timeout=600,
    )
    print(f"[http] {time.time()-t0:.1f}s status={r.status_code}")
    print(r.json())


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "2+2=?"
    mode = sys.argv[2] if len(sys.argv) > 2 else "sdk"
    if mode == "http":
        call_via_http(q)
    else:
        call_via_modal_sdk(q)
