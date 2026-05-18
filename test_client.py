"""
Client test — gọi RunPod Serverless endpoint từ máy local.

Chạy:
    python test_client.py "Câu hỏi của bạn"
"""
import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")

if not API_KEY or not ENDPOINT_ID:
    print("❌ Thiếu RUNPOD_API_KEY hoặc RUNPOD_ENDPOINT_ID trong .env")
    sys.exit(1)

BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


def ask(prompt: str, max_tokens: int = 200, thinking: bool = False):
    payload = {
        "input": {
            "messages": [{"role": "user", "content": prompt}],
            "sampling_params": {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": max_tokens,
            },
            "thinking_mode": thinking,
        }
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/run", headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    job_id = r.json()["id"]
    print(f"Job: {job_id}")

    while True:
        elapsed = time.time() - t0
        if elapsed > 300:
            print("⏱ Timeout 5 phút")
            return
        time.sleep(2)
        data = requests.get(f"{BASE}/status/{job_id}", headers=HEADERS, timeout=15).json()
        status = data.get("status")
        print(f"  [{elapsed:5.1f}s] {status}")
        if status == "COMPLETED":
            out = data.get("output", {})
            print(f"\n✅ DONE ({elapsed:.1f}s)")
            print("─" * 50)
            print(out.get("text", out))
            print("─" * 50)
            print(f"tokens: in={out.get('prompt_tokens')} out={out.get('completion_tokens')}")
            return
        if status in ("FAILED", "CANCELLED"):
            print(f"\n❌ {status}")
            print(data)
            return


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "2+2=?"
    ask(q)
