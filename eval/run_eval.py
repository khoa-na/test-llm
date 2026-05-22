import os
import sys
import time
import yaml
import requests
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

load_dotenv()
env_path = Path(__file__).parent.parent / ".env"
env_config = {
    **dotenv_values(env_path),
    **os.environ
}

# Cấu hình tham số từ biến môi trường hoặc tham số dòng lệnh
MODE = sys.argv[1] if len(sys.argv) > 1 else env_config.get("EVAL_MODE", "dashscope")  # "dashscope", "sdk" hoặc "http"
MAX_TOKENS = 512

def call_model_dashscope(messages, thinking_mode=False):
    from openai import OpenAI
    t0 = time.time()
    api_key = env_config.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Thiếu DASHSCOPE_API_KEY trong .env để chạy DashScope model")
        
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    
    extra_body = {}
    if thinking_mode:
        extra_body["enable_thinking"] = True
        
    completion = client.chat.completions.create(
        model="qwen3.6-max-preview",
        messages=messages,
        extra_body=extra_body,
        stream=True,
        max_tokens=MAX_TOKENS,
        stream_options={"include_usage": True}
    )
    
    reasoning_chunks = []
    content_chunks = []
    prompt_tokens = 0
    completion_tokens = 0
    
    for chunk in completion:
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
            continue
            
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            reasoning_chunks.append(delta.reasoning_content)
        if hasattr(delta, "content") and delta.content is not None:
            content_chunks.append(delta.content)
            
    reasoning = "".join(reasoning_chunks)
    content = "".join(content_chunks)
    
    full_text = ""
    if reasoning:
        full_text += f"<thinking>\n{reasoning}\n</thinking>\n"
    full_text += content
    
    latency = time.time() - t0
    
    if prompt_tokens == 0:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = int(total_chars / 4)
    if completion_tokens == 0:
        completion_tokens = int(len(full_text) / 4)
        
    return {
        "text": full_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }, latency

def call_model_sdk(messages, thinking_mode=False):
    import modal
    cls = modal.Cls.from_name("test-llm-chatbot-thuky", "LLMServer")
    t0 = time.time()
    res = cls().generate.remote(
        messages=messages,
        max_tokens=MAX_TOKENS,
        thinking_mode=thinking_mode
    )
    latency = time.time() - t0
    return res, latency

def call_model_http(messages, thinking_mode=False):
    url = env_config.get("MODAL_ENDPOINT_URL", "")
    if not url:
        raise ValueError("Thiếu MODAL_ENDPOINT_URL trong .env khi chạy chế độ HTTP")
    t0 = time.time()
    r = requests.post(
        url,
        json={
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "thinking_mode": thinking_mode
        },
        timeout=600
    )
    latency = time.time() - t0
    r.raise_for_status()
    return r.json(), latency

def evaluate_case(case, system_prompt=""):
    # Trích xuất dữ liệu cuộc hội thoại
    messages = list(case.get("turns", []))
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
        
    thinking_mode = case.get("thinking_mode", False) or case.get("thinking_compare", False)
    
    # Thực hiện gọi API
    if MODE == "dashscope":
        res, latency = call_model_dashscope(messages, thinking_mode)
    elif MODE == "http":
        res, latency = call_model_http(messages, thinking_mode)
    else:
        res, latency = call_model_sdk(messages, thinking_mode)
        
    output_text = res.get("text", "")
    prompt_tokens = res.get("prompt_tokens", 0)
    completion_tokens = res.get("completion_tokens", 0)

    
    # Xác thực kết quả
    expected = case.get("expected", {})
    must_contain = expected.get("must_contain", [])
    must_not_contain = expected.get("must_not_contain", [])
    
    passed = True
    failed_reasons = []
    
    for word in must_contain:
        if word.lower() not in output_text.lower():
            passed = False
            failed_reasons.append(f"Thiếu từ bắt buộc: '{word}'")
            
    for word in must_not_contain:
        if word.lower() in output_text.lower():
            passed = False
            failed_reasons.append(f"Chứa từ cấm: '{word}'")
            
    return {
        "id": case.get("id"),
        "name": case.get("name"),
        "use_case": case.get("use_case"),
        "criteria": case.get("criteria", []),
        "passed": passed,
        "failed_reasons": failed_reasons,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output": output_text,
        "turns": case.get("turns", []),
        "eval_notes": expected.get("eval_notes", "")
    }

def main():
    yaml_path = os.path.join(os.path.dirname(__file__), "test_cases.yaml")
    if not os.path.exists(yaml_path):
        print(f"❌ Không tìm thấy file cấu hình: {yaml_path}")
        sys.exit(1)
        
    print(f"📖 Đang đọc các ca kiểm thử từ {yaml_path}...")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    test_cases = data.get("test_cases", [])
    system_prompt = data.get("system_prompt", "")
    print(f"📋 Tìm thấy {len(test_cases)} ca kiểm thử.")
    print(f"🚀 Bắt đầu đánh giá tự động ở chế độ: {MODE.upper()}...\n")
    
    results = []
    passed_count = 0
    total_latency = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Đang chạy {case['id']} - {case['name']}...", end="", flush=True)
        try:
            res = evaluate_case(case, system_prompt)
            results.append(res)
            total_latency += res["latency"]
            total_prompt_tokens += res["prompt_tokens"]
            total_completion_tokens += res["completion_tokens"]
            
            if res["passed"]:
                passed_count += 1
                print(f" ✅ PASS ({res['latency']:.2f}s)")
            else:
                print(f" ❌ FAIL ({res['latency']:.2f}s) - Lý do: {', '.join(res['failed_reasons'])}")
        except Exception as e:
            print(f" 💥 ERROR: {str(e)}")
            results.append({
                "id": case.get("id"),
                "name": case.get("name"),
                "use_case": case.get("use_case"),
                "passed": False,
                "failed_reasons": [f"Lỗi thực thi: {str(e)}"],
                "latency": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "output": "",
                "eval_notes": "Lỗi hệ thống"
            })
            
    # Ghi nhận báo cáo kết quả đánh giá (eval_report.md)
    report_path = os.path.join(os.path.dirname(__file__), "eval_report.md")
    avg_latency = total_latency / len(test_cases) if test_cases else 0
    
    print("\n" + "="*50)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TỰ ĐỘNG CHUNG")
    print("="*50)
    print(f"Tổng số ca kiểm thử: {len(test_cases)}")
    print(f"Thành công (PASS)  : {passed_count} ({passed_count/len(test_cases)*100:.1f}%)")
    print(f"Thất bại (FAIL)    : {len(test_cases) - passed_count}")
    print(f"Thời gian TB (s)   : {avg_latency:.2f}s")
    print(f"Tổng Prompt Tokens : {total_prompt_tokens}")
    print(f"Tổng Compl. Tokens : {total_completion_tokens}")
    print("="*50)
    
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Báo cáo Kết quả Đánh giá Tự động (Auto-Evaluation Report)\n\n")
        rf.write(f"* **Thời gian thực hiện**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"* **Chế độ kiểm thử**: {MODE.upper()}\n")
        rf.write(f"* **Tổng số ca kiểm thử**: {len(test_cases)}\n")
        rf.write(f"* **Tỷ lệ vượt qua (PASS)**: **{passed_count}/{len(test_cases)} ({passed_count/len(test_cases)*100:.1f}%)**\n")
        rf.write(f"* **Thời gian đáp ứng trung bình**: {avg_latency:.2f}s\n")
        rf.write(f"* **Tổng tài nguyên tiêu thụ**: In: {total_prompt_tokens} tokens | Out: {total_completion_tokens} tokens\n\n")
        
        rf.write("## Chi tiết kết quả từng ca kiểm thử\n\n")
        rf.write("| ID | Use Case | Ca kiểm thử | Trạng thái | Latency (s) | Ghi chú đánh giá |\n")
        rf.write("|---|---|---|---|---|---|\n")
        for r in results:
            status_str = "🟢 PASS" if r["passed"] else "🔴 FAIL"
            rf.write(f"| `{r['id']}` | `{r['use_case']}` | {r['name']} | {status_str} | {r['latency']:.2f}s | {r['eval_notes']} |\n")
            
        rf.write("\n## Chi tiết các ca kiểm thử thất bại (FAIL Details)\n\n")
        failed_cases = [r for r in results if not r["passed"]]
        if not failed_cases:
            rf.write("🎉 Tuyệt vời! Không có ca kiểm thử nào bị thất bại.\n\n")
        else:
            for fc in failed_cases:
                rf.write(f"### ❌ `{fc['id']}` - {fc['name']} (Use Case: `{fc['use_case']}`)\n\n")
                rf.write(f"* **Lý do thất bại**: {', '.join(fc['failed_reasons'])}\n")
                rf.write(f"* **Output thực tế**:\n```text\n{fc['output']}\n```\n")
                rf.write(f"* **Ghi chú đánh giá**: {fc['eval_notes']}\n\n")

        rf.write("## Chi tiết toàn bộ các câu trả lời từ Model (All Responses Details)\n\n")
        rf.write("Dưới đây là chi tiết câu hỏi đầu vào và phản hồi thực tế của mô hình cho từng ca kiểm thử:\n\n")
        for r in results:
            status_symbol = "🟢 PASS" if r["passed"] else "🔴 FAIL"
            rf.write(f"### 📌 `{r['id']}` - {r['name']} ({status_symbol})\n\n")
            rf.write(f"* **Use Case**: `{r['use_case']}` | **Latency**: {r['latency']:.2f}s | **Tokens**: In {r['prompt_tokens']} / Out {r['completion_tokens']}\n")
            rf.write("* **Hội thoại đầu vào**:\n")
            for t in r.get("turns", []):
                role_capitalized = t.get("role", "user").capitalize()
                content_escaped = t.get("content", "").replace("\n", "\n  ")
                rf.write(f"  * **{role_capitalized}**: {content_escaped}\n")
            rf.write(f"* **Câu trả lời của Model**:\n```text\n{r['output']}\n```\n")
            rf.write(f"* **Ghi chú tiêu chí**: {r['eval_notes']}\n\n")
            rf.write("---\n\n")

    print(f"\n📝 Đã lưu báo cáo chi tiết tại: {report_path}")

if __name__ == "__main__":
    main()
