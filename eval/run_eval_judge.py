"""
LLM-as-Judge evaluation runner.

Pipeline:
  1. Đọc test_cases.yaml
  2. Gọi Modal model (SDK hoặc HTTP) → lấy response
  3. Gửi (câu hỏi + eval_notes + response) cho Gemini làm judge
  4. Judge trả JSON {passed, score, reasoning} → tổng hợp report

Chạy:
  python eval/run_eval_judge.py                 # mode sdk, model judge mặc định
  python eval/run_eval_judge.py http            # gọi target qua HTTP
  python eval/run_eval_judge.py sdk gemini-2.5-pro
"""
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import dotenv_values, load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

# Windows console mặc định cp1252 → crash khi print Unicode tiếng Việt. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

load_dotenv()
env_path = Path(__file__).parent.parent / ".env"
env_config = {
    **dotenv_values(env_path),
    **os.environ
}

# ───────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────
# Khám phá tham số dòng lệnh thông minh
args = [a.lower() for a in sys.argv[1:]]

# Tìm Stage: "generate", "judge", "all" (mặc định: "all")
STAGE = "all"
for s in ["generate", "judge", "all"]:
    if s in args:
        STAGE = s
        args.remove(s)
        break

# Tìm Mode target — chỉ chấp nhận Modal (sdk hoặc http endpoint).
# DashScope/Qwen-max cloud KHÔNG còn được hỗ trợ ở stage generate; benchmark chỉ
# chấm model self-hosted trên Modal.
MODE = env_config.get("EVAL_MODE", "sdk")
for m in ["sdk", "http"]:
    if m in args:
        MODE = m
        args.remove(m)
        break
if MODE not in ("sdk", "http"):
    print(f"⚠️ EVAL_MODE='{MODE}' không hỗ trợ. Fallback về 'sdk' (Modal SDK).")
    MODE = "sdk"

# Đối số còn lại nếu có sẽ là JUDGE_MODEL
JUDGE_MODEL = args[0] if args else env_config.get("JUDGE_MODEL", "gemini-3.1-flash-lite")

MAX_TOKENS = int(env_config.get("EVAL_MAX_TOKENS", "512"))
JUDGE_TEMPERATURE = float(env_config.get("JUDGE_TEMPERATURE", "0.0"))
JUDGE_RPM = int(env_config.get("JUDGE_RPM", "15"))  # Free-tier Gemini = 15 req/min
OUTPUTS_JSON_PATH = Path(__file__).parent / "eval_outputs.json"

# Tên model target (để ghi vào output cho biết câu trả lời sinh từ model nào).
# Đọc cùng key MODEL_NAME như modal_app.py để đồng bộ.
TARGET_MODEL_NAME = env_config.get("MODEL_NAME", "Qwen/Qwen3.5-9B")

# Sliding-window rate limiter — đảm bảo không vượt JUDGE_RPM request / 60s
from collections import deque
_judge_call_times: "deque[float]" = deque()


def _rate_limit_wait():
    """Block đến khi an toàn gọi request kế tiếp dưới giới hạn JUDGE_RPM."""
    if JUDGE_RPM <= 0:
        return
    now = time.time()
    # Loại các timestamp đã quá 60s
    while _judge_call_times and now - _judge_call_times[0] >= 60.0:
        _judge_call_times.popleft()
    if len(_judge_call_times) >= JUDGE_RPM:
        wait = 60.0 - (now - _judge_call_times[0]) + 0.2  # buffer 200ms
        if wait > 0:
            print(f" [rate-limit: sleep {wait:.1f}s] ", end="", flush=True)
            time.sleep(wait)
        # Cleanup lại sau khi sleep
        now = time.time()
        while _judge_call_times and now - _judge_call_times[0] >= 60.0:
            _judge_call_times.popleft()
    _judge_call_times.append(time.time())

# Khởi tạo Gemini client nếu có key
GEMINI_API_KEY = env_config.get("GEMINI_API_KEY") or env_config.get("GOOGLE_API_KEY")
gemini_judge_client = None
if GEMINI_API_KEY:
    try:
        gemini_judge_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo Gemini client: {e}")

# DashScope client (chỉ dùng cho judge khi JUDGE_MODEL bắt đầu bằng "qwen").
# Generate stage không còn gọi DashScope nữa.
DASHSCOPE_API_KEY = env_config.get("DASHSCOPE_API_KEY")
dashscope_judge_client = None
if DASHSCOPE_API_KEY:
    try:
        dashscope_judge_client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo DashScope client: {e}")


# ───────────────────────────────────────────────
# Target model callers (chỉ Modal: SDK + HTTP endpoint)
# ───────────────────────────────────────────────
def call_model_sdk(messages, thinking_mode=False):
    import modal
    cls = modal.Cls.from_name("test-llm-chatbot-thuky", "LLMServer")
    t0 = time.time()
    res = cls().generate.remote(
        messages=messages, max_tokens=MAX_TOKENS, thinking_mode=thinking_mode
    )
    return res, time.time() - t0


def call_model_http(messages, thinking_mode=False):
    url = env_config.get("MODAL_ENDPOINT_URL", "")
    if not url:
        raise ValueError("Thiếu MODAL_ENDPOINT_URL trong .env khi chạy HTTP")
    t0 = time.time()
    r = requests.post(
        url,
        json={"messages": messages, "max_tokens": MAX_TOKENS, "thinking_mode": thinking_mode},
        timeout=600,
    )
    r.raise_for_status()
    return r.json(), time.time() - t0



# ───────────────────────────────────────────────
# Judge — Gemini
# ───────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────
# Tiêu chí đánh giá — nguồn gốc duy nhất là Chatbot_ThuKy_UseCases_EvalCriteria_v3.xlsx
# (sheet "Tiêu chí đánh giá"). Wording mô tả + thang điểm copy gần nguyên văn.
# Xlsx KHÔNG có TC-07 (Latency) và TC-10 (Cost Efficiency) — hai chiều đó
# được đo quantitative riêng (latency seconds, token counts), không chấm qua judge.
# Xlsx cũng KHÔNG khai báo trọng số — báo cáo dùng avg đơn giản per TC.
# ───────────────────────────────────────────────────────────────────────────
CRITERIA_DEFINITIONS = {
    "TC-01": (
        "Accuracy – Độ đúng của thông tin: Model trả về thông tin đúng, không bịa đặt, "
        "không nhầm lẫn dữ liệu (ngày giờ, tên, task...). "
        "Thang điểm: 1=Sai hoàn toàn | 2=Sai nhiều | 3=Đúng một phần | 4=Gần đúng | "
        "5=Đúng hoàn toàn. [Tiêu chí quan trọng nhất – sai thông tin gây hậu quả nghiêm trọng.]"
    ),
    "TC-02": (
        "Intent Recognition – Nhận diện ý định: Model xác định đúng mục đích của yêu cầu "
        "(nhắc lịch, tạo task, tìm kiếm, soạn thảo, v.v.). "
        "Thang điểm: 1=<60% | 2=60-70% | 3=70-80% | 4=80-90% | 5=>90% chính xác. "
        "[Áp dụng cho cả câu mơ hồ, câu có nhiều ý định.]"
    ),
    "TC-03": (
        "Multi-turn Context Retention – Duy trì ngữ cảnh: Model ghi nhớ và sử dụng đúng "
        "ngữ cảnh từ các tin nhắn trước trong cùng cuộc trò chuyện. "
        "Thang điểm: 1=Mất context ngay | 2=Giữ 1-2 lượt | 3=Giữ 3-4 lượt | "
        "4=Giữ 5-7 lượt | 5=Giữ xuyên suốt. [Đặc biệt quan trọng cho luồng đặt lịch, "
        "tạo task nhiều bước.]"
    ),
    "TC-04": (
        "Language Quality – Văn phong & Ngữ pháp: Output viết đúng ngữ pháp, văn phong "
        "phù hợp (lịch sự, chuyên nghiệp), tự nhiên, không cứng nhắc. "
        "Thang điểm: 1=Sai ngữ pháp nặng | 2=Có lỗi rõ | 3=Chấp nhận được | 4=Tốt | "
        "5=Tự nhiên, chuyên nghiệp. [Quan trọng cho use case soạn thảo email, biên bản.]"
    ),
    "TC-05": (
        "Multilingual Support (VI/EN): Model nhận diện đúng ngôn ngữ input và phản hồi "
        "bằng đúng ngôn ngữ đó mà không pha trộn. "
        "Thang điểm: 1=<70% | 2=70-80% | 3=80-88% | 4=88-95% | 5=>95% đúng ngôn ngữ. "
        "[Ưu tiên tiếng Việt vì đây là ngôn ngữ chính của sếp.]"
    ),
    "TC-06": (
        "Temporal Reasoning – Lý luận thời gian: Model hiểu đúng các biểu thức thời gian "
        "tương đối ('tuần sau', 'thứ 4 tới', '2h chiều', 'sáng mai'...). "
        "Thang điểm: 1=<50% đúng | 2=50-65% | 3=65-80% | 4=80-92% | 5=>92% đúng. "
        "[Critical cho chức năng nhắc lịch – sai giờ = vô dụng.]"
    ),
    "TC-08": (
        "Robustness – Xử lý tình huống ngoại lệ: Model phản ứng hợp lý khi gặp yêu cầu "
        "không rõ ràng, thiếu thông tin, hoặc không thể thực hiện. "
        "Thang điểm: 1=Crash/Silent | 2=Lỗi không rõ | 3=Thông báo lỗi | 4=Hỏi rõ thêm | "
        "5=Xử lý khéo léo. [Model tốt sẽ hỏi thêm thông tin thay vì đoán mò.]"
    ),
    "TC-09": (
        "Consistency – Nhất quán câu trả lời: Với cùng câu hỏi hoặc câu hỏi tương tự, "
        "model cho kết quả nhất quán qua nhiều lần thử. "
        "Thang điểm: 1=Mâu thuẫn thường xuyên | 3=Đôi khi khác nhau | 5=Luôn nhất quán. "
        "[Đặc biệt quan trọng với thông tin ngày giờ, tên người, task.]"
    ),
}

_CRITERIA_BLOCK = "\n".join(f"- {tc}: {desc}" for tc, desc in CRITERIA_DEFINITIONS.items())

JUDGE_SYSTEM = """Bạn là chuyên gia đánh giá chatbot thư ký doanh nghiệp tiếng Việt.

Nhiệm vụ: chấm câu trả lời của model theo bộ tiêu chí dưới đây, kết hợp với eval_notes và system prompt của chatbot cho từng test case.

BỘ TIÊU CHÍ ĐÁNH GIÁ:
""" + _CRITERIA_BLOCK + """

Quy tắc chấm điểm (Thang 0-100, áp dụng RIÊNG cho mỗi TC trong criteria và một điểm OVERALL tổng):
- 90-100 (Xuất sắc): Đáp ứng đầy đủ tiêu chí, đúng hoàn toàn, văn phong tự nhiên, chuyên nghiệp, phù hợp cấp bậc người nhận.
- 75-89 (Tốt, production-ready): Đầy đủ thông tin cốt lõi, có thể có lỗi nhỏ về văn phong/hơi dài, không ảnh hưởng nghiệp vụ.
- 60-74 (Đạt, biên giới pass): Đáp ứng yêu cầu cốt lõi, có thể thiếu thông tin phụ hoặc lỗi định dạng nhẹ; KHÔNG được phép có hallucination/sai logic.
- 40-59 (Không đạt): Thiếu thông tin quan trọng, từ chối sai khi dữ liệu đầy đủ, vi phạm nhẹ ràng buộc cấm, hoặc tính toán/logic thời gian sai nhẹ.
- 0-39 (Thất bại): Vi phạm nghiêm trọng — hallucination, sai intent hoàn toàn, trả lời sai ngôn ngữ yêu cầu, hoặc vi phạm CẤM/CRITICAL.

Mỗi tiêu chí (per_tc) được chấm ĐỘC LẬP theo rubric riêng (xem BỘ TIÊU CHÍ phía trên) và scale về 0-100, kèm 1 CÂU NHẬN XÉT (`note`) giải thích NGẮN GỌN vì sao điểm như vậy — note phải nói cụ thể (vd. "Tính sai 1 ngày khi parse 'thứ 4 tuần sau'"), KHÔNG generic ("OK", "Tốt").

Điểm OVERALL là tổng hợp phản ánh chất lượng câu trả lời, KHÔNG nhất thiết là trung bình cộng — nếu có 1 TC vi phạm CRITICAL (vd. hallucination khi case test TC-01) thì OVERALL phải kéo xuống vùng FAIL bất kể các TC khác cao.

CHỈ chấm điểm cho những TC nằm trong `criteria` của case. KHÔNG tự thêm TC khác. Trả về JSON với `per_tc` chỉ chứa các TC này — mỗi TC có `score` (0-100) và `note` (1 câu).

`reasoning` ở cấp ngoài cùng chỉ là tóm tắt tổng quan 1 câu (vì per_tc đã có note chi tiết cho từng chiều).

Quy tắc chấm bổ sung:
- PASS khi câu trả lời thoả mãn TINH THẦN của tiêu chí, không cần khớp keyword cụ thể.
- FAIL khi vi phạm tiêu chí cốt lõi (đặc biệt: hallucination — bịa data không có như tên người, lịch, email; sai intent; sai ngôn ngữ).
- Việc bot có thêm hướng dẫn / gợi ý ngoài lề là CHẤP NHẬN ĐƯỢC nếu phần cốt lõi đúng.
- Bỏ qua các khối suy nghĩ <thinking>...</thinking> bên trong phản hồi của model khi đánh giá độ dài/định dạng/văn phong — chỉ chấm phần văn bản hiển thị cho người dùng.
- KHÔNG áp đặt yêu cầu xưng hô đặc biệt nào ngoài "lịch sự, chuyên nghiệp" theo định nghĩa TC-04 trong xlsx; văn phong phù hợp với cấp bậc người nhận của từng case (trang trọng với cấp trên/BGĐ/đối tác, thân thiện với cấp dưới).

QUY TẮC CHI TIẾT THEO TỪNG TIÊU CHÍ (BẮT BUỘC TUÂN THỦ — diễn giải để áp dụng cho single-case judging):

1. TC-01 / ACCURACY — Phân biệt rõ 3 trường hợp:
   (a) Bot KHẲNG ĐỊNH một sự kiện/dữ liệu cụ thể (tên người, lịch, email, số liệu) không có trong context → HALLUCINATION, score ≤ 2.
   (b) Bot KHẲNG ĐỊNH ĐÃ THỰC HIỆN một hành động tác động ra ngoài (đã gửi email, đã đặt lịch trong hệ thống, đã thông báo cho team) khi context không cho thấy bot có integration thực thi → HALLUCINATION, score ≤ 2. Lưu ý: "đã GHI NHẬN yêu cầu", "đã SOẠN THẢO bản nháp", "đã chuẩn bị" là CHẤP NHẬN; "đã GỬI cho BGĐ", "đã THIẾT LẬP thông báo trong hệ thống" khi không có dữ liệu hậu thuẫn = HALLUCINATION.
   (c) Bot trả lời "chưa có quyền truy cập / chưa có dữ liệu / vui lòng cung cấp thêm" KHI context thật sự thiếu data = HÀNH VI ĐÚNG, score 4-5. KHÔNG FAIL vì lý do "không đưa ra câu trả lời cụ thể".

2. TC-04 / LANGUAGE QUALITY:
   - Văn phong phải đúng ngữ pháp, tự nhiên, không cứng nhắc, lịch sự, chuyên nghiệp.
   - Mức độ trang trọng tương ứng cấp bậc người nhận trong case: cấp trên/BGĐ/đối tác → trang trọng ("Kính gửi", "Trân trọng"); đồng nghiệp/cấp dưới → thân thiện vừa phải ("em", "anh/chị"); trả lời sếp → ngắn gọn, lịch sự, không cứng nhắc.
   - KHÔNG ép buộc cụm xưng hô cụ thể (như "Dạ thưa sếp", "Báo cáo sếp") trừ khi case explicitly yêu cầu.
   - Sai chính tả nặng, văn phong không phù hợp ngữ cảnh → POOR_LANGUAGE.

3. TC-05 / LANGUAGE — Áp dụng nghiêm:
   - Nếu user viết 100% tiếng Anh (case `language: en`): response BẮT BUỘC 100% tiếng Anh từ đầu đến cuối, KỂ CẢ phần mở đầu/lời chào/ký tên. Bất kỳ cụm tiếng Việt nào (vd. "Kính gửi", "Trân trọng", "Tôi đã ghi nhận") → POOR_LANGUAGE, score ≤ 2.
   - Nếu user viết tiếng Việt: response 100% tiếng Việt, không chèn từ tiếng Anh trừ thuật ngữ kỹ thuật (KPI, deadline, email...).
   - Code-switch (case `language: mixed`): ưu tiên trả lời bằng tiếng Việt lịch sự.

4. TC-08 / REFUSAL & ROBUSTNESS:
   - Khi user hỏi data cụ thể mà context KHÔNG cung cấp (vd. "email từ X nói gì", "task Y đang ở trạng thái nào", "lịch họp 14h với ai") → bot trả lời "chưa có quyền truy cập / chưa có dữ liệu" + xin sếp cung cấp = PASS hoàn toàn (4-5 điểm). KHÔNG được trừ điểm vì "thiếu thông tin trong câu trả lời" — đó chính là hành vi đúng.
   - Khi yêu cầu THIẾU info bắt buộc (vd. "đặt lịch họp đi" mà thiếu thời gian/đối tượng) → bot phải hỏi lại để làm rõ, không tự đoán = PASS.
   - Khi yêu cầu NGOÀI phạm vi nghiệp vụ → bot từ chối lịch sự + hướng dẫn kênh phù hợp = PASS.

5. CÁCH ĐÁNH GIÁ `must_not_contain`:
   - Cụm từ trong `must_not_contain` chỉ là TỪ KHÓA GỢI Ý cảnh báo hallucination, KHÔNG phải pattern khớp tuyệt đối.
   - Nếu cụm từ xuất hiện trong NGỮ CẢNH PHỦ ĐỊNH/REFUSAL (vd. "không có trùng lặp", "chưa thể xác nhận đã gửi", "vui lòng cung cấp nội dung email") → KHÔNG tính là vi phạm.
   - Chỉ tính vi phạm khi cụm từ xuất hiện trong NGỮ CẢNH KHẲNG ĐỊNH (vd. bot tự bịa "anh Tuấn nói rằng...", "Minh đã gửi báo cáo Q2 hôm qua").

Mã vi phạm trong trường `violations` chỉ được phép chọn từ danh sách ENUM sau:
- `HALLUCINATION`: Bịa đặt thông tin (tên người, lịch, email, số liệu) không có trong dữ liệu đầu vào.
- `WRONG_INTENT`: Không nhận diện đúng hoặc hiểu sai ý định/yêu cầu của người dùng.
- `POOR_LANGUAGE`: Vi phạm quy định về ngôn ngữ (ví dụ: dùng sai tiếng Anh/Việt), lỗi chính tả/ngữ pháp nặng, hoặc văn phong không phù hợp.
- `MISSING_INFORMATION`: Thiếu thông tin cốt lõi bắt buộc theo tiêu chí đánh giá.
- `WRONG_LOGIC_CALCULATION`: Tính toán sai số liệu, sai lệch logic thời gian, hoặc sắp xếp sai độ ưu tiên.
- `OVER_VERBOSITY`: Phản hồi quá dài dòng, vi phạm ràng buộc về độ dài ngắn (ví dụ: yêu cầu trả lời đúng 1 câu nhưng viết đoạn dài).
- `INCORRECT_REFUSAL`: Từ chối xử lý sai (tự ý từ chối trong khi dữ liệu đầu vào đã cung cấp đầy đủ).
- `OTHER`: Các lỗi nghiệp vụ khác ngoài danh sách trên.

Trả về JSON đúng schema: `overall` (0-100), `per_tc` (dict TC→{score, note} chỉ cho criteria của case), `reasoning` (tóm tắt 1 câu tiếng Việt), `violations` (list mã)."""


# Ngưỡng (thang 0-100): pass = overall >= 60, production-ready = overall >= 75.
PASS_THRESHOLD = 60
PRODUCTION_THRESHOLD = 75

_TC_VERDICT_OBJ = {
    "type": "OBJECT",
    "description": "Điểm + nhận xét RIÊNG cho 1 tiêu chí.",
    "properties": {
        "score": {"type": "INTEGER", "description": "0-100"},
        "note": {"type": "STRING", "description": "1 câu giải thích vì sao điểm như vậy"},
    },
    "required": ["score", "note"],
}

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "overall": {
            "type": "INTEGER",
            "description": "Điểm tổng 0-100 cho cả response (xem rubric).",
        },
        "per_tc": {
            "type": "OBJECT",
            "description": (
                "Điểm + nhận xét RIÊNG cho từng tiêu chí ÁP DỤNG cho case "
                "(chỉ bao gồm các TC liệt kê trong criteria của case, không thêm)."
            ),
            "properties": {
                "TC-01": _TC_VERDICT_OBJ,
                "TC-02": _TC_VERDICT_OBJ,
                "TC-03": _TC_VERDICT_OBJ,
                "TC-04": _TC_VERDICT_OBJ,
                "TC-05": _TC_VERDICT_OBJ,
                "TC-06": _TC_VERDICT_OBJ,
                "TC-08": _TC_VERDICT_OBJ,
                "TC-09": _TC_VERDICT_OBJ,
            },
        },
        "reasoning": {"type": "STRING", "description": "Tóm tắt tổng quan 1 câu (per_tc đã có note chi tiết)"},
        "violations": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
                "enum": [
                    "HALLUCINATION",
                    "WRONG_INTENT",
                    "POOR_LANGUAGE",
                    "MISSING_INFORMATION",
                    "WRONG_LOGIC_CALCULATION",
                    "OVER_VERBOSITY",
                    "INCORRECT_REFUSAL",
                    "OTHER"
                ]
            },
            "description": "Danh sách mã vi phạm tiêu chí (rỗng nếu PASS)",
        },
    },
    "required": ["overall", "per_tc", "reasoning", "violations"],
}


# ───────────────────────────────────────────────
# Context-aware must_not_contain checker
# ───────────────────────────────────────────────
# Phát hiện cụm cấm xuất hiện trong ngữ cảnh refusal/asking (bot nói "chưa có data"
# hoặc "vui lòng cung cấp X"). Nếu cụm cấm chỉ xuất hiện gần các marker này, đó là
# refusal hợp lệ, KHÔNG phải hallucination → KHÔNG override fail.
REFUSAL_MARKERS = [
    # Tiếng Việt — negation tokens cơ bản (có space để khớp như token)
    " không ", " chưa ", " chẳng ",
    # Tiếng Việt — phủ định / thiếu data (cụm dài)
    "chưa có", "chưa nhận", "chưa được", "chưa truy cập", "chưa thể",
    "chưa biết", "chưa xác", "chưa rõ", "chưa ghi nhận", "chưa tìm",
    "không có", "không thể", "không nhận", "không truy cập", "không tìm",
    "không biết", "không xác", "không rõ", "không phải", "không trùng",
    # Tiếng Việt — asking / yêu cầu cung cấp
    "vui lòng cung cấp", "sếp cung cấp", "cần cung cấp", "xin cung cấp",
    "yêu cầu cung cấp", "hãy cung cấp", "cung cấp thêm", "cung cấp lại",
    "cho tôi biết", "có thể cho", "tránh nhầm", "tránh sai", "tránh bịa",
    # English
    "don't have", "do not have", "haven't", "have not received", "no access",
    "cannot confirm", "cannot verify", "cannot determine", "unable to",
    "please provide", "could you provide", "i don't know", "without",
    " no ", " not ", " never ",
]


def _has_unsafe_assertion(text: str, phrase: str, strict: bool = False, window: int = 80) -> bool:
    """Trả True nếu `phrase` xuất hiện trong `text` như một khẳng định (hallucination).

    - strict=True (case language=en): bất kỳ occurrence nào cũng tính → dùng phát hiện
      lẫn ngôn ngữ (bot trả tiếng Việt khi user hỏi tiếng Anh).
    - strict=False (default): chỉ tính occurrence KHÔNG nằm trong window ~80 ký tự
      cạnh một refusal marker. Nếu mọi occurrence đều gần refusal marker → coi như
      bot đang refuse/asking, không phải bịa.
    """
    text_lower = text.lower()
    phrase_lower = phrase.lower()
    start = 0
    while True:
        idx = text_lower.find(phrase_lower, start)
        if idx < 0:
            return False
        if strict:
            return True
        ctx_start = max(0, idx - window)
        ctx_end = min(len(text_lower), idx + len(phrase_lower) + window)
        context = text_lower[ctx_start:ctx_end]
        if not any(m in context for m in REFUSAL_MARKERS):
            return True
        start = idx + len(phrase_lower)


def build_judge_prompt(case, output_text, system_prompt=""):
    turns = case.get("turns", [])
    convo = "\n".join(f"[{t.get('role','user').upper()}] {t.get('content','')}" for t in turns)
    expected = case.get("expected", {}) or {}
    eval_notes = expected.get("eval_notes", "(không có)")
    must_contain = expected.get("must_contain", [])
    must_not_contain = expected.get("must_not_contain", [])
    criteria = case.get("criteria", [])
    criteria_str = ", ".join(criteria) if criteria else "(Không có chỉ định cụ thể)"

    extra = ""
    if must_contain:
        extra += f"\nGỢI Ý cần có (không bắt buộc keyword, chỉ cần ý): {must_contain}"
    if must_not_contain:
        extra += (
            f"\nCỤM TỪ CẤM (chỉ tính vi phạm khi xuất hiện như KHẲNG ĐỊNH, không tính nếu "
            f"nằm trong câu phủ định/refusal/asking): {must_not_contain}"
        )

    return f"""TEST CASE: {case.get('id')} — {case.get('name')}
Use case: {case.get('use_case')}
Loại: {case.get('type','?')} | Ngôn ngữ: {case.get('language','vi')}

SYSTEM PROMPT CỦA CHATBOT THƯ KÝ (Sử dụng để đối chiếu hành vi & văn phong giao tiếp):
\"\"\"
{system_prompt}
\"\"\"

CÁC TIÊU CHÍ ÁP DỤNG CHO CASE NÀY (xem định nghĩa chi tiết trong system prompt — đánh giá nghiêm ngặt theo các chiều này):
{criteria_str}

TIÊU CHÍ THÀNH CÔNG (EVAL NOTES):
{eval_notes}{extra}

HỘI THOẠI ĐẦU VÀO:
{convo}

CÂU TRẢ LỜI CỦA MODEL CẦN ĐÁNH GIÁ (Chú ý: Bỏ qua các khối suy nghĩ <thinking>...</thinking> bên trong phản hồi nếu có khi chấm các tiêu chí về độ dài dòng, văn phong hoặc định dạng, chỉ đánh giá phần văn bản hiển thị cho người dùng):
\"\"\"
{output_text}
\"\"\"

Hãy chấm theo schema JSON."""


def judge(case, output_text, system_prompt="", retries=2):
    prompt = build_judge_prompt(case, output_text, system_prompt)
    last_err = None
    
    is_qwen = JUDGE_MODEL.startswith("qwen")
    
    for attempt in range(retries + 1):
        try:
            # Rate limit trước mỗi request để không vượt JUDGE_RPM
            _rate_limit_wait()

            if is_qwen:
                if not dashscope_judge_client:
                    raise ValueError("Thiếu DASHSCOPE_API_KEY trong .env để chạy DashScope judge.")

                completion = dashscope_judge_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=JUDGE_TEMPERATURE,
                    response_format={"type": "json_object"},
                    stream=True
                )
                
                reasoning_chunks = []
                content_chunks = []
                for chunk in completion:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        reasoning_chunks.append(delta.reasoning_content)
                    if hasattr(delta, "content") and delta.content is not None:
                        content_chunks.append(delta.content)
                        
                reasoning = "".join(reasoning_chunks)
                content = "".join(content_chunks)
                
                data = json.loads(content)
                if reasoning:
                    data["judge_thinking"] = reasoning
            else:
                if not gemini_judge_client:
                    raise ValueError("Thiếu GEMINI_API_KEY trong .env để chạy Gemini judge.")
                    
                resp = gemini_judge_client.models.generate_content(
                    model=JUDGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=JUDGE_SYSTEM,
                        temperature=JUDGE_TEMPERATURE,
                        response_mime_type="application/json",
                        response_schema=JUDGE_SCHEMA,
                    ),
                )
                data = json.loads(resp.text)
            
            # Safety net cho must_not_contain — context-aware, không naive substring.
            # Mục tiêu: bắt hallucination thật, BỎ QUA cụm cấm xuất hiện trong câu refusal.
            must_not_contain = (case.get("expected", {}) or {}).get("must_not_contain", [])
            if must_not_contain:
                # Bỏ block <thinking> trước khi check — chỉ đánh giá text hiển thị.
                clean_text = output_text
                if "<thinking>" in output_text and "</thinking>" in output_text:
                    parts = output_text.split("</thinking>")
                    clean_text = parts[-1] if len(parts) > 1 else output_text

                # Strict mode cho case yêu cầu language=en: bất kỳ cụm tiếng Việt cấm
                # nào cũng tính, không cần check refusal context.
                strict_language_check = (case.get("language") == "en")

                for forbidden in must_not_contain:
                    if _has_unsafe_assertion(clean_text, forbidden, strict=strict_language_check):
                        # Cap overall và TC liên quan xuống vùng FAIL (≤ 35).
                        data["overall"] = min(int(data.get("overall", 0)), 35)
                        per_tc = data.get("per_tc") or {}
                        violations = data.get("violations") or []
                        if strict_language_check:
                            tag = "POOR_LANGUAGE"
                            affected = "TC-05"
                            override_note = (
                                f"[Override] Case yêu cầu 100% tiếng Anh nhưng chứa cụm Việt cấm '{forbidden}'."
                            )
                        else:
                            tag = "HALLUCINATION"
                            affected = "TC-01"
                            override_note = (
                                f"[Override] Khẳng định cụm cấm '{forbidden}' ngoài ngữ cảnh refusal — dấu hiệu bịa."
                            )
                        if affected in per_tc:
                            entry = per_tc[affected]
                            if isinstance(entry, dict):
                                entry["score"] = min(int(entry.get("score", 0)), 35)
                                entry["note"] = override_note
                            else:
                                per_tc[affected] = {"score": 35, "note": override_note}
                        data["per_tc"] = per_tc
                        if tag not in violations:
                            violations.append(tag)
                        data["violations"] = violations
                        data["reasoning"] = override_note
                        break

            # Dedup violations để report sạch
            if data.get("violations"):
                seen = set()
                deduped = []
                for v in data["violations"]:
                    if v not in seen:
                        seen.add(v)
                        deduped.append(v)
                data["violations"] = deduped

            # Sanitize: clamp overall & per_tc về [0,100]; chỉ giữ TC nằm trong criteria của case.
            # per_tc giờ là dict {TC: {"score": int, "note": str}}.
            overall = max(0, min(100, int(data.get("overall", 0))))
            data["overall"] = overall
            allowed = set(case.get("criteria", []) or [])
            raw_per_tc = data.get("per_tc") or {}
            cleaned_per_tc = {}
            for tc, v in raw_per_tc.items():
                if tc not in allowed:
                    continue
                if isinstance(v, dict):
                    try:
                        score = max(0, min(100, int(v.get("score", 0))))
                    except (ValueError, TypeError):
                        continue
                    note = str(v.get("note", "")).strip()
                    cleaned_per_tc[tc] = {"score": score, "note": note}
                else:
                    # Backward-compat: nếu judge trả về int (schema cũ).
                    try:
                        cleaned_per_tc[tc] = {"score": max(0, min(100, int(v))), "note": ""}
                    except (ValueError, TypeError):
                        continue
            data["per_tc"] = cleaned_per_tc
            data["passed"] = overall >= PASS_THRESHOLD

            return data
        except Exception as e:
            last_err = e
            if attempt < retries:
                # Nếu là 429 / RESOURCE_EXHAUSTED → sleep ít nhất 60s để reset cửa sổ RPM
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str:
                    wait = 65.0
                    print(f" [429: sleep {wait:.0f}s để reset RPM] ", end="", flush=True)
                    _judge_call_times.clear()
                else:
                    wait = 2 * (attempt + 1)
                time.sleep(wait)
    return {
        "passed": False,
        "overall": 0,
        "per_tc": {},
        "reasoning": f"Judge error: {last_err}",
        "violations": ["OTHER"],
    }


# ───────────────────────────────────────────────
# Main loop
# ───────────────────────────────────────────────
def evaluate_case(case, system_prompt=""):
    messages = list(case.get("turns", []))
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    thinking_mode = case.get("thinking_mode", False) or case.get("thinking_compare", False)

    res, latency = _call_target(messages, thinking_mode)

    output_text = res.get("text", "")
    verdict = judge(case, output_text, system_prompt)

    return {
        "id": case.get("id"),
        "name": case.get("name"),
        "use_case": case.get("use_case"),
        "criteria": case.get("criteria", []),
        "passed": bool(verdict.get("passed")),
        "score": int(verdict.get("score", 0)),
        "reasoning": verdict.get("reasoning", ""),
        "violations": verdict.get("violations", []),
        "judge_thinking": verdict.get("judge_thinking", ""),
        "latency": latency,
        "prompt_tokens": res.get("prompt_tokens", 0),
        "completion_tokens": res.get("completion_tokens", 0),
        "output": output_text,
        "turns": case.get("turns", []),
        "eval_notes": (case.get("expected", {}) or {}).get("eval_notes", ""),
    }


def _call_target(messages, thinking_mode):
    """Gọi target Modal — chỉ SDK hoặc HTTP endpoint, không còn DashScope."""
    if MODE == "http":
        return call_model_http(messages, thinking_mode)
    return call_model_sdk(messages, thinking_mode)


def generate_responses(test_cases, system_prompt):
    print(f"Giai doan GENERATE: Sinh cau tra loi cho {len(test_cases)} test cases...")
    print(f"  Target: Modal {MODE.upper()} | Model: {TARGET_MODEL_NAME}\n")

    # `_meta` lưu tên model + mode vào file output để biết câu trả lời được sinh
    # bởi model nào. Khi judge stage đọc lại sẽ skip key này.
    outputs = {
        "_meta": {
            "model": TARGET_MODEL_NAME,
            "mode": MODE,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    for i, case in enumerate(test_cases, 1):
        cid = case.get("id")
        cname = case.get("name", "")
        n_runs = max(1, int(case.get("rerun", 1)))
        suffix = f" x{n_runs}" if n_runs > 1 else ""
        print(f"[{i}/{len(test_cases)}] {TARGET_MODEL_NAME} via {MODE.upper()} -> {cid}{suffix}...", end="", flush=True)
        try:
            messages = list(case.get("turns", []))
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

            thinking_mode = case.get("thinking_mode", False) or case.get("thinking_compare", False)

            if n_runs == 1:
                res, latency = _call_target(messages, thinking_mode)
                outputs[cid] = {
                    "model": TARGET_MODEL_NAME,
                    "output": res.get("text", ""),
                    "latency": latency,
                    "prompt_tokens": res.get("prompt_tokens", 0),
                    "completion_tokens": res.get("completion_tokens", 0),
                }
                print(f" Done ({latency:.1f}s)")
            else:
                # Multi-run case (phục vụ TC-09 Consistency) — chạy N lần, lưu list runs.
                runs = []
                for k in range(n_runs):
                    res, latency = _call_target(messages, thinking_mode)
                    runs.append({
                        "output": res.get("text", ""),
                        "latency": latency,
                        "prompt_tokens": res.get("prompt_tokens", 0),
                        "completion_tokens": res.get("completion_tokens", 0),
                    })
                outputs[cid] = {"model": TARGET_MODEL_NAME, "runs": runs}
                total_lat = sum(r["latency"] for r in runs)
                print(f" Done ({n_runs} runs, total {total_lat:.1f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
            outputs[cid] = {
                "model": TARGET_MODEL_NAME,
                "output": "",
                "latency": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e)
            }

    with open(OUTPUTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Da luu cau tra loi vao {OUTPUTS_JSON_PATH}\n")


def _compute_per_tc_scores(results):
    """Aggregate điểm 0-100 trung bình mỗi TC.

    Ưu tiên dùng `per_tc` breakdown của từng case (chính xác); chỉ fallback sang
    `overall` nếu case không có per_tc cho TC đó.
    Trả về dict {tc: {"avg": float, "count": int, "pass_rate": float}} theo TC-01..TC-09.
    """
    bucket = {}  # tc → list[int 0-100]
    for r in results:
        per_tc = r.get("per_tc") or {}
        for tc in r.get("criteria", []) or []:
            entry = per_tc.get(tc)
            if isinstance(entry, dict) and "score" in entry:
                v = int(entry["score"])
            elif isinstance(entry, int):
                v = entry
            else:
                v = int(r.get("overall", 0) or 0)
            bucket.setdefault(tc, []).append(v)
    summary = {}
    for tc in sorted(CRITERIA_DEFINITIONS.keys()):
        scores = bucket.get(tc, [])
        if scores:
            passes = sum(1 for s in scores if s >= PASS_THRESHOLD)
            summary[tc] = {
                "avg": sum(scores) / len(scores),
                "count": len(scores),
                "pass_rate": passes / len(scores) * 100,
            }
    return summary


def write_markdown_report(test_cases, results, passed_count, total_latency, total_prompt, total_completion, total_score, target_meta=None):
    n = len(test_cases) or 1
    avg_latency = total_latency / n
    avg_overall = total_score / n
    target_meta = target_meta or {}
    pass_pct = passed_count / n * 100
    production_count = sum(1 for r in results if r.get("overall", 0) >= PRODUCTION_THRESHOLD)
    production_pct = production_count / n * 100

    per_tc = _compute_per_tc_scores(results)

    print("\n" + "=" * 60)
    print("KET QUA LLM-AS-JUDGE (Thang 0-100)")
    print("=" * 60)
    print(f"Tong test cases    : {n}")
    print(f"PASS (>= {PASS_THRESHOLD})       : {passed_count} ({pass_pct:.1f}%)")
    print(f"Production (>= {PRODUCTION_THRESHOLD}): {production_count} ({production_pct:.1f}%)")
    print(f"FAIL               : {n - passed_count}")
    print(f"Overall TB         : {avg_overall:.1f}/100")
    print(f"Latency TB         : {avg_latency:.2f}s")
    print(f"Prompt tokens      : {total_prompt}")
    print(f"Completion tokens  : {total_completion}")
    print(f"Target model       : {target_meta.get('model', TARGET_MODEL_NAME)} "
          f"(Modal {target_meta.get('mode', MODE).upper()})")
    print(f"Judge model        : {JUDGE_MODEL}")
    print("-" * 60)
    print("Diem TB theo tieu chi (xlsx v3, thang 0-100):")
    for tc, s in per_tc.items():
        print(f"  {tc}: {s['avg']:5.1f}/100  pass={s['pass_rate']:5.1f}%  (n={s['count']})")
    print("=" * 60)

    report_path = Path(__file__).parent / "eval_report_judge.md"
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Bao cao Danh gia LLM-as-Judge\n\n")
        rf.write(f"* **Thoi gian**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        target_model = target_meta.get("model", TARGET_MODEL_NAME)
        target_mode_str = target_meta.get("mode", MODE).upper()
        gen_at = target_meta.get("generated_at", "?")
        rf.write(f"* **Target model**: `{target_model}` (Modal {target_mode_str}, generated {gen_at})\n")
        rf.write(f"* **Judge model**: `{JUDGE_MODEL}`\n")
        rf.write(f"* **Nguon tieu chi**: `Chatbot_ThuKy_UseCases_EvalCriteria_v3.xlsx` (sheet 'Tieu chi danh gia')\n")
        rf.write(f"* **Thang diem**: 0-100 (PASS >= {PASS_THRESHOLD}, Production-ready >= {PRODUCTION_THRESHOLD})\n")
        rf.write(f"* **Test cases**: {n}\n")
        rf.write(f"* **PASS**: **{passed_count}/{n} ({pass_pct:.1f}%)**\n")
        rf.write(f"* **Production-ready**: **{production_count}/{n} ({production_pct:.1f}%)**\n")
        rf.write(f"* **Overall trung binh**: **{avg_overall:.1f}/100**\n")
        rf.write(f"* **Latency TB**: {avg_latency:.2f}s\n")
        rf.write(f"* **Tokens**: In {total_prompt} | Out {total_completion}\n\n")

        rf.write("## Diem trung binh theo tung tieu chi (xlsx v3)\n\n")
        rf.write("> Lay trung binh cua `per_tc` breakdown qua tat ca case co khai bao TC do trong `criteria`. "
                 "Xlsx khong khai bao trong so nen khong tinh tong co weight.\n\n")
        rf.write("| TC | Ten tieu chi | So case | Diem TB | Pass rate |\n")
        rf.write("|---|---|---|---|---|\n")
        for tc, s in per_tc.items():
            short = CRITERIA_DEFINITIONS[tc].split(":")[0]
            rf.write(
                f"| `{tc}` | {short} | {s['count']} | {s['avg']:.1f}/100 | "
                f"{s['pass_rate']:.1f}% |\n"
            )
        rf.write("\n")

        def _per_tc_short(per_tc):
            """Hiển thị 1 dòng: 'TC-02=80, TC-06=60' (chỉ score, dùng cho bảng tổng)."""
            parts = []
            for tc, v in (per_tc or {}).items():
                if isinstance(v, dict):
                    parts.append(f"{tc}={v.get('score', '?')}")
                else:
                    parts.append(f"{tc}={v}")
            return ", ".join(parts) or "-"

        rf.write("## Bang tong hop\n\n")
        rf.write("| ID | Use Case | Test | Trang thai | Overall | Per-TC | Latency | Tom tat |\n")
        rf.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            if r["overall"] >= PRODUCTION_THRESHOLD:
                status = "PROD"
            elif r["passed"]:
                status = "PASS"
            else:
                status = "FAIL"
            reason = r["reasoning"].replace("\n", " ").replace("|", "\\|")[:120]
            rf.write(
                f"| `{r['id']}` | `{r['use_case']}` | {r['name']} | {status} | "
                f"{r['overall']}/100 | {_per_tc_short(r.get('per_tc'))} | "
                f"{r['latency']:.2f}s | {reason} |\n"
            )

        rf.write("\n## Chi tiet cac case FAIL\n\n")
        fails = [r for r in results if not r["passed"]]
        if not fails:
            rf.write("Khong co FAIL.\n\n")
        else:
            for fc in fails:
                rf.write(f"### FAIL `{fc['id']}` - {fc['name']} (`{fc['use_case']}`)\n\n")
                rf.write(f"* **Overall**: {fc['overall']}/100\n")
                rf.write(f"* **Per-TC**:\n")
                for tc, v in (fc.get('per_tc') or {}).items():
                    if isinstance(v, dict):
                        rf.write(f"  * `{tc}`: **{v.get('score', '?')}** — {v.get('note', '')}\n")
                    else:
                        rf.write(f"  * `{tc}`: **{v}**\n")
                rf.write(f"* **Vi pham**: {fc.get('violations', [])}\n")
                rf.write(f"* **Tom tat**: {fc['reasoning']}\n")
                rf.write(f"* **Eval notes**: {fc['eval_notes']}\n")
                rf.write(f"* **Output**:\n```text\n{fc['output']}\n```\n\n")

        rf.write("## Chi tiet toan bo responses\n\n")
        for r in results:
            sym = "PASS" if r["passed"] else "FAIL"
            rf.write(f"### `{r['id']}` - {r['name']} ({sym} -- {r['overall']}/100)\n\n")
            rf.write(
                f"* **UC**: `{r['use_case']}` | **Criteria**: {r['criteria']} | "
                f"**Latency**: {r['latency']:.2f}s | **Tokens**: In {r['prompt_tokens']} / Out {r['completion_tokens']}\n"
            )
            rf.write("* **Per-TC**:\n")
            per_tc = r.get('per_tc') or {}
            if per_tc:
                for tc, v in per_tc.items():
                    if isinstance(v, dict):
                        rf.write(f"  * `{tc}`: **{v.get('score', '?')}/100** — {v.get('note', '')}\n")
                    else:
                        rf.write(f"  * `{tc}`: **{v}/100**\n")
            else:
                rf.write("  * (judge không trả per_tc breakdown)\n")
            rf.write("* **Hoi thoai**:\n")
            for t in r.get("turns", []):
                role = t.get("role", "user").capitalize()
                content = t.get("content", "").replace("\n", "\n  ")
                rf.write(f"  * **{role}**: {content}\n")
            rf.write(f"* **Model tra loi**:\n```text\n{r['output']}\n```\n")
            rf.write(f"* **Tom tat judge**: {r['reasoning']}\n")
            if r.get("judge_thinking"):
                rf.write(f"* **Suy nghi cua Judge**:\n```text\n{r['judge_thinking']}\n```\n")
            if r.get("violations"):
                rf.write(f"* **Vi pham**: {r['violations']}\n")
            rf.write(f"* **Eval notes**: {r['eval_notes']}\n\n---\n\n")

    print(f"\nBao cao: {report_path}")


def run_judge_stage(test_cases, system_prompt):
    if not OUTPUTS_JSON_PATH.exists():
        print(f"Loi: Khong tim thay tep {OUTPUTS_JSON_PATH}. Hay chay stage 'generate' truoc!")
        sys.exit(1)

    print(f"Doc cau tra loi da sinh tu {OUTPUTS_JSON_PATH}")
    with open(OUTPUTS_JSON_PATH, "r", encoding="utf-8") as f:
        outputs = json.load(f)

    # Hiển thị model nào đã sinh ra batch output này.
    meta = outputs.get("_meta") or {}
    if meta:
        print(f"  -> Output sinh bởi: {meta.get('model','?')} ({meta.get('mode','?')}) "
              f"luc {meta.get('generated_at','?')}")

    print(f"Giai doan JUDGE: Cham diem bang {JUDGE_MODEL}...")
    results = []
    passed_count = 0
    total_latency = 0.0
    total_prompt = 0
    total_completion = 0
    total_score = 0
    
    for i, case in enumerate(test_cases, 1):
        cid = case.get("id")
        cname = case.get("name", "")
        print(f"[{i}/{len(test_cases)}] Judge {JUDGE_MODEL} -> {cid}...", end="", flush=True)

        case_output = outputs.get(cid, {})

        if case_output.get("error"):
            print(f" SKIP due to target model error: {case_output['error']}")
            results.append({
                "id": cid, "name": cname, "use_case": case.get("use_case"),
                "passed": False, "overall": 0, "per_tc": {},
                "reasoning": f"Original execution error: {case_output['error']}",
                "violations": ["execution_error"],
                "latency": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                "output": "", "turns": case.get("turns", []),
                "eval_notes": (case.get("expected", {}) or {}).get("eval_notes", ""),
                "criteria": case.get("criteria", []),
            })
            continue

        # Phân nhánh single-run vs multi-run (TC-09 Consistency).
        is_multi = "runs" in case_output
        if is_multi:
            runs = case_output["runs"]
            sub_verdicts = []
            sub_errs = []
            for k, run in enumerate(runs):
                try:
                    v = judge(case, run.get("output", ""), system_prompt)
                    sub_errs.append(None)
                except Exception as e:
                    v = {"passed": False, "overall": 0, "per_tc": {},
                         "reasoning": f"Judge execution error (run {k+1}): {e}",
                         "violations": ["OTHER"], "judge_thinking": ""}
                    sub_errs.append(e)
                sub_verdicts.append(v)

            overalls = [int(v.get("overall", 0)) for v in sub_verdicts]
            min_overall = min(overalls)
            max_overall = max(overalls)
            variance = max_overall - min_overall

            # Aggregate verdict: dùng run có điểm thấp nhất làm verdict chính.
            worst_idx = overalls.index(min_overall)
            verdict = dict(sub_verdicts[worst_idx])
            base_reason = verdict.get("reasoning", "")
            verdict["reasoning"] = (
                f"[Multi-run x{len(runs)} | overall={overalls}, variance={variance}] {base_reason}"
            )

            # TC-09 Consistency penalty: variance ≥ 20 (trên scale 0-100) → trừ thêm.
            if "TC-09" in case.get("criteria", []) and variance >= 20:
                penalty = min(20, variance // 2)
                penalized = max(0, min_overall - penalty)
                verdict["overall"] = penalized
                verdict["passed"] = penalized >= PASS_THRESHOLD
                # Cap TC-09 per_tc xuống vùng FAIL.
                per_tc = dict(verdict.get("per_tc") or {})
                tc09_note = f"Multi-run variance={variance} → các lần chạy không nhất quán."
                if "TC-09" in per_tc and isinstance(per_tc["TC-09"], dict):
                    per_tc["TC-09"] = {
                        "score": min(int(per_tc["TC-09"].get("score", 0)), 40),
                        "note": tc09_note,
                    }
                else:
                    per_tc["TC-09"] = {"score": 40, "note": tc09_note}
                verdict["per_tc"] = per_tc
                violations = list(verdict.get("violations") or [])
                if "OTHER" not in violations:
                    violations.append("OTHER")
                verdict["violations"] = violations
                verdict["reasoning"] = (
                    f"[TC-09 inconsistency penalty: variance={variance} ≥ 20 → -{penalty}] "
                    + verdict["reasoning"]
                )

            judge_err = next((e for e in sub_errs if e), None)

            # Aggregate latency/tokens
            latency = sum(r.get("latency", 0.0) for r in runs) / len(runs)
            p_tokens = sum(r.get("prompt_tokens", 0) for r in runs)
            c_tokens = sum(r.get("completion_tokens", 0) for r in runs)
            output_text = "\n\n--- RUN SEPARATOR ---\n\n".join(
                f"[Run {k+1}/{len(runs)} | overall={overalls[k]}]\n{r.get('output','')}"
                for k, r in enumerate(runs)
            )
        else:
            output_text = case_output.get("output", "")
            latency = case_output.get("latency", 0.0)
            p_tokens = case_output.get("prompt_tokens", 0)
            c_tokens = case_output.get("completion_tokens", 0)

            try:
                verdict = judge(case, output_text, system_prompt)
                judge_err = None
            except Exception as e:
                verdict = {"passed": False, "overall": 0, "per_tc": {},
                           "reasoning": f"Judge execution error: {e}",
                           "violations": ["OTHER"], "judge_thinking": ""}
                judge_err = e

        r = {
            "id": cid,
            "name": cname,
            "use_case": case.get("use_case"),
            "criteria": case.get("criteria", []),
            "passed": bool(verdict.get("passed")),
            "overall": int(verdict.get("overall", 0)),
            "per_tc": dict(verdict.get("per_tc") or {}),
            "reasoning": verdict.get("reasoning", ""),
            "violations": verdict.get("violations", []),
            "judge_thinking": verdict.get("judge_thinking", ""),
            "latency": latency,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "output": output_text,
            "turns": case.get("turns", []),
            "eval_notes": (case.get("expected", {}) or {}).get("eval_notes", ""),
        }
        results.append(r)
        total_latency += latency
        total_prompt += p_tokens
        total_completion += c_tokens
        total_score += r["overall"]

        # Print log — bọc try/except để encoding crash không phá pipeline
        try:
            if judge_err:
                print(f" ERROR: {judge_err}")
            elif r["passed"]:
                passed_count += 1
                print(f" PASS (overall={r['overall']}/100)")
            else:
                print(f" FAIL (overall={r['overall']}/100) -- {r['reasoning'][:80]}")
        except UnicodeEncodeError:
            if not judge_err and r["passed"]:
                passed_count += 1
            print(f" [overall={r['overall']}/100 — log unicode skipped]")
            
    write_markdown_report(test_cases, results, passed_count, total_latency, total_prompt, total_completion, total_score, target_meta=meta)


def main():
    yaml_path = Path(__file__).parent / "test_cases.yaml"
    if not yaml_path.exists():
        print(f"Khong tim thay {yaml_path}")
        sys.exit(1)

    print(f"Doc test cases tu {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    test_cases = data.get("test_cases", [])
    system_prompt = data.get("system_prompt", "")
    
    print(f"STAGE: {STAGE.upper()} | Target: {MODE.upper()} | Judge: {JUDGE_MODEL} | Total cases: {len(test_cases)}\n")

    if STAGE == "generate":
        generate_responses(test_cases, system_prompt)
    elif STAGE == "judge":
        run_judge_stage(test_cases, system_prompt)
    else:
        # Chạy "all"
        generate_responses(test_cases, system_prompt)
        run_judge_stage(test_cases, system_prompt)


if __name__ == "__main__":
    main()
