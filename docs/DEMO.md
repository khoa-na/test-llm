# Demo transcripts

Real responses captured from the deployed service (`Qwen/Qwen3.5-9B`, `temperature=0.2`, server-side retrieval on, `python_exec` tool enabled). The bot answers in Vietnamese; English captions added for context. The reference "today" is anchored to `2026-05-27` for the structured corpus.

---

## 1. Calendar — structured retrieval

**User:** `lịch chiều nay có gì` *(what is on my calendar this afternoon)*

**Assistant:**
```
Dựa trên dữ liệu lịch đã cung cấp, chiều nay (2026-05-27) có 2 sự việc:

1. 14:00 – 15:00: Họp ngân sách Q3 tại phòng họp A (Thành phần: anh Nam - CFO, chị Lan - Kế toán trưởng).
2. 16:00 – 16:30: Call khách hàng X (Thành phần: Sales).
```
**Retrieved:** `calendar/2026-05-27/14:00`, `calendar/2026-05-27/16:00`
*Time-of-day filtering: morning events are correctly excluded.*

---

## 2. Document RAG — semantic retrieval

**User:** `lợi nhuận sau thuế quý 1 năm nay là bao nhiêu` *(what was Q1 net profit after tax)*

**Assistant:**
```
Lợi nhuận sau thuế quý 1/2026 là 5.84 tỷ đồng.
```
**Retrieved:** `docs/bao_cao_tai_chinh_q1_2026.md#5. Kết luận`, `#1. Kết quả kinh doanh`
*Pulls the exact figure from the financial report; does not confuse it with revenue (48.2) or pre-tax profit (7.3).*

---

## 3. RAG + tool-calling

**User:** `tỷ suất lợi nhuận sau thuế trên doanh thu quý 1 là bao nhiêu phần trăm` *(what is the Q1 net profit margin, in percent)*

**Assistant:**
```
Tỷ suất lợi nhuận sau thuế trên doanh thu quý 1/2026 là 12,12%.
```
**Retrieved:** `docs/bao_cao_tai_chinh_q1_2026.md#5. Kết luận`, `#1. Kết quả kinh doanh`

**Tool call (`python_exec`, run in an isolated Modal Sandbox):**
```python
doanh_thu_q1 = 48.2  # tỷ
loi_nhuan_sau_thue_q1 = 5.84  # tỷ
ty_suat = (loi_nhuan_sau_thue_q1 / doanh_thu_q1) * 100
print(f"{ty_suat:.2f}")   # => 12.12
```
*The model retrieves the two figures from the document, then computes the ratio deterministically via the tool instead of guessing.*

---

## 4. Refusal — no hallucination

**User:** `email từ chị Hương phòng nhân sự gửi hôm qua nói gì` *(what did the email from Ms. Huong in HR say yesterday)*

**Assistant:**
```
Chưa có dữ liệu email cụ thể từ chị Hương trong hệ thống. Tôi chưa có quyền truy cập
vào hộp thư hoặc lịch sử trao đổi để xem nội dung chi tiết.

Bạn vui lòng cung cấp nội dung email hoặc tóm tắt những điểm chính chị Hương đã đề cập
để tôi hỗ trợ xử lý tiếp.
```
**Retrieved:** *(none)*
*There is no such sender in the corpus, so retrieval returns nothing and the model refuses instead of fabricating content.*
