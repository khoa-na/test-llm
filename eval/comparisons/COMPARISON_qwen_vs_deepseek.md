# So sánh model: Qwen3.5-9B vs DeepSeek-R1-0528-Qwen3-8B

> **Mục đích:** Đánh giá DeepSeek-R1-0528-Qwen3-8B (reasoning specialist trong shortlist v3) trên cùng pipeline, cùng judge, cùng config với Qwen3.5-9B (ứng viên #1 đã có kết quả).
>
> **Điều kiện so sánh — giữ NGUYÊN, chỉ đổi `MODEL_NAME`:**
> - Judge: `gemini-3.1-flash-lite`, `JUDGE_TEMPERATURE=0.0`, `JUDGE_SEED=42`
> - Target: `EVAL_TEMPERATURE=0.2`, `EVAL_MAX_TOKENS=512`, L40S, BF16, `MAX_MODEL_LEN=65536`
> - Tool `python_exec` BẬT, RAG BẬT cho set `rag_live`
> - Cùng bộ test, cùng corpus, neo `RAG_REFERENCE_DATE=2026-05-27`
> - Branch: `eval/deepseek-r1`

---

## 1. Tổng quan — hai bộ test

### Bộ CHAT (63 case)

| Chỉ số | Qwen3.5-9B | DeepSeek-R1-8B | Chênh |
|---|---|---|---|
| **PASS (≥60)** | 61/63 (96.8%) | **63/63 (100%)** | +2 case |
| **Production-ready (≥75)** | 59/63 (93.7%) | **62/63 (98.4%)** | +3 case |
| **FAIL** | 2 | **0** | −2 |
| **Overall TB** | 92.9/100 | **95.6/100** | +2.7 |
| Latency TB | **8.10s** | 9.07s | +0.97s |
| Tokens (In/Out) | 152400 / 12030 | 153497 / **11887** | ~ngang |

### Bộ RAG_LIVE (11 case — retrieval thật server-side)

| Chỉ số | Qwen3.5-9B | DeepSeek-R1-8B | Chênh |
|---|---|---|---|
| **PASS (≥60)** | 10/11 (90.9%) | 10/11 (90.9%) | = |
| **Production-ready (≥75)** | 9/11 (81.8%) | **10/11 (90.9%)** | +1 case |
| **FAIL** | 1 (RL04) | 1 (RL04 — *cùng case*) | = |
| **Overall TB** | 91.4/100 | **92.3/100** | +0.9 |
| Latency TB | 21.44s | **16.66s** | −4.78s |

---

## 2. Điểm trung bình theo từng tiêu chí (thang 0-100)

### CHAT

| TC | Tiêu chí | Qwen | DeepSeek | Ghi chú |
|---|---|---|---|---|
| TC-01 | Accuracy | 97.0 (96.4%) | **97.5 (100%)** | DeepSeek nhỉnh |
| TC-02 | Intent Recognition | 96.9 (97.1%) | **98.2 (100%)** | DeepSeek nhỉnh |
| TC-03 | Multi-turn | 96.7 (100%) | **100.0 (100%)** | DeepSeek nhỉnh |
| TC-04 | Language Quality | 93.3 (100%) | 93.3 (100%) | ngang |
| TC-05 | Multilingual VI/EN | 100.0 (100%) | 100.0 (100%) | ngang |
| TC-06 | Temporal Reasoning | 95.0 (100%) | **97.5 (100%)** | DeepSeek nhỉnh |
| TC-08 | **Robustness** | 86.5 (94.1%) | **93.5 (100%)** | **DeepSeek vượt rõ (+7.0)** |
| TC-09 | Consistency | 90.0 (100%) | 90.0 (100%) | ngang |

### RAG_LIVE

| TC | Tiêu chí | Qwen | DeepSeek | Ghi chú |
|---|---|---|---|---|
| TC-01 | Accuracy | 92.7 (90.9%) | **93.6 (90.9%)** | DeepSeek nhỉnh |
| TC-02 | Intent Recognition | 93.3 (88.9%) | **94.4 (88.9%)** | DeepSeek nhỉnh |
| TC-04 | Language Quality | 100.0 | 100.0 | ngang |
| TC-06 | Temporal Reasoning | 98.3 | **100.0** | DeepSeek nhỉnh |
| TC-08 | Robustness | **100.0** (n=2) | 95.0 (n=2) | Qwen nhỉnh (lệch 1 case) |

> **Composite có trọng số (xlsx v3, chỉ trên 8 TC được judge chấm — bỏ TC-07 latency & TC-10 cost vì đo riêng), bộ chat:**
> Qwen ≈ **95.3** · DeepSeek ≈ **97.0**

---

## 3. Phân tích case quan trọng

### Qwen FAIL (2 case, chat) — DeepSeek đều PASS
- **T08.v1** (UC-14, phát hiện xung đột lịch) — Qwen 45/100: sai logic phân tích trùng giờ. **DeepSeek PASS.**
- **RB.v4** (robustness, input sai chính tả nặng + viết tắt) — Qwen 20/100 (TC-08=0). **DeepSeek PASS.**
- → Đây là nguồn gốc khoảng cách TC-08 Robustness: model reasoning xử lý input nhiễu/edge-case tốt hơn.

### RL04 — cả hai cùng FAIL (35/100), KHÔNG phân biệt model
Câu hỏi: *"task nào đang trễ hạn"*. Cả hai liệt kê 2 task, **bao gồm "Gửi proposal khách Y" (đã xong)** dưới nhãn "đang trễ hạn" → vi phạm logic lọc + kích hoạt override cụm cấm `proposal`.
- DeepSeek diễn giải cẩn thận hơn ("đã xong... trễ nhưng đã hoàn thành", "việc khác chưa đến hạn") nhưng vẫn xếp nhầm vào nhóm trễ.
- **Bản chất:** điểm yếu logic CHUNG của cả 2 model trên case này (hoặc override hơi gắt). Cần xem lại nếu muốn nâng cả hai.

### T12.v1 (DeepSeek, chat) — outlier 49.63s, trait reasoning model
Sắp xếp 8 task theo ưu tiên. DeepSeek **rò văn phong suy luận tiếng Anh thẳng vào câu trả lời** ("Interesting - the current date... Let me recalculate. Actually...") — KHÔNG bọc trong tag `<think>` nên framework không strip được (framework chỉ strip `<thinking>`). Out 1756 tok (cao nhất bộ), 49.6s. Vẫn PASS 65/100 nhưng:
- Đây là rủi ro đặc trưng của DeepSeek-R1: verbose, đôi khi lộ reasoning + drift tiếng Anh ở task suy luận nhiều bước.
- Toàn bộ 63 output chat: **0 case có tag `<think>`** — model nói chung emit sạch, nhưng case nặng reasoning có thể leak prose.

---

## 4. Kết luận

**DeepSeek-R1-0528-Qwen3-8B (8B) ngang bằng hoặc nhỉnh hơn Qwen3.5-9B (9B) trên bộ eval thư ký này**, dù nhỏ hơn 1B tham số:

✅ **Mạnh hơn:**
- Pass/Production rate cao hơn cả 2 bộ (chat: 0 FAIL vs 2; production 98.4% vs 93.7%).
- Overall cao hơn (chat 95.6 vs 92.9; rag_live 92.3 vs 91.4).
- **Robustness (TC-08) vượt rõ** — xử lý input nhiễu, edge-case, phát hiện xung đột tốt hơn.
- Temporal reasoning nhỉnh hơn ở cả 2 bộ.
- Latency rag_live **thấp hơn** (16.7s vs 21.4s).

⚠️ **Đánh đổi / rủi ro:**
- Latency chat nhỉnh hơn một chút (9.1s vs 8.1s) và **variance cao** — 1 outlier 49.6s ở task reasoning nhiều bước.
- **Rò văn phong suy luận + drift tiếng Anh** ở task nặng reasoning (T12.v1); framework strip `<thinking>` nhưng DeepSeek leak prose không tag → cân nhắc thêm hậu xử lý nếu lên production.
- Theo doc: tool-calling **không hoạt động trong thinking mode** — eval này chạy thinking_mode=False nên chưa chạm giới hạn đó; cần lưu ý cho UC-18 (workflow tool-calling).

🟰 **Chung (không phân biệt):** RL04 — cả hai sai logic lọc task trễ hạn.

**Khuyến nghị:** DeepSeek-R1-8B là ứng viên rất cạnh tranh, đặc biệt cho các use case cần robustness/reasoning (UC-06, UC-12, UC-14). Trước khi thay Qwen làm primary, cần xử lý 2 điểm: (1) chặn leak reasoning-prose/drift EN ở output, (2) xác nhận tool-calling cho UC-18 (thinking mode trade-off).

---

*Sinh từ branch `eval/deepseek-r1`. Report gốc: `eval/results/judge/eval_report_judge__{chat,rag_live}__*.md`.*
