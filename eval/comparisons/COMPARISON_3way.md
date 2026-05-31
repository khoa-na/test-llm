# So sánh 3 model: Gemma 4-E4B vs DeepSeek-R1-8B vs Qwen3.5-9B

> Cùng pipeline, cùng judge `gemini-3.1-flash-lite` (temp 0, seed 42), cùng config
> (EVAL_TEMPERATURE=0.2, MAX_TOKENS=512, L40S, BF16), chỉ khác `MODEL_NAME`.
> rag_live: cả 3 chạy lại trên bộ 34 case ĐÃ-FIX (apples-to-apples). chat: 63 case.
> Branch: `eval/gemma-3way`.

## 1. Tổng quan từng bộ

### CHAT (63 case)
| Model | PASS | Production (≥75) | Overall | Latency |
|---|---|---|---|---|
| **DeepSeek-R1-8B** | **63/63 (100%)** | **62/63 (98.4%)** | **95.6** | 9.07s |
| **Gemma 4-E4B** | 61/63 (96.8%) | 59/63 (93.7%) | 94.3 | **7.67s** |
| **Qwen3.5-9B** | 61/63 (96.8%) | 59/63 (93.7%) | 92.9 | 8.10s |

### RAG_LIVE (34 case, đã-fix)
| Model | PASS | Production (≥75) | Overall | Latency | FAIL |
|---|---|---|---|---|---|
| **Gemma 4-E4B** | **33/34 (97.1%)** | **31/34 (91.2%)** | **94.1** | **9.23s** | RL31 |
| **DeepSeek-R1-8B** | 32/34 (94.1%) | 31/34 (91.2%) | 93.1 | 9.40s | RL14, RL31 |
| **Qwen3.5-9B** | 31/34 (91.2%) | 29/34 (85.3%) | 90.7 | 10.25s | RL12, RL18, RL31 |

## 2. Gộp cả 2 bộ (97 case)
| Model | PASS | Production | Overall (trọng số) |
|---|---|---|---|
| **DeepSeek-R1-8B** | **95/97 (97.9%)** | **93/97 (95.9%)** | **94.7** |
| **Gemma 4-E4B** | 94/97 (96.9%) | 90/97 (92.8%) | 94.2 |
| **Qwen3.5-9B** | 92/97 (94.8%) | 88/97 (90.7%) | 92.1 |

## 3. Điểm theo tiêu chí (0-100) — Gemma / DeepSeek / Qwen

| TC | Bộ | Gemma | DeepSeek | Qwen | Tốt nhất |
|---|---|---|---|---|---|
| TC-01 Accuracy | chat | **99.3** | 97.5 | 97.0 | Gemma |
| TC-01 Accuracy | rag | **94.4** | 93.7 | 91.0 | Gemma |
| TC-02 Intent | chat | 96.0 | **98.2** | 96.9 | DeepSeek |
| TC-02 Intent | rag | 97.5 | **97.7** | 96.4 | DeepSeek |
| TC-03 Multi-turn | chat | 99.2 | **100** | 96.7 | DeepSeek |
| TC-04 Language | chat | **95.0** | 93.3 | 93.3 | Gemma |
| TC-05 Multilingual | chat | 100 | 100 | 100 | = |
| TC-06 Temporal | chat | **97.5** | **97.5** | 95.0 | Gemma/DS |
| TC-06 Temporal | rag | 88.9 | **90.6** | 84.4 | DeepSeek |
| TC-08 Robustness | chat | 92.4 | **93.5** | 86.5 | DeepSeek |
| TC-08 Robustness | rag | **94.0** | 80.0 | **94.0** | Gemma/Qwen |
| TC-09 Consistency | chat | 70.0 | **90.0** | **90.0** | DS/Qwen |

## 4. Đặc điểm & khác biệt thực chất

**Gemma 4-E4B (~4.5B effective — NHỎ nhất):** gây bất ngờ.
- ✅ **Accuracy cao nhất** (TC-01 chat 99.3, rag 94.4) và **nhanh nhất** (chat 7.67s).
- ✅ **Thắng bộ rag_live** (94.1, chỉ 1 FAIL) — trích xuất doc/structured tốt.
- ⚠️ **Consistency yếu** (TC-09 chat 70, pass 50%): `CST.v2` variance 25 giữa các lần chạy → kém ổn định khi hỏi lặp.
- ⚠️ chat `VRB.v2`: khuyên dời lịch ngược ngữ cảnh.

**DeepSeek-R1-8B:** nhỉnh nhất tổng thể (94.7).
- ✅ **Thắng bộ chat** (95.6, 0 FAIL), mạnh Intent + Multi-turn + Consistency.
- ⚠️ rag_live **TC-08 tụt còn 80**: lần này `RL14` BỎ SÓT xung đột lịch 02/06 (trước đó mạnh ở chat) → có variance.
- ⚠️ Trait reasoning: thi thoảng rò văn phong suy luận / drift EN ở task nặng.

**Qwen3.5-9B (LỚN nhất, 9B):** bám sát nhưng xếp 3.
- ⚠️ **Temporal yếu nhất** (TC-06 rag 84.4): `RL12` tính sai phạm vi "tuần sau" → bỏ sót dữ liệu.
- ⚠️ `RL18` đếm task quá hạn lẫn task đã xong (variance giữa các lần chạy).

**Yếu điểm CHUNG — cả 3 đều FAIL `RL31`:** trích nhiều dữ kiện quyết định từ biên bản giao ban 20/5 (KPI +10%, tuyển 3 dev, dời ra mắt tháng 7). Đây là điểm yếu chung về multi-fact extraction từ doc — đáng để tinh chỉnh chunking/prompt nếu nâng cấp.

## 5. Kết luận

Ba model **rất sát nhau** (overall 92-95). Xếp hạng tổng: **DeepSeek (94.7) ≈ Gemma (94.2) > Qwen (92.1)**.

- **DeepSeek-R1-8B** — cân bằng nhất, mạnh chat/intent/consistency; nhưng verbose + variance ở robustness.
- **Gemma 4-E4B** — **ấn tượng nhất theo hiệu năng/tham số**: nhỏ nhất nhưng accuracy cao nhất, nhanh nhất, thắng rag_live. Nhược: consistency. Là alternative-ecosystem rất đáng giá (đúng như doc gợi ý).
- **Qwen3.5-9B** — ổn định, đa năng, nhưng temporal rag_live là gót chân; lớn nhất mà điểm thấp nhất.

*Report gốc: `eval/results/judge/eval_report_judge__{chat,rag_live}__{google_gemma-4-e4b-it,deepseek-ai_*,Qwen_*}.md`*
