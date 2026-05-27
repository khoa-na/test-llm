# Chatbot Thư Ký Sếp – Tài liệu Use Cases & Tiêu chí Đánh giá

> **Mục đích:** Tài liệu tổng hợp use cases, tiêu chí đánh giá model AI, ma trận so sánh và hướng dẫn test cho hệ thống chatbot nhắc việc & hỗ trợ thư ký lãnh đạo.

> **Phạm vi shortlist (v3):** Chỉ tập trung vào 3 model dưới 10B đã được chọn để test thực tế: **Qwen3.5-9B**, **Gemma 4-E4B**, **DeepSeek-R1-0528-Qwen3-8B**.

---

## Mục lục

1. [Use Cases Chính](#1-use-cases-chính)
2. [Tiêu chí Đánh giá Model](#2-tiêu-chí-đánh-giá-model)
3. [Ma trận Đánh giá Model](#3-ma-trận-đánh-giá-model)
4. [Hướng dẫn Test & Mẫu Prompt](#4-hướng-dẫn-test--mẫu-prompt)
5. [Thông tin 3 Model trong Shortlist](#5-thông-tin-3-model-trong-shortlist)

---

## 1. Use Cases Chính

**Tổng hợp các tình huống sử dụng chatbot nhắc việc & hỗ trợ thư ký cho lãnh đạo**

### Nhóm: Nhắc lịch & Thời gian

**UC-01 · Nhắc lịch họp sắp tới** | Ưu tiên: 🔴 Cao | Tầng: 1 & 2

- **Mô tả:** Bot tự động hoặc theo yêu cầu thông báo các cuộc họp trong ngày/tuần, bao gồm thời gian, địa điểm, thành phần tham dự.
- **Ví dụ lệnh:** "Hôm nay tôi có lịch gì?" / "Nhắc tôi cuộc họp 2h chiều nay"
- **Kết quả kỳ vọng:** Trả về danh sách lịch họp đúng, đầy đủ thông tin; gửi nhắc nhở trước 15–30 phút.

---

**UC-02 · Nhắc deadline công việc** | Ưu tiên: 🔴 Cao | Tầng: 1 & 2

- **Mô tả:** Thông báo các nhiệm vụ sắp đến hạn, quá hạn hoặc cần xử lý gấp theo mức ưu tiên.
- **Ví dụ lệnh:** "Deadline nào đến trong tuần này?" / "Công việc nào đang trễ hạn?"
- **Kết quả kỳ vọng:** Liệt kê chính xác các task với ngày hết hạn, trạng thái; sắp xếp theo mức độ khẩn cấp.

---

**UC-03 · Đặt & quản lý lịch hẹn** | Ưu tiên: 🔴 Cao | Tầng: 1

- **Mô tả:** Hỗ trợ tạo lịch hẹn mới, kiểm tra xung đột thời gian, đề xuất khung giờ phù hợp.
- **Ví dụ lệnh:** "Đặt lịch gặp anh Nam vào thứ 4 tuần sau lúc 10h" / "Tôi có rảnh không vào 3h chiều thứ 6?"
- **Kết quả kỳ vọng:** Xác nhận đặt lịch thành công, cảnh báo nếu trùng lịch, đề xuất giờ khác nếu cần.

---

### Nhóm: Quản lý Công việc

**UC-04 · Tạo & phân công to-do list** | Ưu tiên: 🔴 Cao | Tầng: 1

- **Mô tả:** Ghi nhận danh sách việc cần làm, gán người phụ trách, deadline và mức ưu tiên cho từng task.
- **Ví dụ lệnh:** "Tạo task gửi báo cáo tuần cho phòng kế toán, deadline thứ 6, giao cho Minh"
- **Kết quả kỳ vọng:** Task được tạo đầy đủ thông tin, xác nhận rõ ràng với người dùng trước khi lưu.

---

**UC-05 · Cập nhật trạng thái công việc** | Ưu tiên: 🔴 Cao | Tầng: 1

- **Mô tả:** Cho phép đánh dấu hoàn thành, tạm hoãn, hoặc chuyển giao nhiệm vụ; tổng hợp tiến độ.
- **Ví dụ lệnh:** "Đánh dấu hoàn thành task họp hội đồng quản trị" / "Báo cáo tóm tắt tiến độ tuần này"
- **Kết quả kỳ vọng:** Cập nhật đúng trạng thái, tóm tắt tiến độ rõ ràng, không nhầm lẫn giữa các task.

---

**UC-06 · Ưu tiên hóa công việc** | Ưu tiên: 🟡 Trung bình | Tầng: 1

- **Mô tả:** Phân tích danh sách việc tồn đọng và đề xuất thứ tự xử lý dựa trên deadline và tầm quan trọng.
- **Ví dụ lệnh:** "Tôi có 10 việc cần làm, giúp tôi sắp xếp thứ tự ưu tiên"
- **Kết quả kỳ vọng:** Đưa ra danh sách sắp xếp có lý do rõ ràng, phù hợp với context công việc.

---

### Nhóm: Xử lý Thông tin

**UC-07 · Tóm tắt email/tin nhắn** | Ưu tiên: 🟡 Trung bình | Tầng: 1

- **Mô tả:** Phân tích và tóm gọn nội dung email dài, chuỗi tin nhắn, hoặc văn bản quan trọng cần sếp nắm bắt nhanh.
- **Ví dụ lệnh:** "Tóm tắt email từ Giám đốc Tài chính gửi sáng nay cho tôi"
- **Kết quả kỳ vọng:** Bản tóm tắt súc tích, nêu đúng nội dung chính, không bỏ sót điểm quan trọng.

---

**UC-08 · Tìm kiếm & truy xuất thông tin** | Ưu tiên: 🟡 Trung bình | Tầng: 1

- **Mô tả:** Tìm kiếm nhanh tài liệu, quyết định, biên bản họp, hoặc thông tin từ lịch sử trao đổi.
- **Ví dụ lệnh:** "Tìm biên bản cuộc họp tháng 3 với đối tác ABC" / "Quyết định phê duyệt dự án X ở đâu?"
- **Kết quả kỳ vọng:** Trả về đúng tài liệu/thông tin yêu cầu, kèm nguồn tham chiếu rõ ràng.

---

### Nhóm: Soạn thảo Văn bản

**UC-09 · Soạn thảo email chuyên nghiệp** | Ưu tiên: 🟡 Trung bình | Tầng: 1

- **Mô tả:** Hỗ trợ viết email, công văn, thông báo theo yêu cầu ngắn gọn từ sếp với đúng văn phong.
- **Ví dụ lệnh:** "Soạn email mời họp ban giám đốc vào thứ 2, 9h sáng, tại phòng họp A"
- **Kết quả kỳ vọng:** Email được soạn đúng format, văn phong phù hợp, đầy đủ thông tin, chỉ cần sếp review & gửi.

---

**UC-10 · Tạo biên bản & báo cáo** | Ưu tiên: 🟡 Trung bình | Tầng: 1

- **Mô tả:** Tự động tạo biên bản họp, báo cáo công việc định kỳ từ thông tin đầu vào của người dùng.
- **Ví dụ lệnh:** "Tạo biên bản họp dựa trên các ý chính sau: [danh sách ý]"
- **Kết quả kỳ vọng:** Biên bản/báo cáo đúng format chuẩn, logic rõ ràng, không có thông tin sai lệch.

---

### Nhóm: Giao tiếp & Phối hợp

**UC-11 · Theo dõi phản hồi & cam kết** | Ưu tiên: 🔴 Cao | Tầng: 2

- **Mô tả:** Nhắc sếp về các email chưa trả lời, cam kết cần thực hiện, hoặc chờ phản hồi từ bên khác.
- **Ví dụ lệnh:** "Email nào tôi chưa trả lời quá 24h?" / "Ai chưa phản hồi yêu cầu của tôi?"
- **Kết quả kỳ vọng:** Danh sách chính xác, phân loại rõ theo người gửi, chủ đề, thời gian chờ.

---

**UC-12 · Phân tích & tổng hợp cuộc họp** | Ưu tiên: 🟢 Bình thường | Tầng: 1

- **Mô tả:** Từ ghi chú hoặc transcript, trích xuất action items, quyết định quan trọng và người chịu trách nhiệm.
- **Ví dụ lệnh:** "Từ transcript cuộc họp này, liệt kê các action items và người phụ trách"
- **Kết quả kỳ vọng:** Danh sách action items rõ ràng, đúng người, đúng deadline như đã thảo luận trong meeting.

---

### Nhóm: Phân tích & Hỗ trợ Quyết định

**UC-13 · Chuẩn bị thông tin trước cuộc họp** | Ưu tiên: 🟢 Bình thường | Tầng: 1

- **Mô tả:** Tổng hợp tài liệu, lịch sử trao đổi, và thông tin cần thiết trước một cuộc họp quan trọng.
- **Ví dụ lệnh:** "Chuẩn bị brief cho cuộc họp với đối tác Singapore vào chiều nay"
- **Kết quả kỳ vọng:** Bản brief súc tích, đầy đủ context, background, và các điểm cần lưu ý khi đàm phán.

---

**UC-14 · Nhắc nhở & cảnh báo thông minh** | Ưu tiên: 🔴 Cao | Tầng: 2

- **Mô tả:** Chủ động phát hiện và cảnh báo các mâu thuẫn lịch, sự kiện quan trọng, hoặc rủi ro về deadline.
- **Ví dụ lệnh:** "Cảnh báo nếu tôi có 2 cuộc họp trùng giờ" / "Nhắc tôi đọc báo cáo trước 8h sáng thứ 2"
- **Kết quả kỳ vọng:** Cảnh báo kịp thời, đúng lúc, không gây phiền; ưu tiên đúng mức độ quan trọng.

---

### Nhóm: Đa ngôn ngữ & Hội thoại

**UC-15 · Hỗ trợ đa ngôn ngữ (Việt/Anh)** | Ưu tiên: 🟢 Bình thường | Tầng: 1 & 2

- **Mô tả:** Xử lý yêu cầu và phản hồi bằng cả tiếng Việt và tiếng Anh, tự động nhận diện ngôn ngữ.
- **Ví dụ lệnh:** "Send a meeting invitation to the board for Monday 9am" / "Nhắc tôi họp lúc 2h chiều"
- **Kết quả kỳ vọng:** Phản hồi đúng ngôn ngữ người dùng sử dụng, không pha trộn, dịch chính xác.

---

**UC-16 · Xử lý hội thoại nhiều lượt (Multi-turn)** | Ưu tiên: 🔴 Cao | Tầng: 1 & 2

- **Mô tả:** Duy trì ngữ cảnh qua nhiều câu hỏi liên tiếp trong cùng một cuộc trò chuyện.
- **Ví dụ lệnh:** "Đặt lịch họp với Tuấn" → "Vào lúc 10h" → "Thứ 5 tuần này" → "Ở phòng họp B"
- **Kết quả kỳ vọng:** Bot hiểu đúng ngữ cảnh tích lũy, không hỏi lại thông tin đã được cung cấp trước đó.

---

### Nhóm: Báo cáo & Theo dõi

**UC-17 · Tự động báo cáo định kỳ** | Ưu tiên: 🔴 Cao | Tầng: 2

- **Mô tả:** Bot chủ động tổng hợp và gửi báo cáo tình hình công việc cuối ngày/tuần mà không cần lãnh đạo yêu cầu. Nội dung gồm: task hoàn thành, task tồn đọng, cảnh báo sắp quá hạn.
- **Ví dụ lệnh:** [Bot tự động gửi lúc 17h30] "📋 Báo cáo cuối ngày: Hoàn thành 3/5 task. Còn 2 task chưa xong, 1 task hạn ngày mai. Xem chi tiết..."
- **Kết quả kỳ vọng:** Báo cáo đúng giờ cài đặt, nội dung chính xác, không cần nhắc, đủ thông tin để lãnh đạo nắm bắt nhanh mà không cần hỏi thêm.

---

### Nhóm: Workflow & Tự động hóa

**UC-18 · Kích hoạt chuỗi hành động tự động (Workflow)** | Ưu tiên: 🟡 Trung bình | Tầng: 2

- **Mô tả:** Khi nhận lệnh hoặc phát hiện sự kiện trigger (xác nhận biên bản, task mới tạo...), bot tự thực thi chuỗi bước liên tiếp: phân công → gửi thông báo → đặt nhắc → theo dõi, không cần can thiệp thủ công từng bước.
- **Ví dụ lệnh:** "Sau khi xác nhận biên bản họp, tự tạo task cho từng action item, gán đúng người phụ trách, đặt deadline và gửi thông báo cho họ"
- **Kết quả kỳ vọng:** Toàn bộ chuỗi workflow thực thi đúng thứ tự, không bỏ sót bước, xác nhận kết quả với người dùng sau khi hoàn tất.

---

## 2. Tiêu chí Đánh giá Model

**Bộ tiêu chí đánh giá tính phù hợp của mô hình AI cho từng use case – thang điểm 1–5**

> **Công thức tổng điểm:** Tổng = Σ (Điểm × Trọng số) | Ngưỡng production: ≥75 điểm

### TC-01 · Accuracy – Độ đúng của thông tin | Trọng số: 20%

- **Mô tả:** Model trả về thông tin đúng, không bịa đặt, không nhầm lẫn dữ liệu (ngày giờ, tên, task...).
- **Cách đo:** So sánh output vs. dữ liệu thực tế. Đếm % câu trả lời hoàn toàn đúng trên bộ test cases.
- **Thang điểm:** 1=Sai hoàn toàn | 2=Sai nhiều | 3=Đúng một phần | 4=Gần đúng | 5=Đúng hoàn toàn
- **Ghi chú:** Tiêu chí quan trọng nhất – sai thông tin gây hậu quả nghiêm trọng.

---

### TC-02 · Intent Recognition – Nhận diện ý định | Trọng số: 18%

- **Mô tả:** Model xác định đúng mục đích của yêu cầu: nhắc lịch, tạo task, tìm kiếm, soạn thảo, v.v.
- **Cách đo:** Kiểm tra với 30–50 câu đa dạng cách diễn đạt. Đo tỷ lệ phân loại đúng intent (%).
- **Thang điểm:** 1=<60% | 2=60–70% | 3=70–80% | 4=80–90% | 5=>90% chính xác
- **Ghi chú:** Cần test cả câu mơ hồ, câu có nhiều ý định.

---

### TC-03 · Multi-turn Context Retention | Trọng số: 15%

- **Mô tả:** Model ghi nhớ và sử dụng đúng ngữ cảnh từ các tin nhắn trước trong cùng cuộc trò chuyện.
- **Cách đo:** Tạo các kịch bản hội thoại 5–10 lượt. Kiểm tra model có hỏi lại thông tin đã cung cấp không.
- **Thang điểm:** 1=Mất context ngay | 2=Giữ 1–2 lượt | 3=Giữ 3–4 lượt | 4=Giữ 5–7 lượt | 5=Giữ xuyên suốt
- **Ghi chú:** Đặc biệt quan trọng cho luồng đặt lịch, tạo task nhiều bước.

---

### TC-04 · Language Quality – Văn phong & Ngữ pháp | Trọng số: 12%

- **Mô tả:** Output viết đúng ngữ pháp, văn phong phù hợp (lịch sự, chuyên nghiệp), tự nhiên, không cứng nhắc.
- **Cách đo:** Đánh giá thủ công bởi 2–3 reviewers theo rubric. Dùng LLM-as-judge với prompt chuyên biệt.
- **Thang điểm:** 1=Sai ngữ pháp nặng | 2=Có lỗi rõ | 3=Chấp nhận được | 4=Tốt | 5=Tự nhiên, chuyên nghiệp
- **Ghi chú:** Quan trọng cho use case soạn thảo email, biên bản.

---

### TC-05 · Multilingual Support (VI/EN) | Trọng số: 8%

- **Mô tả:** Model nhận diện đúng ngôn ngữ input và phản hồi bằng đúng ngôn ngữ đó mà không pha trộn.
- **Cách đo:** Test 20 câu tiếng Việt + 20 câu tiếng Anh + 10 câu mixed. Đo % phản hồi đúng ngôn ngữ.
- **Thang điểm:** 1=<70% | 2=70–80% | 3=80–88% | 4=88–95% | 5=>95% đúng ngôn ngữ
- **Ghi chú:** Ưu tiên tiếng Việt vì đây là ngôn ngữ chính của sếp.

---

### TC-06 · Temporal Reasoning – Lý luận thời gian | Trọng số: 12%

- **Mô tả:** Model hiểu đúng các biểu thức thời gian tương đối: "tuần sau", "thứ 4 tới", "2h chiều", "sáng mai"...
- **Cách đo:** Tạo 30 test cases với các biểu thức thời gian đa dạng. So sánh datetime output vs. expected.
- **Thang điểm:** 1=<50% đúng | 2=50–65% | 3=65–80% | 4=80–92% | 5=>92% đúng
- **Ghi chú:** Critical cho chức năng nhắc lịch – sai giờ = vô dụng.

---

### TC-07 · Response Latency – Độ trễ phản hồi | Trọng số: 7%

- **Mô tả:** Thời gian từ khi gửi tin nhắn đến khi nhận được phản hồi hoàn chỉnh, tính theo giây.
- **Cách đo:** Đo latency trung bình P50 và P95 trên 100 request thực tế trong giờ cao điểm.
- **Thang điểm:** 1=>10s | 2=5–10s | 3=3–5s | 4=1–3s | 5=<1s
- **Ghi chú:** Yêu cầu <3s cho trải nghiệm tốt trong môi trường thực tế. Lưu ý: model có thinking mode (Qwen3.5/Qwen3/DeepSeek-R1) sẽ chậm hơn khi bật chế độ này.

---

### TC-08 · Robustness – Xử lý tình huống ngoại lệ | Trọng số: 8%

- **Mô tả:** Model phản ứng hợp lý khi gặp yêu cầu không rõ ràng, thiếu thông tin, hoặc không thể thực hiện.
- **Cách đo:** Test 20 edge cases: câu mơ hồ, thiếu thông tin, yêu cầu không hợp lệ, câu hỏi ngoài phạm vi.
- **Thang điểm:** 1=Crash/Silent | 2=Lỗi không rõ | 3=Thông báo lỗi | 4=Hỏi rõ thêm | 5=Xử lý khéo léo
- **Ghi chú:** Model tốt sẽ hỏi thêm thông tin thay vì đoán mò.

---

### TC-09 · Consistency – Nhất quán câu trả lời | Trọng số: 5%

- **Mô tả:** Với cùng câu hỏi hoặc câu hỏi tương tự, model cho kết quả nhất quán qua nhiều lần thử.
- **Cách đo:** Hỏi lại cùng câu 5 lần, so sánh nội dung. Kiểm tra các câu paraphrase có cho cùng kết quả không.
- **Thang điểm:** 1=Mâu thuẫn thường xuyên | 3=Đôi khi khác nhau | 5=Luôn nhất quán
- **Ghi chú:** Đặc biệt quan trọng với thông tin ngày giờ, tên người, task.

---

### TC-10 · Cost Efficiency – Hiệu quả chi phí | Trọng số: 5%

- **Mô tả:** Chi phí API / token tiêu thụ trung bình cho một interaction, so sánh với các model khác.
- **Cách đo:** Đo average tokens/request × unit price. Tính cost/1000 interactions. So sánh across models.
- **Thang điểm:** 1=>$0.05/req | 2=$0.02–0.05 | 3=$0.01–0.02 | 4=$0.005–0.01 | 5=<$0.005/req
- **Ghi chú:** Cân đối với chất lượng – không nên hy sinh accuracy để tiết kiệm. Với on-premise: tính theo VRAM × tokens consumed.

---

**Tổng trọng số: 100%**

---

## 3. Ma trận Đánh giá Model

**MA TRẬN ĐÁNH GIÁ – 4 MODEL TRONG SHORTLIST (DƯỚI 10B THAM SỐ)**

> Điền điểm 1–5 theo kết quả test thực tế | Tổng điểm = Σ (Điểm × Trọng số) | ≥75 điểm: phù hợp production

### Cột điểm đánh giá

| Tiêu chí | Trọng số |
|---|---|
| Accuracy | ×20% |
| Intent Recognition | ×18% |
| Multi-turn | ×15% |
| Language Quality | ×12% |
| Multilingual | ×8% |
| Temporal Reasoning | ×12% |
| Latency | ×7% |
| Robustness | ×8% |
| Consistency | ×5% |
| Cost Efficiency | ×5% |

### Danh sách Model Test

| Model | Tham số | Context | Phát hành | License |
|---|---|---|---|---|
| Qwen3.5-9B (Alibaba) | 9B / 262K ctx | 262K → 1M (YaRN) | 03/2026 🆕 Mới nhất | Apache 2.0 |
| Gemma 4-E4B (Google) | ~4.5B effective / 128K ctx | 128K | 04/2026 🆕 Mới nhất | Gemma License |
| DeepSeek-R1-0528-Qwen3-8B (DeepSeek) | 8B / 128K ctx | 128K | 05/2025 💡 Reasoning | MIT |

### Bảng chấm điểm

| Tiêu chí | Trọng số | Qwen3.5-9B | Gemma 4-E4B | DeepSeek-R1-0528-Qwen3-8B |
|---|---|---|---|---|
| TC-01 Accuracy | 20% | | | |
| TC-02 Intent Recognition | 18% | | | |
| TC-03 Multi-turn | 15% | | | |
| TC-04 Language Quality | 12% | | | |
| TC-05 Multilingual | 8% | | | |
| TC-06 Temporal Reasoning | 12% | | | |
| TC-07 Latency | 7% | | | |
| TC-08 Robustness | 8% | | | |
| TC-09 Consistency | 5% | | | |
| TC-10 Cost Efficiency | 5% | | | |
| **TỔNG ĐIỂM** | **100%** | | | |

> **📌 Ghi chú quan trọng cho test:**
> - **Thinking mode:** Qwen3.5-9B và DeepSeek-R1-0528-Qwen3-8B đều hỗ trợ thinking mode. Cần test cả 2 chế độ (`enable_thinking=True/False`) để so sánh trade-off giữa chất lượng và latency.
> - **Context dài:** Qwen3.5-9B (262K) phù hợp nhất cho UC-07, UC-08, UC-12 (xử lý transcript/email dài).
> - **Multimodal:** Gemma 4-E4B hỗ trợ text + image + audio natively – có thể test thêm UC-13 với input đính kèm.
> - **Cùng precision khi so sánh:** Trên RTX 4090 24GB, recommend dùng BF16 cho cả 3 model để fair comparison.

---

## 4. Hướng dẫn Test & Mẫu Prompt

**HƯỚNG DẪN TEST & MẪU PROMPT ĐÁNH GIÁ MODEL**

---

### Test 01 · Nhắc lịch họp

- **Prompt input:** "Hôm nay là thứ 3 ngày 6/5. Tôi có những lịch họp gì trong ngày hôm nay?"
- **Kết quả kỳ vọng:** Liệt kê đầy đủ các cuộc họp: tên cuộc họp, giờ, địa điểm, người tham dự.
- **Lưu ý đánh giá:** Kiểm tra: đúng ngày, đúng thứ, không thiếu lịch nào.

---

### Test 02 · Nhắc deadline

- **Prompt input:** "Công việc nào của tôi sẽ đến hạn trong 3 ngày tới? Sắp xếp theo mức độ khẩn cấp."
- **Kết quả kỳ vọng:** Danh sách task có deadline trong 3 ngày, sắp xếp đúng thứ tự ưu tiên.
- **Lưu ý đánh giá:** Kiểm tra: tính ngày đúng, ưu tiên có hợp lý không.

---

### Test 03 · Đặt lịch hẹn

- **Prompt input:** "Đặt lịch gặp chị Lan Phòng HR vào chiều thứ 4 tuần sau, khoảng 2–3h, tại văn phòng tôi."
- **Kết quả kỳ vọng:** Xác nhận: "Đã đặt lịch gặp chị Lan (HR) - Thứ 4 [ngày], 14:00–15:00, Văn phòng của bạn"
- **Lưu ý đánh giá:** Kiểm tra: tính đúng ngày thứ 4 tuần sau.

---

### Test 04 · Tạo to-do

- **Prompt input:** "Tạo task: Gửi báo cáo Q2 cho Hội đồng Quản trị, deadline cuối tháng này, ưu tiên cao, giao cho Minh kế toán."
- **Kết quả kỳ vọng:** Xác nhận task được tạo với đầy đủ: tên, deadline, người thực hiện, mức ưu tiên.
- **Lưu ý đánh giá:** Kiểm tra model có hỏi lại thông tin còn thiếu không.

---

### Test 05 · Multi-turn (4 lượt)

- **Prompt input:**
  - Turn 1: "Đặt lịch họp với team kỹ thuật"
  - Turn 2: "Vào thứ 5"
  - Turn 3: "10 giờ sáng"
  - Turn 4: "Ở phòng meeting B tầng 3"
- **Kết quả kỳ vọng:** Sau 4 lượt: Xác nhận đặt lịch họp team kỹ thuật, Thứ 5 [ngày], 10:00, Phòng meeting B tầng 3.
- **Lưu ý đánh giá:** Model KHÔNG được hỏi lại thông tin đã cung cấp ở các lượt trước.

---

### Test 06 · Tóm tắt email

- **Prompt input:** [Paste nội dung email dài ~300 từ] → "Tóm tắt email này trong 3 gạch đầu dòng chính"
- **Kết quả kỳ vọng:** 3 điểm chính, súc tích, không bỏ sót ý quan trọng, không thêm thông tin không có trong email.
- **Lưu ý đánh giá:** Kiểm tra: không hallucinate thêm thông tin, giữ đúng ý chính.

---

### Test 07 · Soạn email

- **Prompt input:** "Soạn email mời họp Ban Giám đốc, chủ đề Kế hoạch Q3 2025, thứ 2 tuần sau lúc 9h, phòng họp Hội đồng."
- **Kết quả kỳ vọng:** Email hoàn chỉnh: Kính gửi, nội dung mời họp rõ ràng, đủ thông tin, văn phong trang trọng.
- **Lưu ý đánh giá:** Kiểm tra: đúng format email, văn phong phù hợp với cấp bậc, không thiếu thông tin cần thiết.

---

### Test 08 · Nhắc nhở thông minh (Tầng 2)

- **Prompt input:** [Không có lệnh từ người dùng] → Bot tự phát hiện: 2 cuộc họp vào lúc 14h và 14h30 bị trùng nhau.
- **Kết quả kỳ vọng:** Bot chủ động cảnh báo: "⚠️ Phát hiện xung đột lịch: Họp với team Marketing và họp BGĐ cùng lúc 14h. Bạn muốn dời buổi nào?"
- **Lưu ý đánh giá:** Kiểm tra: Bot có tự phát hiện xung đột không cần hỏi, cảnh báo đúng lúc, đề xuất phương án xử lý.

---

### Test 09 · Tự báo cáo định kỳ (Tầng 2)

- **Prompt input:** [Bot tự kích hoạt lúc 17h30] → Tổng hợp công việc trong ngày từ danh sách task.
- **Kết quả kỳ vọng:** "📋 Báo cáo cuối ngày [Ngày/Tháng]: ✅ Hoàn thành: [task 1, task 2] | ⏳ Chưa xong: [task 3] – hạn mai | ⚠️ Quá hạn: [task 4]"
- **Lưu ý đánh giá:** Kiểm tra: Đúng giờ không? Phân loại task chính xác? Không bỏ sót hay thêm task sai?

---

### Test 10 · Theo dõi phản hồi (Tầng 2)

- **Prompt input:** "Ai trong team chưa phản hồi yêu cầu gửi báo cáo Q2 mà tôi giao tuần trước?"
- **Kết quả kỳ vọng:** Danh sách tên người chưa phản hồi, số ngày đã chờ, mức độ ưu tiên yêu cầu.
- **Lưu ý đánh giá:** Kiểm tra: Đúng người chưa trả lời, tính ngày chờ chính xác, không nhầm với các yêu cầu khác.

---

### Test 11 · Workflow tự động (Tầng 2)

- **Prompt input:** "Xác nhận biên bản họp này. Tự tạo task cho từng action item và gán cho đúng người đã được phân công trong biên bản."
- **Kết quả kỳ vọng:** Bot tạo lần lượt: 3 task với tên, người phụ trách, deadline đúng theo biên bản. Gửi xác nhận: "Đã tạo 3 task và thông báo đến [A], [B], [C]."
- **Lưu ý đánh giá:** Kiểm tra: Đủ số task? Đúng người phụ trách? Deadline đúng? Không tạo task không có trong biên bản?

---

## 5. Thông tin 3 Model trong Shortlist

**THÔNG TIN CHI TIẾT – 3 MODEL DƯỚI 10B ĐƯỢC CHỌN ĐỂ TEST**

> Thông số kỹ thuật, điểm mạnh, hạn chế và khuyến nghị cho chatbot thư ký tiếng Việt

---

**Qwen3.5-9B** · Alibaba | 9B | 03/2026 | Apache 2.0 | Context: 262K → 1M (YaRN) | VI: ★★★★★

- **Điểm mạnh:** Thinking mode toggle được, hỗ trợ 201 ngôn ngữ (tiếng Việt mạnh), multimodal text+image+video, BFCL function calling 66.1, MMLU-Pro 82.5%, context dài nhất trong shortlist; Apache 2.0 thoải mái commercial.
- **Hạn chế:** Cần ~6.6GB VRAM (Q4) / ~18GB (BF16); thinking mode tiêu nhiều token hơn; phiên bản mới nên ít kiểm chứng production thực tế.
- **Cách chạy:** Ollama (`qwen3.5:9b`) / vLLM / LM Studio / SGLang
- **VRAM ước tính:** Q4_K_M ~6.6GB, BF16 ~18GB
- **Khuyến nghị:** ✅ **Ứng viên #1** – Toàn diện nhất, đặc biệt mạnh ở UC-07, UC-08, UC-12 (context dài), UC-18 (tool calling).

---

**Gemma 4-E4B** · Google | ~4.5B effective (8B total) | 04/2026 | Gemma License (commercial OK) | Context: 128K | VI: ★★★★☆

- **Điểm mạnh:** Hỗ trợ 140+ ngôn ngữ (có tiếng Việt); native multimodal (text + image + audio); kiến trúc Per-Layer Embedding (PLE) hiệu quả; configurable thinking modes; chạy được trên edge device; native function calling; ecosystem khác hệ Qwen → good alternative cho diversification risk.
- **Hạn chế:** Tiếng Việt không được tối ưu hóa explicit như Qwen; capacity nhỏ hơn (4.5B effective) → có thể yếu hơn ở reasoning phức tạp (UC-06, UC-12); Gemma License (không phải Apache 2.0 hoàn toàn tự do, nhưng OK cho commercial).
- **Cách chạy:** Ollama (`gemma4:e4b`) / LM Studio / vLLM / Google AI Studio / Vertex AI
- **VRAM ước tính:** Q4_K_M ~5-6GB, BF16 ~10GB
- **Khuyến nghị:** ✅ **Alternative ecosystem** – Khác hệ với Qwen, tốt cho cross-check; mạnh ở multimodal nếu cần xử lý ảnh/audio trong UC-13.

---

**DeepSeek-R1-0528-Qwen3-8B** · DeepSeek | 8B | 05/2025 | MIT | Context: 128K | VI: ★★★★☆

- **Điểm mạnh:** Reasoning mạnh nhất trong shortlist (distill từ DeepSeek-R1-0528 với chain-of-thought đầy đủ); cải tiến lớn so với R1-Distill đời cũ – giảm 45-50% hallucination, thêm function calling, thêm system prompt support; base Qwen3 nên thừa hưởng năng lực tiếng Việt; MIT license tự do nhất.
- **Hạn chế:** Verbose (thinking chain tiêu nhiều token); tool calling **không hoạt động trong thinking mode** (limitation lớn cho UC-18); tiếng Việt không tốt hơn base model (vì chỉ distill reasoning); latency cao hơn.
- **Cách chạy:** Ollama (`deepseek-r1:8b`) / vLLM / SGLang / llama.cpp
- **VRAM ước tính:** Q4_K_M ~4.5-5GB, BF16 ~16GB
- **Khuyến nghị:** ✅ **Reasoning specialist** – Test riêng cho UC-06 (ưu tiên hóa), UC-12 (phân tích meeting), UC-14 (cảnh báo thông minh). Không nên dùng làm primary model cho toàn bộ chatbot vì hạn chế về tool calling trong thinking mode.

---

### Khuyến nghị Test Strategy

| Mục tiêu | Model nên dùng |
|---|---|
| 🥇 Primary candidate (test trước) | **Qwen3.5-9B** – Toàn diện nhất, context dài, tool calling tốt nhất nhóm |
| 🔄 Alternative ecosystem (cross-check) | **Gemma 4-E4B** – Khác hệ Qwen, có multimodal native |
| 🧠 Reasoning specialist (chỉ test UC-06/12/14) | **DeepSeek-R1-0528-Qwen3-8B** – Khi cần phân tích sâu, ưu tiên hóa task phức tạp |

### Lưu ý Test Setup trên RTX 4090

- **Precision:** Khuyến nghị dùng **BF16 cho cả 3 model** để fair comparison. RTX 4090 24GB VRAM dư sức.
- **Thinking mode:** Test cả `enable_thinking=true` và `false` cho Qwen3.5-9B, DeepSeek-R1 để hiểu trade-off.
- **Sampling params:** Cùng `temperature=0.7, top_p=0.95, max_tokens=1024` cho cả 3 model.
- **Tool calling test:** Đặc biệt chú ý UC-04, UC-18 – Qwen3.5-9B có BFCL 66.1, Gemma 4-E4B có native function calling, DeepSeek-R1 có nhưng không dùng được khi bật thinking.

---

*Tài liệu rút gọn từ `Chatbot_ThuKy_UseCases_EvalCriteria_v2.md` | Chỉ giữ 3 model trong shortlist test | Cập nhật: 05/2026*
