# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiện tượng đảo nghịch giữa Training Loss và Target Accuracy ở NB4: cấu hình `attn_only` (ép rank lên 283 để bằng số tham số) có training loss thấp hơn hẳn bản chuẩn `correct` (0.5366 vs 0.6264), nhưng khi đánh giá thực tế trên tập kiểm thử thì không hề vượt trội hơn. Điều này chứng minh trực quan rằng tối ưu hóa hàm loss không đồng nghĩa với khả năng tổng quát hóa tác vụ.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở khâu sinh văn bản (text generation) để đánh giá các baseline và các mô hình đối chứng (NB2 và NB5), mất khoảng hơn 40 phút trên GPU T4. Ban đầu tôi nghĩ giai đoạn huấn luyện (NB3, NB4) sẽ chiếm phần lớn thời lượng, nhưng thực tế việc sinh văn bản tự hồi quy trên hàng chục mẫu với 3-4 phiên bản mô hình khác nhau lại ngốn nhiều thời gian suy luận hơn dự tính.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi từng tin rằng: (1) Cứ tăng rank LoRA thật cao ($r=64, 128$) thì mô hình sẽ càng thông minh; và (2) Chỉ cần so sánh kết quả fine-tune với zero-shot prompt mặc định là đủ để tuyên bố chiến thắng. Lab này đã chứng minh rằng vị trí đặt adapter (toàn bộ các lớp linear) quan trọng hơn rank nhiều lần, và fine-tune chỉ có ý nghĩa thực sự khi nó đánh bại được một Prompt Engineering đã được tối ưu hóa nghiêm túc (Baseline b).

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để giải thích các cơ chế tính toán trong `labkit` (như cách tính `matched_rank` và cơ chế gradient scaling của fp16), phát hiện lỗi line-ending CRLF trên Windows khiến hash dataset bị lệch, và hỗ trợ soạn thảo khung báo cáo. Chỗ AI thường gợi ý sai là hay vội vàng đề xuất đổi hyperparameters (tăng LR, tăng Epochs) theo cảm tính khi thấy loss cao, thay vì kiểm tra nguyên nhân gốc rễ ở chat template hay mask.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi làm không phải là mở notebook để train, mà là xây dựng một **Tập đánh giá chuẩn (Golden Evaluation Set)** và tối ưu hóa Prompt Engineering tốt nhất có thể trên mô hình nền để thiết lập mốc sàn tham chiếu cứng (frozen baseline). Sau đó, kiểm tra tính đúng đắn tuyệt đối của Loss Mask trên dữ liệu trước khi chi tiêu bất kỳ chu kỳ tính toán GPU nào.
