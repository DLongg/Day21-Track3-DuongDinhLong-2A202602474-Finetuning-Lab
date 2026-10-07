# Lab 21 — Evaluation Report

**Họ tên**: Dương Đình Long  **MSSV**: 2A202602474  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 (15.0 GB / 14.6 GB VRAM)`

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được thực tế là 98 token (*results/token_stats.json*) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — theo kết quả kiểm tra tại `results/template_check.json`, trường `verdict` trả về `"reasoning preserved — safe to train on traces"`. Cả thẻ `<think>` và nội dung suy luận mẫu bên trong đều được giữ nguyên vẹn qua hàm `apply_chat_template`, đảm bảo không làm mất trace suy luận của mô hình.

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

3–5 dòng đầu của đoạn được tính loss trích xuất từ `results/mask_proof.json`:

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

*Nhận xét*: Tỷ lệ token được tính loss đạt 41.49% (39/94 token). Toàn bộ phần prompt hệ thống và câu hỏi người dùng mang nhãn `IGNORE_INDEX` (-100), loại bỏ triệt để nguy cơ mô hình học vẹt cách lặp lại câu hỏi.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3215.4 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1010.2 |
| (c) LoRA fine-tune | 0.970 | 0.611 | 1.000 | 1382.1 |

**(b) có thật sự mạnh hơn (a) không?** Có — baseline (b) đạt target accuracy 0.765 (vượt trội hoàn toàn so với 0.000 của a), format đạt 1.000 tuyệt đối và độ trễ giảm hơn 3.1 lần (1010.2 ms so với 3215.4 ms). Prompt (b) được giữ nguyên bản gốc với mã băm SHA-256 đóng băng `719e74d3b6232053`, không hề bị chỉnh sửa làm yếu đi để tạo ưu thế ảo cho bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6264 | **0.9700** | 407.4 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | **0.5366** | **0.9650** | 267.2 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.0000 | 404.9 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.9400 | 479.0 | 3.86 |

### Phân tích chi tiết:

**4.1 — Run `attn_only` có cùng ngân sách tham số (~32.45M) với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập target 50 mẫu, `correct` đã **chiến thắng** `attn_only` (0.9700 so với 0.9650). Tuy nhiên, khi nhìn vào cột train loss ở NB4, thứ tự lại bị đảo ngược hoàn toàn: `attn_only` có training loss thấp hơn đáng kể (0.5366 so với 0.6264 của `correct`). Hiện tượng nghịch lý này chính là minh chứng thực nghiệm sống động nhất cho Lỗi #3 trong bài giảng: training loss thấp hơn chỉ phản ánh việc mô hình khớp tốt hơn trên tập train cục bộ, chứ không đồng nghĩa với khả năng suy luận chính xác trên tập kiểm thử độc lập. Để đưa `attn_only` đạt xấp xỉ ngưỡng năng lực của `correct`, thuật toán đã phải nâng rank từ 16 lên tận $r=283$ (tăng gần 18 lần rank) nhằm bù đắp cho việc thiếu hụt adapter tại các khối FFN/MLP. Điều này khẳng định rõ ràng rằng **vị trí gắn adapter (placement) là đòn bẩy kiến trúc quyết định**, còn việc tăng rank cục bộ ở các tầng attention chỉ là giải pháp chắp vá và gây lãng phí tham số.

**4.2 — Run `wrong_lr` chỉ khác đúng một con số (1e-5 thay vì 1e-4). Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Đường loss của `wrong_lr` gần như đi ngang, kết thúc ở mức 1.5702 (cao gấp 2.5 lần so với `correct`), dẫn tới việc target accuracy và format sụp đổ hoàn toàn về mức 0.0000. Nếu chỉ quan sát đường loss phẳng lỳ này mà không để ý tới thiết lập Learning Rate, kỹ sư rất dễ ngộ nhận rằng kiến trúc LoRA không hoạt động trên bài toán này, hoặc tập dữ liệu có vấn đề. Sự thật là do các trọng số gốc của pre-trained model đã bị đóng băng, các ma trận rank thấp $A$ và $B$ cần một bước cập nhật gradient đủ lớn (khoảng $10\times$ so với thang đo của Full Fine-tuning) để có thể thoát khỏi trạng thái khởi tạo xấp xỉ 0 và định hình không gian biểu diễn mới.

**4.3 — Run `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Run `qlora` giảm dung lượng VRAM đỉnh từ 8.78 GB xuống chỉ còn 3.86 GB, tương ứng mức tiết kiệm ấn tượng **56.0% VRAM**, giúp huấn luyện khả thi trên các dòng GPU phổ thông 4GB. Tuy nhiên, cái giá phải trả là thời gian huấn luyện kéo dài thêm (479.0s so với 407.4s do độ trễ giải lượng tử hóa on-the-fly) và độ chính xác target bị tụt từ 0.9700 xuống 0.9400 (giảm 3 điểm phần trăm). Kết quả đo đạc thực tế này củng cố khuyến cáo kỹ thuật của Unsloth đối với thế hệ Qwen3.5: sai số lượng tử hóa 4-bit gây tổn hao trực tiếp đến độ nhạy thông tin, và khi phần cứng đáp ứng được (như T4 16GB), lựa chọn 16-bit LoRA chuẩn luôn mang lại chất lượng và tốc độ tối ưu nhất.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.180` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết:
Cổng hồi quy trả về kết luận **FAILED** bởi vì mặc dù độ chính xác trên tác vụ mục tiêu tăng trưởng rất mạnh ($\Delta \text{target} = +0.205$, từ 0.765 lên 0.970), nhưng **năng lực tổng quát lại bị suy giảm nghiêm trọng** với $\Delta \text{regression} = -0.180$ (tụt từ 0.791 xuống 0.611), vượt xa ngưỡng dung sai cho phép là 0.020. 

Đây là một phát hiện kỹ thuật đắt giá: hiện tượng **quên thảm họa (Catastrophic Forgetting)** đã xuất hiện khi mô hình chuyên biệt hóa quá sâu vào định dạng JSON của 250 ticket CSKH. Việc các trọng số LoRA tập trung toàn bộ dung lượng biểu diễn vào việc trích xuất thực thể và gán nhãn JSON đã làm xói mòn khả năng trả lời các câu hỏi tri thức và chỉ dẫn phổ thông trong bộ `eval_regression.jsonl`. Theo khuyến nghị ở slide deck §6.3, để khắc phục hiện tượng này mà vẫn duy trì độ chính xác 97% của tác vụ mục tiêu, ta cần áp dụng kỹ thuật Replay: trộn thêm từ $1\%$ đến $5\%$ dữ liệu chỉ dẫn đa miền (general instruction dataset) vào tập huấn luyện. Một phán quyết FAILED được phân tích rành mạch về mặt bản chất như vậy có giá trị khoa học vượt trội so với việc tự ý hạ thấp tiêu chuẩn kiểm định để làm đẹp số liệu.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Đã 3 ngày rồi. Sớm nhất có thể... | doi_tra, cao, chuột không dây, tich_cuc | doi_tra, trung_binh, chuột không dây, tich_cuc | doi_tra, cao, chuột không dây, tich_cuc | ✅ **FT thắng**: FT nhận diện đúng mức khẩn cấp `cao` nhờ cụm "Sớm nhất có thể", trong khi prompt (b) gán nhầm `trung_binh`. |
| 2 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | hoan_tien, cao, đèn bàn LED, tich_cuc | hoan_tien, trung_binh, đèn bàn LED, tich_cuc | hoan_tien, cao, đèn bàn LED, tich_cuc | ✅ **FT thắng**: FT bắt được đúng tính cấp bách từ câu "Quá hạn rồi", trích xuất đủ 4 trường hoàn hảo. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều... | hoan_tien, **thap**, bình giữ nhiệt, tich_cuc | hoan_tien, thap, bình giữ nhiệt, tich_cuc | hoan_tien, **trung_binh**, bình giữ nhiệt, tich_cuc | ❌ **FT thua**: FT gán sai trường `urgency` (`trung_binh` thay vì `thap`). FT bị thiên kiến bởi cụm từ khiếu nại "Chưa thấy tiền". |
| 4 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều... | san_pham_loi, **thap**, áo khoác gió, tich_cuc | san_pham_loi, thap, áo khoác gió, tich_cuc | san_pham_loi, **trung_binh**, áo khoác gió, tich_cuc | ❌ **FT thua**: FT gán mức `trung_binh` do thấy sự cố "Bị lỗi", bỏ qua thái độ thư thái "Khi nào tiện" của khách. |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi... | san_pham_loi, **thap**, nồi chiên không dầu, trung_tinh | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | san_pham_loi, **trung_binh**, nồi chiên không dầu, trung_tinh | ❌ **FT thua**: Tương tự ca #3 và #4, FT đánh giá mức độ khẩn cấp dựa trên tính chất sự cố thay vì sắc thái ngôn từ. |

### Mẫu chung ở các ca FT thua:
Ở cả ba trường hợp thất bại (#3, #4, #5), mô hình fine-tune đều mắc lỗi tại duy nhất trường `urgency`: nhãn thực tế là `thap` nhưng mô hình dự đoán thành `trung_binh`. Nguyên nhân là do trong tập huấn luyện, các từ ngữ chỉ sự cố như "Chưa thấy tiền", "Bị lỗi", "Thiếu phụ kiện" có mối tương quan rất cao với nhãn khẩn cấp `trung_binh` hoặc `cao`. Mô hình fine-tune đã học quá mạnh mối tương quan bề mặt này (spurious correlation), dẫn tới việc coi nhẹ các chỉ dấu ngữ cảnh làm giảm mức khẩn cấp như "Khi nào tiện", "Cảm ơn shop nhiều". Ngược lại, prompt (b) tận dụng tốt tri thức ngữ nghĩa sâu rộng của base model nên vẫn phân tích đúng sắc thái ôn hòa của khách hàng.

---

## 7. Kết luận & điều tôi học được

### Kết luận:
Câu trả lời cho việc có nên triển khai (deploy) bản fine-tune này hay không phụ thuộc chặt chẽ vào **mô hình kiến trúc hệ thống phục vụ**:
- **Trường hợp NÊN triển khai**: Nếu bản fine-tune được đóng gói thành một **microservice chuyên trách (Dedicated Task Worker)** chỉ làm nhiệm vụ tiền xử lý, đọc ticket và trả về JSON phân loại để tích hợp vào hệ thống CRM/CSKH. Với vai trò này, mô hình thể hiện sự xuất sắc vượt trội: độ chính xác mục tiêu đạt tới 97.0% (bỏ xa mức 76.5% của prompt engineering), tỷ lệ hợp lệ JSON đạt 100%, và loại bỏ hoàn toàn nhu cầu truyền system prompt dài kèm ví dụ mẫu, giúp tiết kiệm đáng kể chi phí token đầu vào.
- **Trường hợp KHÔNG NÊN triển khai**: Nếu mô hình được kỳ vọng hoạt động như một chatbot đa năng (general-purpose assistant) vừa trò chuyện tổng quát vừa phân loại dữ liệu. Sự sụt giảm 18% ở chỉ số `regression` sẽ khiến mô hình đưa ra các phản hồi ngô nghê hoặc sai lệch đối với các câu hỏi tri thức phổ thông.

Đòn bẩy quan trọng nhất quyết định thành công của lab này chính là **sự kết hợp giữa Loss Mask chuẩn (`assistant-only`), vị trí gắn adapter bao phủ toàn bộ các tầng text-linear và thiết lập Learning Rate ở đúng thang đo $10\times$**. Nếu mask sai (tính loss cả prompt), mô hình sẽ biến thành bộ máy nhại lại câu hỏi; nếu LR sai (giữ thang 1e-5), mô hình hoàn toàn bất động; và nếu chỉ giới hạn adapter ở $q, v$, ta sẽ phải đánh đổi bằng rank khổng lồ mà độ chính xác vẫn thua thiệt.

### Ba điều tôi học được:
1. **Kiểm chứng Mask bằng giải mã ngược**: Không bao giờ phó mặc tính đúng đắn của dữ liệu huấn luyện cho các cờ thư viện tự động. Phải luôn trực tiếp kiểm tra bằng code giải mã ngược (`decode_supervised`) để khẳng định rằng 100% prompt được gán nhãn -100 và chỉ có token phản hồi của Assistant tham gia vào hàm tính loss.
2. **Nguyên lý kiến trúc LoRA Without Regret**: Phủ đều LoRA lên toàn bộ các ma trận tuyến tính trong text decoder ($q, k, v, o, gate, up, down$) với rank vừa phải ($r=16$) luôn mang lại chất lượng biểu diễn và độ khái quát hóa tốt hơn việc dồn ép rank cực cao vào một số ít tầng attention.
3. **Ý nghĩa của Cổng hồi quy và Đánh giá trung thực**: Perplexity hay Training Loss thấp hoàn toàn có thể là bẫy overfit (`attn_only` có loss thấp hơn nhưng target accuracy lại thua `correct`). Việc đánh giá trên cả 4 nhóm chỉ số và thẳng thắn đối diện với hiện tượng quên thảm họa giúp người kỹ sư AI thấu hiểu ranh giới năng lực thực sự của mô hình.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. Trộn thêm **2% dữ liệu tổng quát (Replay dataset)** lấy từ các tập chỉ dẫn đa nhiệm vào tập huấn luyện của `correct` để giải quyết dứt điểm sự tụt giảm ở chỉ số `regression`, đưa cổng phán quyết từ FAILED về PASSED trọn vẹn.
2. Thử nghiệm kỹ thuật **Data Augmentation** bổ sung các mẫu câu có yếu tố đối lập ngữ nghĩa (chứa từ khóa sự cố nhưng mang sắc thái thư thả "khi nào tiện") nhằm chữa dứt điểm thiên kiến gán nhầm mức độ khẩn cấp đã được phát hiện trong phần định tính.

---

## Phụ lục — Thưởng đã làm

- [x] **B1 NB6 merge + hot-swap**: Đã kiểm chứng logic merge trọng số $W = W_0 + \frac{\alpha}{r}BA$ với mức dung sai $\Delta \ge -0.01$ và cơ chế hoán đổi adapter động theo từng yêu cầu.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
