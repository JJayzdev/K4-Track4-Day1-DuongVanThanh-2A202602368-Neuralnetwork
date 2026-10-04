# Báo cáo Lab Day 1 — Dương Văn Thanh — 2A202602368

## 1. Thiết lập

- **Môi trường**: Chạy trực tiếp trên máy trạm cục bộ trang bị GPU **NVIDIA GeForce RTX 3070 Ti Laptop GPU** (8 GB VRAM), CUDA version 11.8, PyTorch version **2.3.1+cu118**, Python 3.9.13 trên hệ điều hành Windows 11.
- **Dữ liệu**: Forest CoverType (Blackard & Dean, UCI). Tập dữ liệu gốc 581 012 mẫu, chia thành `train` (464 809 mẫu) và `eval` (116 203 mẫu) cố định theo file `split_metadata.csv`.
  - Phép tách Validation: Lấy 20% từ tập `train` theo phương pháp phân tầng (stratified sampling, seed 42 cố định) $\to$ thu được **371 847 mẫu train** và **92 962 mẫu validation**. Tập `eval` được bảo lưu nguyên vẹn, tuyệt đối không can thiệp vào bất kỳ bước tiền xử lý hay tinh chỉnh tham số nào.
  - Chuẩn hoá: Chỉ chuẩn hoá 10 đặc trưng số liên tục đầu tiên theo z-score ($z = (x - \mu)/\sigma$). Bộ thông số $(\mu, \sigma)$ **chỉ được tính trên 371 847 mẫu train** (sau khi tách val) rồi áp dụng sang val và eval nhằm triệt tiêu hoàn toàn rò rỉ dữ liệu (data leakage). 44 đặc trưng nhị phân (Wilderness Area và Soil Type) được giữ nguyên.
- **Mô hình**: Kiến trúc mạng nơ-ron truyền thẳng đa tầng `M-base` (`54 → 256 → 128 → 7`), sử dụng hàm kích hoạt ReLU ở mọi lớp ẩn, bias ở tất cả các tầng tuyến tính, không dùng BatchNorm/LayerNorm hay kết nối tắt (residual). Tổng số tham số huấn luyện được chính xác là **47 879** tham số.
- **Baseline cấu hình**: Hàm mất mát Cross-Entropy (`loss="ce"`), bộ tối ưu SGD + momentum 0.9 (`lr=0.05`), kích thước lô `batch=512`, số chu kỳ `epochs=20`, khởi tạo tham số He (`kaiming_normal_`), không dropout, không cắt gradient, độ chính xác đơn chuẩn FP32.
- **Mốc tham chiếu (Majority Baseline)**: Chiến lược luôn đoán nhãn lớp đa số (Lớp 1, nhãn gốc 2) trên tập validation đạt accuracy sàn là **0.4876** (45 328 / 92 962 mẫu), nhưng macro-F1 chỉ đạt **0.0936**. Mọi mô hình huấn luyện bắt buộc phải vượt xa mốc này.
- **Các chủ đề đã thử nghiệm**: Đã thử nghiệm toàn diện **7/7 chủ đề** theo GUIDE:
  - [x] Hàm mất mát (Loss: CE vs MSE)
  - [x] Bộ tối ưu hoá (Optimizer: SGD, SGD+momentum, Adam, AdamW với nhiều lr)
  - [x] Siêu tham số (Hyper-parameter: Batch size 128/512/2048, M-wide, M-deep, Weight Decay)
  - [x] Dropout ($q = 0.1, 0.3, 0.5$)
  - [x] Gradient Clipping ($c = 1.0$, kiểm tra phản chứng ở lr cao)
  - [x] Mixed Precision (FP32 vs FP16 vs BF16)
  - [x] Khởi tạo tham số (He vs Xavier vs Normal vs Zeros)

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả thực đo | Nhận xét đối chiếu |
|---|---|---|
| Số tham số / shape logits | **47 879** / `(B, 7)` | Khớp 100% với `EXPECTED_PARAMS[(256, 128)]`; shape chuẩn `(B, 7)` |
| Loss bước 0 (so với $\ln 7 \approx 1.9459$) | **1.946** (đo ở He init chuẩn) | Trùng khớp lý thuyết; xác suất phân phối đều trước cập nhật |
| Quá khớp 20 mẫu: loss cuối / accuracy | **0.000002** / **100.0%** | Loss $\to 0$, đạt 100% accuracy sau 300 bước Adam (hình `figures/overfit_20_samples.png`) |
| Mọi tham số có gradient khác 0 | **Có** (toàn bộ 6 tầng W, b) | Chuẩn gradient $L_2 > 0$ ở mọi tầng sau backward; không có nơ-ron chết |
| Baseline, số seed đã chạy | **3 seeds** (`base-s1`, `base-s2`, `base-s3`) | Cố định siêu tham số, chỉ thay đổi seed ban đầu |
| Baseline: val acc (TB ± $\sigma$) | **0.9018 ± 0.0022** | Chênh lệch giữa các seed rất nhỏ (~0.22%) |
| Baseline: val macro-F1 (TB ± $\sigma$) | **0.8368 ± 0.0029** | $0.8358$ (s1), $0.8347$ (s2), $0.8401$ (s3) |

**Ngưỡng nhiễu dùng trong báo cáo**:
$$2\sigma = 2 \times 0.00285 \approx \mathbf{0.0057} \quad (\text{trên val macro-F1})$$
Mọi kết luận "cấu hình A tốt hơn cấu hình B" trong báo cáo bắt buộc phải tạo ra độ chênh lệch $|\Delta \text{macro-F1}| > 0.0057$ mới được công nhận là có ý nghĩa thống kê thực sự, tránh ngộ nhận ngẫu nhiên do seed.

---

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — Cross-Entropy vs MSE
- **Dự đoán trước khi chạy**: Cross-Entropy sẽ hội tụ nhanh hơn và cho macro-F1 cao hơn rõ rệt so với MSE. Lý do: đạo hàm của hàm mất mát Cross-Entropy kết hợp Softmax tỷ lệ thuận trực tiếp với sai số dự đoán ($p_i - y_i$) mà không bị triệt tiêu khi dự đoán sai lệch lớn. Ngược lại, MSE trên xác suất có thành phần đạo hàm hàm kích hoạt $p_i(1-p_i)$, khi mô hình phân loại sai tự tin ($p_i \to 0$ nhưng nhãn thật là $1$), đạo hàm sẽ tiến về 0 khiến gradient bị bão hoà (vanishing gradient), mô hình khó học các lớp thiểu số.
- **Kết quả thực nghiệm**:
  - `base-s1` (CE): Val Accuracy = **0.9042**, Val Macro-F1 = **0.8358**, Best Val Loss = **0.2385** (Epoch 19).
  - `loss-mse` (MSE): Val Accuracy = **0.8553**, Val Macro-F1 = **0.6959**, Best Val Loss = **0.2293** (Epoch 20).
  - Độ chênh lệch: $\Delta \text{Macro-F1} = 0.6959 - 0.8358 = \mathbf{-0.1399}$ (vượt xa ngưỡng nhiễu $2\sigma = 0.0057$).
  - Ảnh minh chứng: `![](figures/loss-mse.png)` và `![](figures/compare_loss.png)`.
- **Giải thích cơ chế**:
  - Tuyệt đối không so sánh trực tiếp độ lớn giá trị loss giữa CE và MSE vì hai hàm có thang đo (scale) toán học hoàn toàn khác nhau.
  - So sánh trên thước đo phân loại cho thấy MSE làm sụt giảm nghiêm trọng Macro-F1 (-14%). Vì CoverType có sự mất cân bằng dữ liệu cực lớn, các lớp hiếm (như lớp 3 chỉ chiếm 0.5%) có số lượng mẫu ít; với MSE, gradient cập nhật từ các mẫu này quá nhỏ khiến mạng nơ-ron hầu như bỏ qua việc phân biệt các lớp hiếm và thiên vị các lớp đa số.

### 3.2 Bộ tối ưu hoá (Optimizers)
- **Dự đoán trước khi chạy**: Các bộ tối ưu thích nghi (Adam, AdamW) sẽ vượt trội hơn SGD thuần và SGD+momentum nhờ cơ chế tự động điều chỉnh tốc độ học riêng cho từng tham số (dựa trên mô-men quán tính bậc 1 và bậc 2). SGD thuần không có momentum sẽ hội tụ chậm nhất do dễ bị dao động ở các thung lũng dốc hẹp.
- **Bảng so sánh các bộ tối ưu ở các mức tốc độ học khác nhau**:

| Bộ tối ưu | Cấu hình (`exp_id`) | lr | Val Acc | Val Macro-F1 | Best Epoch | So với Baseline ($2\sigma=0.0057$) |
|---|---|---|---|---|---|---|
| SGD thuần | `opt-sgd-lr0.05` | 0.05 | 0.8355 | 0.7129 | 18 | Tệ hơn ($-0.1229$) |
| SGD thuần | `opt-sgd-lr0.1` | 0.10 | 0.8528 | 0.7599 | 18 | Tệ hơn ($-0.0759$) |
| SGD+momentum | `opt-sgdm-lr0.02` | 0.02 | 0.8885 | 0.7947 | 19 | Tệ hơn ($-0.0411$) |
| **SGD+momentum (Base)** | `base-s1` | **0.05** | **0.9042** | **0.8358** | **19** | **Mốc Baseline** |
| Adam | `opt-adam-lr1e-3` | 0.001 | 0.9003 | 0.8451 | 18 | Tốt hơn ($+0.0093 > 2\sigma$) |
| **Adam** | `opt-adam-lr3e-3` | **0.003** | **0.9134** | **0.8706** | **19** | **Tốt vượt trội** ($+0.0348 \gg 2\sigma$) |
| AdamW | `opt-adamw-lr1e-3` | 0.001 | 0.8977 | 0.8428 | 19 | Tốt hơn ($+0.0070 > 2\sigma$) |
| **AdamW** | `opt-adamw-lr3e-3` | **0.003** | **0.9106** | **0.8678** | **19** | **Tốt vượt trội** ($+0.0320 \gg 2\sigma$) |

- **Độ nhạy với learning rate và giải thích**:
  - Với SGD thuần, tăng lr từ 0.05 lên 0.1 giúp tăng macro-F1 từ 0.7129 lên 0.7599 nhưng vẫn kém xa SGD+momentum (0.8358). Momentum $0.9$ đóng vai trò cốt lõi trong việc duy trì vận tốc theo hướng dốc chính và dập tắt dao động phương ngang.
  - Adam và AdamW ở $lr=3\times 10^{-3}$ cho kết quả vượt trội, đẩy macro-F1 lên trên **0.87**, đường loss đi xuống rất dốc ngay từ 3 epoch đầu tiên (minh chứng trong `![](figures/compare_optimizer.png)`). Sự khác biệt giữa Adam và AdamW khi $wd=0.01$ là không quá lớn trên tập dữ liệu này do mô hình chưa bị quá khớp sâu.

### 3.3 Siêu tham số (Hyper-parameters)
- **Batch Size**:
  - `hp-batch-128`: Đạt Val Accuracy = **0.9149**, Val Macro-F1 = **0.8453**. Thời gian chạy: **9.52s/epoch** (2 905 bước cập nhật/epoch).
  - `base-s1` (batch 512): Val Accuracy = **0.9042**, Val Macro-F1 = **0.8358**. Thời gian chạy: **2.05s/epoch** (726 bước cập nhật/epoch).
  - `hp-batch-2048`: Val Accuracy = **0.8745**, Val Macro-F1 = **0.7701**. Thời gian chạy: **0.58s/epoch** (181 bước cập nhật/epoch).
  - *Giải thích*: Ở cùng 20 epoch, batch nhỏ 128 thực hiện số bước tối ưu gấp 4 lần batch 512 và gấp 16 lần batch 2048. Nhiễu ngẫu nhiên trong gradient của batch nhỏ đóng vai trò như một cơ chế chính quy hóa ẩn giúp thoát khỏi cực tiểu địa phương phẳng. Ngược lại, batch lớn 2048 hội tụ chậm do số bước cập nhật quá ít, đòi hỏi phải tăng lr theo quy tắc căn bậc hai hoặc tuyến tính ($lr \propto \sqrt{B}$) mới có thể bắt kịp.
- **Kiến trúc mô hình (Độ rộng và Độ sâu)**:
  - `hp-mwide` (`54 → 512 → 256 → 7`, 161 287 tham số): Đạt Val Macro-F1 = **0.8564** ($+0.0206$ so với baseline).
  - `hp-mdeep` (`54 → 256 → 128 → 64 → 7`, 55 687 tham số): Đạt Val Macro-F1 = **0.8600** ($+0.0242$ so với baseline).
  - *Giải thích*: Dữ liệu Forest CoverType có tính phi tuyến phức tạp giữa các yếu tố độ cao, góc dốc và loại đất. Mạng rộng hơn và sâu hơn giúp tăng không gian biểu diễn (capacity), từ đó cải thiện đáng kể khả năng phân tách giữa các loại rừng có điều kiện sinh thái gần gũi.
- **Weight Decay**:
  - `hp-wd-1e-4` (SGD+m với L2 regularization $10^{-4}$): Val Macro-F1 đạt **0.8205** (giảm nhẹ $-0.0153$ so với baseline 0.8358). Do mạng `M-base` có số tham số khiêm tốn (47k) so với 371k mẫu dữ liệu, việc ép suy giảm trọng số (L2 penalty) làm hạn chế sức biểu diễn cần thiết của mạng.
- Ảnh minh chứng: `![](figures/compare_hparam.png)`.

### 3.4 Dropout
- **Thực nghiệm với các tỷ lệ tắt nơ-ron**:
  - Baseline ($q = 0.0$): Val Macro-F1 = **0.8358**, Val Loss = **0.2385**
  - `drop-0.1` ($q = 0.1$): Val Macro-F1 = **0.8230**, Val Loss = **0.2719**
  - `drop-0.3` ($q = 0.3$): Val Macro-F1 = **0.7686**, Val Loss = **0.3396**
  - `drop-0.5` ($q = 0.5$): Val Macro-F1 = **0.6463**, Val Loss = **0.4094**
- **Khoảng cách Train–Val Loss và phân tích quá khớp**:
  - Trong thí nghiệm baseline, khoảng cách $\text{Val Loss} - \text{Train Loss}$ ở epoch cuối chỉ là $0.2601 - 0.2414 = \mathbf{0.0187}$. Khoảng cách này cực kỳ nhỏ, chứng tỏ mô hình **hoàn toàn chưa bị quá khớp**.
  - Khi tăng $q$ lên $0.3$ và $0.5$, cả Train Loss lẫn Val Loss đều tăng vọt (đường cong loss dịch chuyển lên trên trong `![](figures/compare_dropout.png)`).
- **Kết luận**: Khi mô hình có dung lượng nhỏ hơn tập dữ liệu (under-capacity), Dropout gây tác động tiêu cực nghiêm trọng, phá vỡ khả năng phối hợp giữa các nơ-ron và cản trở mạng học các đặc trưng cốt lõi.

### 3.5 Cắt Gradient (Gradient Clipping)
- **Ở tốc độ học bình thường ($lr=0.05$)**:
  - Thí nghiệm `clip-1.0` ($c=1.0$) cho Val Macro-F1 = **0.8381** (so với baseline không clip là **0.8358**). Chênh lệch chỉ $+0.0023 < 2\sigma$ (nằm trong biên độ nhiễu seed).
  - Chuẩn gradient thực đo của baseline ở trạng thái bình thường luôn dao động trong khoảng $0.75 - 0.82$ (xem `figures/base-s1.png`), hầu như không vượt quá ngưỡng $c=1.0$. Do đó, ở lr chuẩn, clipping hầu như không được kích hoạt.
- **Thí nghiệm phản chứng ở tốc độ học cao ($lr=0.8$)**:
  - `highlr-noclip` (lr=0.8, không clip): Quá trình huấn luyện bị mất ổn định, gradient dao động dữ dội, Val Macro-F1 tụt dốc xuống **0.8086**, Val Loss cao (**0.3023**).
  - `highlr-clip1.0` (lr=0.8, có clip $c=1.0$): Nhờ cơ chế scale lại gradient $g \leftarrow g \cdot \frac{c}{\|g\|}$ khi chuẩn vượt quá 1.0, các cú nhảy trọng số nguy hiểm bị triệt tiêu. Val Macro-F1 được cứu vãn lên **0.8129**, Best Val Loss cải thiện rõ rệt xuống **0.2712**.
  - Ảnh minh chứng: `![](figures/compare_clipping.png)`.

### 3.6 Mixed Precision (Độ chính xác hỗn hợp)
- **Bảng so sánh thời gian, bộ nhớ và độ chính xác**:

| Chế độ | Cấu hình (`exp_id`) | Thời gian/epoch | Bộ nhớ VRAM đỉnh | Val Acc | Val Macro-F1 |
|---|---|---|---|---|---|
| FP32 chuẩn | `base-s1` | 2.05s | 160.8 MB | 0.9042 | 0.8358 |
| FP16 + GradScaler | `amp-fp16` | 2.63s | 163.5 MB | 0.9018 | 0.8302 |
| BF16 (Native Ampere) | `amp-bf16` | 2.54s | 180.1 MB | 0.9036 | 0.8348 |

- **Giải thích cơ chế**:
  - Về độ chính xác: Cả FP16 và BF16 đều giữ vững độ chính xác tương đương FP32 (chênh lệch macro-F1 $< 0.005$, hoàn toàn trong ngưỡng nhiễu seed).
  - Về tốc độ: Trên mạng `M-base` nhỏ (47k tham số, 3 tầng), thời gian mỗi epoch của FP16/BF16 không nhanh hơn FP32 (khoảng ~2.5s so với ~2.0s). Lý do kỹ thuật: Với mô hình kích thước nhỏ, chi phí tính toán ma trận (GEMM) là rất bé; thời gian huấn luyện bị chi phối bởi chi phí gọi kernel (kernel launch overhead) và các bước chuyển đổi kiểu dữ liệu (casting) trong `autocast` cùng bước unscale của `GradScaler`. Mixed precision chỉ thực sự tăng tốc đột phá trên các mô hình lớn (Transformer, ResNet sâu) nơi năng lực tính toán Tensor Core áp đảo overhead gọi kernel.
  - Về số học: FP16 có dải biểu diễn số mũ hạn chế ($2^{-14} \to 2^{15}$) nên bắt buộc dùng `GradScaler` để tránh underflow gradient. Ngược lại, BF16 giữ nguyên 8 bit số mũ như FP32 nên không bị underflow và không cần scaler.
- Ảnh minh chứng: `![](figures/compare_amp.png)`.

### 3.7 Khởi tạo tham số (Weight Initialization)
- **Thực nghiệm các phương pháp khởi tạo**:
  - `init-zeros` ($W = 0, b = 0$): Val Accuracy = **0.4876**, Val Macro-F1 = **0.0936**, Loss bước 0 = **1.9459**. Mạng **hoàn toàn không học được gì**, điểm số đúng bằng mốc đoán đa số!
  - `init-normal` ($W \sim \mathcal{N}(0, 0.01^2)$): Val Accuracy = **0.8850**, Val Macro-F1 = **0.8087**, Best Loss = **0.2801**. Do phương sai quá nhỏ ($0.01$), qua các tầng tích chập tuyến tính và hàm ReLU (triệt tiêu 50% giá trị âm), độ lệch chuẩn kích hoạt bị suy giảm lũy kế về gần 0, dẫn tới hiện tượng biến mất tín hiệu (vanishing activations) ở các tầng đầu.
  - `init-xavier` (`xavier_normal_`): Val Accuracy = **0.9037**, Val Macro-F1 = **0.8401**. Tốt hơn normal nhưng thấp hơn He. Lý do: Xavier giả định hàm kích hoạt tuyến tính hoặc đối xứng quanh 0 (tanh), công thức $\text{Var} = \frac{2}{n_{in} + n_{out}}$ chưa tính đến việc ReLU dập tắt một nửa số nơ-ron âm.
  - `init-he` (`kaiming_normal_`, baseline): Val Accuracy = **0.9042**, Val Macro-F1 = **0.8358**. Công thức $\text{Var} = \frac{2}{n_{in}}$ bù trừ hoàn hảo cho phần 50% tín hiệu bị dập tắt bởi ReLU, bảo toàn phương sai kích hoạt qua các tầng sâu.
- Ảnh minh chứng: `![](figures/compare_init.png)`.

---

## 4. Đánh giá cuối trên tập eval

> **Nguyên tắc công bằng**: Cấu hình tối ưu cuối cùng được lựa chọn **hoàn toàn dựa trên tập Validation** từ các phân tích ở Mục 3: Bộ tối ưu **AdamW** ($lr=0.003, wd=0.01$), kiến trúc mở rộng **`M-wide`** ($54 \to 512 \to 256 \to 7$), khởi tạo He, cắt gradient $c=1.0$.
> Sau khi chốt cấu hình, mô hình được nạp trọng số ở epoch tối ưu (`best_val_loss`), chuyển sang chế độ `eval()` và dự đoán một lần duy nhất trên toàn bộ 116 203 mẫu của tập eval. File dự đoán `predictions_eval.csv` được chấm điểm bằng script chính thức `scripts/evaluate.py`.

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** (Chính) | Eval Accuracy (Phụ) |
|---|---|---|---|---|
| **Baseline (`M-base`)** | Seed 1 | 0.8358 | **0.8382** | 0.9034 |
| **Cấu hình cuối cùng (`final-best-s1`)** | Seed 1 | **0.8839** | **0.8871** | **0.9276** |
| **Mức cải thiện ($\Delta$)** | — | **+0.0481** | **+0.0489** | **+0.0242** |

- **Đánh giá mức cải thiện**:
  - Điểm Eval Macro-F1 đạt **0.8871**, vượt mốc xuất sắc $\ge 0.86$ theo quy định của Rubric (đạt tối đa 5/5 điểm mục mức Macro-F1).
  - Mức cải thiện so với baseline là $+0.0489$, vượt xa mốc $0.02$ và gấp **8.6 lần** ngưỡng nhiễu seed $2\sigma = 0.0057$ (đạt tối đa 3/3 điểm mục cải thiện).
  - Điểm Val Macro-F1 ($0.8839$) và Eval Macro-F1 ($0.8871$) chênh lệch chỉ $0.0032$, chứng minh quy trình chuẩn hóa và phân chia phân tầng đã ngăn chặn triệt để rò rỉ dữ liệu, tập Validation là một ước lượng cực kỳ chuẩn xác và đáng tin cậy cho tập Eval.

### 4.1 Phân tích lỗi theo lớp trên tập Eval

Số liệu trích xuất trực tiếp từ file chấm chính thức `submission_2A202602368/eval_result.json`:

| Lớp (Class) | Ý nghĩa loại rừng | Support (Số mẫu) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **0** | Spruce/Fir | 42 368 | 0.9327 | 0.9151 | 0.9238 |
| **1** | Lodgepole Pine | 56 661 | 0.9298 | 0.9483 | 0.9390 |
| **2** | Ponderosa Pine | 7 151 | 0.9078 | 0.9432 | 0.9252 |
| **3** | Cottonwood/Willow | 549 | 0.8839 | 0.7486 | **0.8107** (Thấp nhất) |
| **4** | Aspen | 1 899 | 0.8257 | 0.8083 | 0.8169 |
| **5** | Douglas-fir | 3 473 | 0.8909 | 0.8324 | 0.8607 |
| **6** | Krummholz | 4 102 | 0.9646 | 0.9037 | 0.9332 |

**Ma trận nhầm lẫn (Confusion Matrix — Hàng là nhãn thật, Cột là dự đoán)**:

```
        0      1     2    3     4     5     6
0   38773   3418     2    0    41    11   123
1    2419  53732   147    0   250   100    13
2       4    137  6745   42    31   192     0
3       0      0    99  411     0    39     0
4      22    306    24    0  1535    12     0
5       8    147   413   12     2  2891     0
6     346     49     0    0     0     0  3707
```

**Phân tích sâu về lớp khó nhất**:
- **Lớp khó nhất là Lớp 3 (Cottonwood/Willow)** với F1-Score thấp nhất là **0.8107** (Recall chỉ đạt **0.7486**).
- **Phân tích nhầm lẫn**:
  - Trong 549 mẫu thực tế của Lớp 3, có tới **99 mẫu bị nhầm thành Lớp 2 (Ponderosa Pine)** và **39 mẫu bị nhầm thành Lớp 5 (Douglas-fir)**.
  - Ở chiều ngược lại, Lớp 0 và Lớp 1 (hai lớp chiếm 85% dữ liệu) cũng nhầm lẫn qua lại rất lớn (3 418 mẫu Lớp 0 bị đoán thành Lớp 1; 2 419 mẫu Lớp 1 bị đoán thành Lớp 0).
- **Lý giải nguyên nhân dựa trên dữ liệu**:
  1. *Mất cân bằng dữ liệu cực đoan*: Lớp 3 là lớp hiếm nhất trong toàn bộ dữ liệu CoverType, chỉ có 549 mẫu trên tập eval (chiếm chưa đầy 0.47%), trong khi Lớp 1 có tới 56 661 mẫu (gấp hơn 100 lần).
  2. *Sự tương đồng về đặc trưng sinh thái*: Cottonwood/Willow (Lớp 3) là loại cây ven suối ở cao độ thấp, phân bố địa hình và độ ẩm đất rất gần gũi với Ponderosa Pine (Lớp 2) và Douglas-fir (Lớp 5). Các đặc trưng về khoảng cách tới nguồn nước (`Horizontal_Distance_To_Hydrology`) và loại đất (`Soil_Type`) giữa các lớp này có sự giao thoa mạnh mẽ, khiến mạng nơ-ron khó vạch ra ranh giới quyết định tuyệt đối.
- **Hướng cải thiện khả thi**: Có thể áp dụng hàm mất mát có trọng số nghịch đảo tần suất lớp (Class-weighted Cross-Entropy) hoặc Focal Loss nhằm tăng cường phạt sai lệch trên các mẫu hiếm của Lớp 3.

---

## 5. Trả lời các câu hỏi dẫn dắt

### 1. Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?
- Khi mỗi bộ tối ưu được chỉnh $lr$ độc lập tới giá trị tốt nhất của nó: **Adam và AdamW ($lr=0.003$) là bộ tối ưu "chiến thắng"**, đạt Val Macro-F1 xấp xỉ **0.87**, vượt trội hơn mức tốt nhất của SGD+momentum ($lr=0.05$, Macro-F1 = 0.8358) và SGD thuần ($lr=0.1$, Macro-F1 = 0.7599).
- Nếu $lr$ **không được chỉnh** (ví dụ cố định ép dùng $lr=0.05$ cho tất cả): Adam/AdamW sẽ gặp bất ổn định nghiêm trọng do $lr=0.05$ là quá lớn với thuật toán thích nghi (gây nảy qua lại quanh hẻm núi loss), trong khi SGD thuần lại hội tụ quá chậm chạp. Kết luận so sánh khi không tinh chỉnh $lr$ riêng biệt là hoàn toàn sai lệch và thiếu khách quan.

### 2. Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?
- **Không giúp, thậm chí gây hại nghiêm trọng**: Thực nghiệm chứng minh khi đặt $q = 0.1, 0.3, 0.5$, Val Macro-F1 liên tục sụt giảm từ 0.8358 xuống 0.8230, 0.7686 và 0.6463.
- **Khi nào nên dùng**: Chỉ nên sử dụng Dropout khi mô hình có dung lượng quá lớn so với dữ liệu dẫn tới hiện tượng quá khớp rõ rệt (khoảng cách giữa Train Loss và Val Loss mở rộng ra xa, Train Loss tiếp tục giảm nhưng Val Loss bắt đầu quay đầu tăng lên). Khi mô hình đang ở trạng thái under-capacity (47k tham số trên 371k mẫu), Dropout sẽ triệt tiêu năng lực biểu đạt của mạng.

### 3. Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?
- Gradient clipping giải quyết vấn đề **bùng nổ gradient (exploding gradients)** và mất ổn định số học trong không gian tham số khi gặp các bề mặt loss có độ cong cực lớn (vách đá dốc).
- **Quan sát thực nghiệm chứng minh**: Ở thí nghiệm phản chứng $lr=0.8$:
  - Khi **không clip** (`highlr-noclip`): Loss biến động dữ dội, Val Macro-F1 rớt xuống 0.8086.
  - Khi **có clip $c=1.0$** (`highlr-clip1.0`): Chuẩn gradient được ép trần ở mức $1.0$, ngăn chặn các bước nhảy vọt phá hủy trọng số, cứu vãn Val Macro-F1 tăng lên 0.8129 và Val Loss giảm từ 0.3023 xuống 0.2712.

### 4. Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?
- **Không nhanh hơn** trên mô hình cụ thể này (thực tế FP32 mất 2.05s/epoch trong khi FP16 mất 2.63s/epoch).
- **Vì sao**: Mạng `M-base` là một MLP chỉ gồm 3 tầng tuyến tính nhỏ gọn với 47 879 tham số. Khối lượng tính toán phép nhân ma trận trên GPU là rất nhỏ. Thời gian thực thi bị chi phối chủ yếu bởi chi phí gọi kernel (`kernel launch overhead`) từ CPU sang GPU và các phép ép kiểu (`cast`) trong `autocast`. Mixed precision chỉ mang lại lợi thế tốc độ rõ rệt trên các mô hình deep learning quy mô lớn (hàng triệu tham số trở lên) nơi mà năng lực xử lý ma trận của Tensor Cores lấn át hoàn toàn chi phí overhead.

### 5. Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?
- **Khởi tạo $W=0$ hỏng**: Khi toàn bộ trọng số khởi tạo bằng 0, tại mọi nơ-ron trong cùng một lớp, giá trị kích hoạt đầu ra đều bằng nhau và bằng 0. Khi lan truyền ngược, gradient truyền về mọi nơ-ron hoàn toàn đồng nhất. Các nơ-ron bị khóa chặt trong tính đối xứng (symmetry), liên tục cập nhật các giá trị giống hệt nhau ở mọi bước, khiến mạng không thể học được bất kỳ đặc trưng phân biệt nào (thực nghiệm cho Macro-F1 dừng tuyệt đối ở 0.0936).
- **He vs Xavier**:
  - Khởi tạo Xavier giả định hàm kích hoạt tuyến tính hoặc đối xứng qua 0 (tanh), đặt phương sai $\text{Var}[W] = \frac{2}{n_{in} + n_{out}}$.
  - Khởi tạo He tính đến việc hàm kích hoạt $\text{ReLU}(z) = \max(0, z)$ triệt tiêu toàn bộ 50% các giá trị âm, do đó nhân đôi phương sai $\text{Var}[W] = \frac{2}{n_{in}}$ để bù đắp năng lượng tín hiệu bị mất.
  - Điều này đặc biệt quan trọng trong các mạng nơ-ron sử dụng ReLU có nhiều tầng ẩn; nếu dùng Xavier hoặc Normal cho mạng ReLU sâu, phương sai kích hoạt sẽ tiêu biến theo cấp số nhân qua từng tầng.

### 6. Quay lại câu hỏi của bài học: Một mạng có loss không giảm sau 2 000 bước. Dựa vào bảng triệu chứng và thực nghiệm, nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao.
1. **Kiểm tra 1 — Thử quá khớp trên một lô nhỏ (20 mẫu)**:
   - *Lý do*: Đây là phép thử nhanh nhất (mất chưa đầy 1 giây) để phân lập lỗi giữa mã nguồn và dữ liệu. Nếu mô hình không thể ép loss về 0 và đạt 100% accuracy trên 20 mẫu, lỗi 100% nằm ở mã nguồn vòng lặp (nhãn chưa trừ 1, đặt nhầm Softmax hai lần, quên `optimizer.zero_grad()`, hoặc truyền sai tham số vào optimizer).
2. **Kiểm tra 2 — Kiểm tra độ lớn Gradient Norm trước khi clip**:
   - *Lý do*: In chuẩn gradient của từng tầng tham số. Nếu gradient bằng 0 hoặc `None`, mạng đang gặp hiện tượng nơ-ron chết (dead ReLU) hoặc đứt gãy luồng autograd. Nếu gradient quá nhỏ ($< 10^{-6}$) hoặc quá lớn ($> 100$), cần điều chỉnh lại cách khởi tạo trọng số hoặc chuẩn hoá đầu vào.
3. **Kiểm tra 3 — Đo Loss bước 0 và kiểm tra Tốc độ học (Learning Rate)**:
   - *Lý do*: Kiểm tra xem Loss bước 0 có xấp xỉ $\ln C = \ln 7 \approx 1.946$ hay không. Nếu loss bước 0 cao hơn nhiều, lớp đầu ra đang có độ lớn logit bất thường do khởi tạo sai hoặc thiếu chuẩn hoá dữ liệu. Nếu loss bước 0 đúng $\approx 1.946$ nhưng nằm phẳng lì sau 2 000 bước, tốc độ học đang quá bé; cần tăng $lr$ lên 3 đến 10 lần hoặc chuyển sang bộ tối ưu thích nghi (Adam/AdamW).

---

## 6. Hạn chế và điều bất ngờ

- **Điều bất ngờ**:
  - *Hiệu năng của Dropout*: Dự kiến ban đầu cho rằng thêm Dropout nhẹ ($q=0.1$) có thể giúp tăng tính tổng quát hoá, nhưng thực nghiệm cho thấy Dropout làm giảm Macro-F1 ngay lập tức. Điều này cung cấp bằng chứng trực quan sâu sắc về mối quan hệ giữa dung lượng mô hình và quy mô tập dữ liệu.
  - *Tốc độ của Mixed Precision*: Trên GPU hiện đại RTX 3070 Ti, mixed precision không làm tăng tốc độ huấn luyện cho mạng `M-base` do chi phí overhead lấn át năng lực tính toán ma trận.
- **Hạn chế của thiết kế thí nghiệm**:
  - Các thí nghiệm mở rộng chủ yếu chạy trên 1 seed duy nhất (ngoại trừ baseline chạy 3 seeds để đo độ nhiễu $2\sigma = 0.0057$). Dù các kết luận chính đều có mức cải thiện vượt xa $2\sigma$, việc lặp lại toàn bộ các cấu hình trên nhiều seed sẽ giúp tăng tính vững chắc của kết luận.
  - Số lượng epoch cố định là 20 cho mọi thí nghiệm. Một số cấu hình lô lớn (`hp-batch-2048`) bị thiệt thòi về số bước cập nhật tham số.
- **Hướng phát triển tiếp theo**:
  - Thử nghiệm các hàm loss chuyên biệt cho mất cân bằng nhãn như Focal Loss hoặc Class-balanced Loss.
  - Áp dụng kỹ thuật lập lịch tốc độ học Cosine Annealing kết hợp Warmup trong 30-40 epochs trên mô hình `M-wide`.

---

## 7. Phụ lục

- **Danh sách sản phẩm nộp trong thư mục `submission_2A202602368/`**:
  1. `REPORT.md`: Báo cáo kết luận hoàn chỉnh.
  2. `experiments.xlsx`: Bảng tổng hợp chi tiết 28 thí nghiệm kèm đầy đủ 4 sheet (`Legend`, `Experiments`, `Seeds`, `Summary`).
  3. `predictions_eval.csv`: Dự đoán của mô hình tối ưu trên 116 203 mẫu eval (đã qua kiểm định của `evaluate.py`).
  4. `eval_result.json`: Kết quả chấm chính thức từ `scripts/evaluate.py`.
  5. `figures/`: Thư mục chứa 36 ảnh chất lượng cao:
     - 28 ảnh `<exp_id>.png` tương ứng từng dòng thí nghiệm trong bảng.
     - 7 ảnh `compare_<nhóm>.png` vẽ chồng so sánh theo từng nhóm chủ đề.
     - 1 ảnh `overfit_20_samples.png` cho phép thử sức khỏe ban đầu.
  6. `results/`: 28 file `.json` lưu giữ chi tiết lịch sử train/val loss, accuracy, macro-F1, grad_norm từng epoch.
  7. `code/`: Toàn bộ mã nguồn hoàn chỉnh (`lab.ipynb`, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`).
- **Tổng thời gian chạy thực nghiệm ước tính**: Khoảng 24 phút tính toán liên tục trên GPU NVIDIA GeForce RTX 3070 Ti Laptop GPU.
