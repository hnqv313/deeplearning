# Hạn chế và các mối đe dọa đến tính hợp lệ

Tài liệu này liệt kê những yếu tố có thể làm sai lệch hoặc giới hạn phạm vi của kết luận. Mỗi
mục ghi rõ bằng chứng trong repository, mức ảnh hưởng, và cách giảm thiểu (nếu có). Số liệu
lấy từ `RESULTS.md`, `outputs/*.csv` và `outputs/colab_runs/`.

Mức ảnh hưởng: **Cao** = có thể đổi kết luận chính; **Trung bình** = ảnh hưởng độ tin cậy hoặc
phạm vi áp dụng; **Thấp** = cần nêu nhưng khó đổi kết luận.

---

## 1. Tính hợp lệ nội tại (internal validity)

### 1.1. Tính không tất định của huấn luyện GPU — Cao

**Bằng chứng.** Ở Stage 0, Naive, EWC, LwF, Replay và Joint thực hiện **cùng một thuật toán**:
Fisher của EWC chưa có, teacher của LwF chưa có, bộ nhớ Replay còn rỗng, và dữ liệu gộp của
Joint trùng dữ liệu mới. Code đặt cùng seed, `cudnn.deterministic = True`. Tuy vậy accuracy
Stage 0 trên cùng một seed khác nhau:

| Seed | Naive | EWC | LwF | Replay | Joint | Chênh lệch lớn nhất |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 91 | 94 | 95 | 99 | 94 | 8 điểm |
| 123 | 94 | 95 | 94 | 95 | 96 | 2 điểm |
| 2026 | 95 | 92 | 97 | 96 | 96 | 5 điểm |

(Stage 0 test có 100 ảnh, nên 1 ảnh = 1 điểm.)

**Ảnh hưởng.** Một phần chênh lệch giữa các phương pháp, đặc biệt Replay so với NCM (1,47
điểm) và độ lệch chuẩn 5,01 của Replay, có thể đến từ nhiễu huấn luyện chứ không phải khác
biệt thuật toán. Nguồn có thể là các kernel GPU không tất định (PyTorch chưa bật
`torch.use_deterministic_algorithms(True)`), mixed precision, và data loader nhiều worker.

**Giảm thiểu.** Bật chế độ tất định đầy đủ và chạy lại một phương pháp hai lần cùng seed để đo
nhiễu; hoặc dùng chung checkpoint Stage 0 cho mọi phương pháp dựa trên gradient; và tăng số
seed.

### 1.2. Siêu tham số không được tinh chỉnh — Cao (với kết luận về EWC/LwF)

**Bằng chứng.** `configs/ewc.yaml` (`λ = 100`, `γ = 1`, 50 batch Fisher), `configs/lwf.yaml`
(`α = 1`, `T = 2`), `configs/replay.yaml` (bộ nhớ 200, tỉ lệ 0,5). Validation được đánh giá mỗi
stage nhưng không dùng để chọn siêu tham số hay checkpoint.

**Ảnh hưởng.** Không thể kết luận EWC hay LwF "luôn thất bại". Chỉ kết luận được: thất bại
**dưới cấu hình đã thử** và giao thức một-class-mỗi-stage. Tuy vậy, thất bại của chúng phù hợp
với tài liệu về class-incremental [6, 19], và ma trận nhầm lẫn cho thấy cơ chế thất bại rõ ràng
(mục 3 bên dưới), nên kết luận định tính khá vững.

**Giảm thiểu.** Quét `λ` (ví dụ nhiều bậc độ lớn) và `α` trên tập validation; ghi lại độ lớn số
hạng phạt EWC và loss distillation theo bước.

### 1.3. Ngân sách tính toán không bằng nhau — Trung bình

**Bằng chứng.** Số bước tối ưu mỗi stage tăng dần: Naive/EWC/LwF 195; Replay 375; Joint 570,
750, 945; NCM 0.

**Ảnh hưởng.** So sánh thời gian phản ánh cấu hình epoch và batch đã chọn, không phải chi phí
tối thiểu của thuật toán. Replay được nhiều bước cập nhật hơn Naive, nên một phần lợi thế của
Replay có thể do huấn luyện nhiều hơn (dù thiết kế "cùng số epoch trên dữ liệu mới" là lựa chọn
phổ biến).

### 1.4. Đo thời gian và bộ nhớ — Trung bình

- Thời gian thực có nhiễu lớn: cùng 195 bước, Stage 1 và Stage 2 của Naive (seed 42) mất 91,1 s
  và 64,9 s. LwF có thêm forward của teacher nhưng tổng thời gian lại nhỏ hơn Naive. Vì vậy
  chênh lệch thời gian dưới khoảng 30% không nên diễn giải.
- Peak memory là `torch.cuda.max_memory_allocated`, **không** gồm bộ nhớ CUDA reserved, context
  CUDA, hay tiến trình khác.
- Peak memory 1039,38 MiB của EWC xuất hiện từ Stage 0, khi số hạng phạt bằng 0. Nó đến từ bước
  ước lượng Fisher chạy **không dùng AMP**. Đây là đặc điểm cài đặt, không phải chi phí tất yếu
  của EWC.
- Bộ nhớ phụ riêng (200 ảnh của Replay, prototype của NCM, Fisher của EWC, teacher của LwF) cần
  báo riêng, không suy ra từ con số GPU.

### 1.5. Định nghĩa metric khác với tài liệu gốc — Thấp

- Forgetting và BWT tính theo **class** thay vì theo **task**. Stage 0 có hai class nên được đếm
  hai lần.
- Forgetting lấy max trên mọi lần đánh giá **kể cả lần cuối**, nên luôn ≥ 0. Chaudhry et al.
  [27] chỉ lấy max trên các lần trước.
- Không đổi thứ hạng phương pháp, nhưng phải ghi rõ trong báo cáo để tránh bị hỏi vì sao
  forgetting bằng đúng `−BWT` với Naive, EWC, LwF và NCM.

### 1.6. Joint không phải huấn luyện lại từ đầu — Thấp

Joint fine-tune tiếp từ mô hình stage trước trên dữ liệu gộp, và vẫn có forgetting 3,00%. Đây
là mốc trên **gần đúng**. Ở seed 42, Replay (96,0%) bằng Joint (96,0%).

### 1.7. Prototype NCM tính trên ảnh có augmentation — Thấp

`ncm.py` tính prototype bằng transform huấn luyện (crop ngẫu nhiên, lật, đổi màu), mỗi ảnh chỉ
một lần. Đây là nguồn biến thiên duy nhất của NCM giữa các seed (Stage 0 của NCM đạt đúng 91% ở
cả 3 seed). Dùng transform đánh giá có thể làm prototype ổn định hơn, và có thể tăng hoặc giảm
nhẹ accuracy.

---

## 2. Tính hợp lệ của phép đo và thống kê (statistical conclusion validity)

### 2.1. Ít seed, test set nhỏ, không kiểm định thống kê — Cao (với so sánh Replay và NCM)

- Chỉ 3 seed. Độ lệch chuẩn ước lượng từ 3 giá trị rất không chắc chắn.
- Test set có 250 ảnh sau Stage 3 (1 ảnh = 0,4 điểm final accuracy; 1 ảnh = 2 điểm accuracy
  từng class).
- Thứ hạng Replay và NCM đổi chiều theo seed (Replay 96,0 / 86,0 / 91,6 so với NCM 92,4 / 92,4
  / 93,2).

**Hệ quả.** Không được viết "NCM tốt hơn Replay". Chỉ nên viết "tương đương về accuracy, rẻ hơn
và ổn định hơn".

---

## 3. Tính hợp lệ của cấu trúc thí nghiệm (construct validity)

### 3.1. Giao thức một-class-mỗi-stage là trường hợp cực đoan

Stage 1–3 mỗi stage chỉ có một class. Cross-entropy khi đó bị tối thiểu hóa bằng cách dự đoán
mọi ảnh là class mới. Đây là kịch bản bất lợi nhất cho các phương pháp không lưu dữ liệu cũ, và
khác với các benchmark phổ biến (thường thêm 5–10 class mỗi bước). Kết quả sụp đổ hoàn toàn của
Naive/EWC/LwF có thể **không** lặp lại ở giao thức nhiều class mỗi stage.

Bằng chứng về cơ chế: LwF seed 2026 sau Stage 1 có 30 ảnh Dog/Cat bị đoán sai, trong đó 29 ảnh
bị đoán là Car, chỉ 1 ảnh nhầm giữa Dog và Cat.

### 3.2. Crop object dễ hơn phân loại cả cảnh

Mỗi ảnh là crop quanh một object với 8% lề. Bài toán vì vậy dễ hơn phân loại ảnh tự nhiên có
nhiều vật thể và nền phức tạp.

### 3.3. Một thứ tự class duy nhất

Chỉ thử thứ tự Dog, Cat → Car → Person → Building. Kết quả continual learning thường nhạy với
thứ tự class; chưa đo được ảnh hưởng này.

---

## 4. Tính khái quát (external validity)

### 4.1. Lợi thế pretrain ImageNet-21k — Cao (với kết luận về NCM)

Trọng số mặc định của `vit_tiny_patch16_224` trong timm (bản local 1.0.30) là
`augreg_in21k_ft_in1k`: pretrain ImageNet-21k, fine-tune ImageNet-1k [25]. Năm class của đồ án
đều là khái niệm phổ biến trong ImageNet. NCM không học đặc trưng mới, nên kết quả tốt của nó
chủ yếu phản ánh độ phù hợp giữa miền pretrain và miền dữ liệu.

**Hệ quả.** Không được khái quát "NCM giải quyết catastrophic forgetting". Kết quả có thể khác
nhiều với miền xa ImageNet (y tế, vệ tinh, ảnh công nghiệp), class tinh (giống chó), hoặc class
đa dạng hình thức (Building đã có accuracy NCM thấp thứ hai, 92%).

Phiên bản timm và tag pretrained trên Colab không được ghi trong artifact, nên không thể khôi
phục chính xác cho 18 run đã hoàn tất. Code hiện tại đã bổ sung hai trường này vào
`environment.json` cho các lần chạy tương lai.

### 4.2. Quy mô nhỏ

5 class, 2.500 ảnh, backbone Tiny. Chưa rõ kết luận có giữ ở quy mô lớn (hàng trăm class, nhiều
stage) hay không.

### 4.3. Chi phí NCM không bằng 0

NCM vẫn phải trích đặc trưng cho mọi ảnh mới (tuyến tính theo số ảnh), và vẫn cần một backbone
pretrained lớn đã được huấn luyện trước đó bằng chi phí rất lớn (không tính trong đồ án).

---

## 5. Dữ liệu, đạo đức và giấy phép

- **Nhiễu nhãn và crop:** một số crop Person chỉ có nửa thân; một số crop Building là lối vào
  hoặc bên trong tòa nhà (`DATASET_CARD.md`).
- **Tỉ lệ và ngữ cảnh khác nhau giữa class:** kích thước box và ngữ cảnh khác nhau theo loại
  object; mô hình có thể học tín hiệu phụ (ví dụ tỉ lệ khung, nền) thay vì hình dạng object.
- **Nguồn dữ liệu gộp:** ảnh lấy từ split validation và test chính thức của Open Images rồi chia
  lại; kết quả không so sánh được với các báo cáo dùng split chính thức.
- **Quyền riêng tư:** class Person chứa ảnh người thật. Chỉ dùng ảnh công khai cho nghiên cứu và
  nhãn chính thức của Open Images; không công bố ảnh ví dụ có thể nhận diện cá nhân nếu không
  cần thiết.
- **Giấy phép:** ảnh Open Images có giấy phép riêng (thường là CC BY). Khi đưa ảnh ví dụ vào báo
  cáo/slide, giữ image ID và ghi nguồn theo yêu cầu.
- **Replay và quyền riêng tư:** Replay phải lưu ảnh cũ. Trong ứng dụng có ràng buộc riêng tư,
  điều này có thể không được phép; NCM chỉ lưu vector trung bình.

---

## 6. Tóm tắt: những gì được và không được kết luận

| Có thể kết luận | Không được kết luận |
|---|---|
| Trong giao thức này, Naive quên gần như hoàn toàn | EWC/LwF luôn thất bại |
| EWC/LwF với cấu hình đã thử không cải thiện so với Naive | NCM tốt hơn Replay |
| Replay với 200 ảnh phục hồi phần lớn hiệu năng nhưng biến thiên lớn | NCM giải quyết catastrophic forgetting nói chung |
| NCM đạt accuracy tương đương Replay với chi phí thấp hơn nhiều, trên bộ dữ liệu này | NCM không tốn chi phí |
| Thất bại của LwF đi kèm việc đoán sai sang class mới | Joint là phương pháp continual learning |
