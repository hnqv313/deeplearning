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

### 4.1. Lợi thế pretrain ImageNet-21k — Đã đo, xác nhận (kết luận về NCM phải giới hạn)

Trọng số mặc định của `vit_tiny_patch16_224` trong timm (bản local 1.0.30) là
`augreg_in21k_ft_in1k`: pretrain ImageNet-21k, fine-tune ImageNet-1k [25]. Năm class của đồ án
đều là khái niệm phổ biến trong ImageNet. NCM không học đặc trưng mới, nên kết quả tốt của nó
chủ yếu phản ánh độ phù hợp giữa miền pretrain và miền dữ liệu.

**Hệ quả.** Không được khái quát "NCM giải quyết catastrophic forgetting". Kết quả có thể khác
nhiều với miền xa ImageNet (y tế, vệ tinh, ảnh công nghiệp), class tinh (giống chó), hoặc class
đa dạng hình thức (Building đã có accuracy NCM thấp thứ hai, 92%).

**Đo trực tiếp, đã hoàn tất.** Vì NCM đóng băng backbone và chỉ dựng prototype từ đặc trưng,
kết quả của nó phụ thuộc hoàn toàn vào nguồn gốc đặc trưng. Đồ án đã chạy hai arm đối chứng để
tách phần đóng góp của trọng số ImageNet-21k khỏi phần đóng góp của bản thân luật NCM. Ba arm
dùng chung dữ liệu, thứ tự class, augmentation, mã đánh giá và seed; chỉ khác ở backbone:

| Arm | Backbone | Final accuracy (%) | Forgetting (%) | Thời gian (s) |
|---|---|---:|---:|---:|
| Pretrained ImageNet-21k | trọng số mặc định của timm | 92,67 ± 0,46 | 2,67 ± 0,58 | 42,1 |
| Pretrain chỉ trên Stage 0 | random init, sau đó Naive trên Dog và Cat | 25,47 ± 1,40 | 21,83 ± 4,80 | 236,6 |
| Chưa huấn luyện | random init, không huấn luyện | 23,47 ± 1,01 | 22,67 ± 2,02 | 50,8 |

Với cùng luật NCM và cùng dữ liệu, chỉ backbone ImageNet-21k đạt accuracy cạnh tranh; hai
backbone còn lại nằm gần mức 20% của bài toán 5 class. Vì vậy 92,67% **không** do bản thân luật
NCM tạo ra. Đây là bằng chứng trực tiếp cho nhận định nêu ở đầu mục, và là lý do kết luận về NCM
trong báo cáo phải giới hạn ở backbone pretrained.

**Mở rộng sang Replay.** Vì NCM không bao giờ huấn luyện backbone, mục trên chỉ trả lời cho NCM.
Đồ án đã chạy thêm Replay trên đúng checkpoint Stage 0 đó, giữ nguyên siêu tham số của Replay:
91,20 ± 5,01% khi có trọng số ImageNet-21k, xuống **45,60 ± 4,33%** khi không, forgetting tăng từ
8,50% lên 39,00%. Rehearsal vì vậy bền hơn prototype đóng băng trên cùng một backbone yếu
(45,60% so với 25,47%), nhưng Replay vẫn mất hơn một nửa accuracy. Thành công của benchmark do
đó không chỉ là hệ quả của thuật toán chống quên, mà phần lớn đến từ trọng số pretrained; đồng
thời nó cũng không phải đặc trưng riêng của NCM, vì Replay cũng hưởng lợi rõ rệt.

Con số 45,60% là sàn do bộ siêu tham số, không phải năng lực của Replay: đây là công thức
fine-tune (AdamW 1e-4, 500 bước ở Stage 0, không warmup/scheduler), và accuracy Stage 0 của arm
này chỉ 56,67% so với 96,67% khi có trọng số pretrained.

**Hybrid Replay + NCM.** Phương pháp kết hợp ở mục 4.7 đạt 95,60% trên backbone ImageNet-21k
và 53,47% trên backbone Stage 0. Kết quả này **củng cố** kết luận trên ở một hướng khác: nó cho
thấy phần lớn mất mát của Replay đến từ đầu softmax chứ không từ biểu diễn, vì thay đầu phân loại
bằng prototype đã giảm forgetting trên backbone yếu từ 39,00% xuống 14,00%.

Ba giới hạn riêng của phép đo này. Thứ nhất, prototype của hybrid chỉ dựng từ 40–100 ảnh bộ nhớ
trong khi arm NCM dùng đủ 400 ảnh/class, nên lợi thế nằm ở sự kết hợp chứ không ở prototype chính
xác hơn. Thứ hai, chênh 0,40 điểm so với mốc trên Joint nằm trong cả hai độ lệch chuẩn, nên không
được nói hybrid vượt Joint về mặt thống kê; chỉ được nói nó ngang bằng Joint trong khoảng một nửa
thời gian. Thứ ba, chi phí dựng lại prototype không tách được khỏi chênh lệch giữa các session
GPU: `training_seconds` mỗi stage cao hơn Replay khoảng 25%, nhưng bước dựng lại chỉ là 200 ảnh
forward mỗi stage, nên không thể quy chênh lệch đó cho nó. Ba seed, không kiểm định thống kê.

**Phần mối đe dọa vẫn còn lại.** Mức ảnh hưởng không thể hạ về Thấp. Hai phép đo trên chỉ chứng
minh rằng lợi thế lớn nằm ở trọng số ImageNet-21k; chúng **không** cho biết kết quả ấy có còn giữ
được với miền xa ImageNet, class tinh hay class đa dạng hình thức, vì cả ba tình huống đó đều
chưa được thử.

**Giới hạn của phép đo.** Arm Stage 0 là giám sát rẻ trong miền (800 ảnh, 2 class, 500 bước tối
ưu), không phải "không pretrain"; nó so 800 ảnh với 21k class và khoảng 300 triệu ảnh, nên đo ảnh
hưởng của quy mô và độ khớp miền. Arm này **underfit rõ rệt**: chính run Naive của nó chỉ đạt
57%, 58% và 59% accuracy ở Stage 0, so với 91% của NCM pretrained. Với 800 ảnh, 500 bước,
AdamW 1e-4, không warmup và không scheduler, đây là công thức fine-tune áp cho random init, chứ
không phải công thức from-scratch. Do đó **không** được kết luận "ViT không thể huấn luyện từ đầu".
Thêm vào đó, backbone của arm Stage 0 được huấn luyện đúng trên các crop Dog và Cat dùng để dựng
prototype, nhưng accuracy cuối theo class của nó lại xếp Dog thấp nhất (11,33%) dù Dog đã được
huấn luyện — cũng là dấu hiệu đặc trưng chưa học được đặc trưng dùng được. Chênh lệch 2,00 điểm giữa
arm Stage 0 và arm chưa huấn luyện **nằm trong nhiễu**: theo seed, arm Stage 0 dao động 24,00–26,80
còn arm chưa huấn luyện dao động 22,40–24,40, hai khoảng có chồng lấn. Ba seed trên tập test 250 ảnh,
không kiểm định thống kê, nên không có cơ sở thống kê để nói hai arm này khác nhau (mục 2.1).

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
| Kết quả NCM phụ thuộc nặng vào trọng số ImageNet-21k, đã đo trực tiếp ở mục 4.1 | NCM vẫn mạnh khi miền xa ImageNet |
| Rehearsal bền hơn prototype đóng băng khi đặc trưng yếu (45,60% so với 25,47%) | Replay không phụ thuộc trọng số pretrained |
| Kết hợp Replay với đầu prototype tốt hơn cả hai phương pháp gốc trên cả hai backbone | Hybrid vượt mốc trên Joint về mặt thống kê |
| Hybrid ngang bằng Joint với khoảng một nửa thời gian, và vẫn là phương pháp continual hợp lệ | Hybrid là phương pháp continual tốt nhất có thể |
| | ViT không thể huấn luyện từ đầu (arm Stage 0 chỉ underfit với bộ siêu tham số hiện có) |
| | Arm Stage 0 và arm chưa huấn luyện khác nhau có ý nghĩa thống kê |
