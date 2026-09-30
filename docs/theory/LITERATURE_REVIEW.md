# Tổng quan lý thuyết: Continual Image Classification without Catastrophic Forgetting

Tài liệu này giải thích nền tảng lý thuyết cho các phương pháp được so sánh trong đồ án
(Naive fine-tuning, EWC, LwF, Replay, Frozen ViT-Tiny + NCM, Joint training) và gắn từng
khái niệm với cách cài đặt thực tế trong repository. Mọi trích dẫn được liệt kê ở mục
[Tài liệu tham khảo](#tài-liệu-tham-khảo) kèm trạng thái kiểm chứng.

## Cách kiểm chứng trích dẫn

- Metadata (tác giả, năm, nơi công bố, tập, trang, DOI) được đối chiếu ngày 2026-09-30 với
  Crossref API (theo DOI), arXiv API (theo mã arXiv), hoặc trang proceedings chính thức
  (CVF Open Access, PMLR, NeurIPS, OpenReview).
- Các định nghĩa metric quan trọng được đối chiếu với nguyên văn bài gốc khi truy cập được:
  forgetting measure (Chaudhry et al., 2018) và average incremental accuracy (Rebuffi et
  al., 2017) đã đối chiếu nguyên văn. Công thức BWT của Lopez-Paz & Ranzato (2017) đã được
  đối chiếu với Phương trình (3) trong bản proceedings chính thức của NeurIPS.
- Nhận định về nội dung bài báo được viết ở dạng diễn giải (paraphrase), không trích dài.

---

## 1. Catastrophic forgetting

**Catastrophic forgetting** (hay catastrophic interference) là hiện tượng một mạng nơ-ron,
khi được huấn luyện tiếp trên dữ liệu mới, mất nhanh và mạnh khả năng xử lý dữ liệu đã học
trước đó. McCloskey & Cohen (1989) [1] là một trong những công trình đầu tiên mô tả hiện
tượng này ở mạng connectionist khi học tuần tự. Ratcliff (1990) [2] cho thấy các mô hình
connectionist của trí nhớ nhận diện gặp ràng buộc tương tự khi học và quên. French (1999)
[3] tổng quan nguyên nhân: các biểu diễn phân tán dùng chung trọng số, nên cập nhật cho dữ
liệu mới ghi đè lên tri thức cũ.

Trong học sâu hiện đại, vấn đề này thường được đặt trong khung **stability–plasticity**:
mô hình cần đủ "dẻo" để học cái mới nhưng đủ "ổn định" để không quên cái cũ. Các bài tổng
quan của Parisi et al. (2019) [4] và De Lange et al. (2022) [5] chia các hướng giải quyết
thành ba nhóm chính:

| Nhóm | Ý tưởng | Đại diện trong đồ án |
|---|---|---|
| Regularization-based | Phạt thay đổi tham số hoặc thay đổi đầu ra quan trọng với tri thức cũ | EWC, LwF |
| Replay / rehearsal | Lưu lại (hoặc sinh lại) một phần dữ liệu cũ để trộn khi học mới | Replay |
| Parameter isolation / kiến trúc | Dành riêng hoặc cố định một phần tham số cho tri thức cũ | Frozen ViT-Tiny + NCM (đóng băng toàn bộ backbone) |

Xếp NCM vào nhóm cuối chỉ là cách định vị gần đúng của nhóm: phương pháp này không học
tham số nào, mà cố định hoàn toàn bộ trích xuất đặc trưng và chỉ lưu một prototype cho mỗi
class.

## 2. Ba kịch bản continual learning và vì sao class-incremental khó nhất

van de Ven & Tolias (2019) [6] và van de Ven, Tuytelaars & Tolias (2022) [7] phân biệt ba
kịch bản:

- **Task-incremental**: khi test, mô hình được biết ảnh thuộc task nào (thường dùng một
  head riêng cho mỗi task).
- **Domain-incremental**: tập nhãn giữ nguyên, phân phối đầu vào thay đổi.
- **Class-incremental**: class mới xuất hiện theo thời gian, và khi test mô hình phải phân
  biệt **tất cả** class đã gặp mà **không** được biết task.

Đồ án thuộc kịch bản **class-incremental**: một head 5 output dùng chung, logit của class
chưa học bị che (mask), và khi đánh giá mô hình phải chọn trong mọi class đã thấy. Theo
[6], các phương pháp regularization như EWC thất bại trong kịch bản class-incremental, và
việc replay (dữ liệu hoặc biểu diễn) có vẻ cần thiết khi mô hình phải tự suy ra task lúc
test. Bài survey thực nghiệm của Masana et al. (2023) [8] trên 13 phương pháp class-
incremental cũng đánh giá các phương pháp không dùng exemplar trong điều kiện này.

**Thiên lệch về class mới (task-recency bias).** Khi chỉ có dữ liệu của class mới, lớp
phân loại cuối có xu hướng cho điểm cao hơn cho class mới. Wu et al. (2019) [18] xác định
sự mất cân bằng dữ liệu giữa class cũ và mới là nguyên nhân chính làm suy giảm hiệu năng,
và đề xuất một lớp hiệu chỉnh bias tuyến tính. Zhao et al. (2020) [19] chỉ ra rằng
knowledge distillation giữ được khả năng phân biệt **giữa các class cũ với nhau** nhưng
không ngăn được thiên lệch về class mới, và đề xuất Weight Aligning để cân lại trọng số
lớp phân loại.

**Liên hệ với đồ án.** Ở Stage 1, 2 và 3, mỗi stage chỉ có **đúng một class mới**. Với
cross-entropy trên các logit đã thấy và toàn bộ nhãn là class mới, mô hình có thể giảm loss
chỉ bằng cách đẩy logit của class mới lên và logit class cũ xuống. Đây là trường hợp cực
đoan của task-recency bias. Kết quả thực nghiệm khớp với dự đoán này (xem báo cáo, mục 5).

## 3. Elastic Weight Consolidation (EWC)

Kirkpatrick et al. (2017) [9] đề xuất làm chậm việc học trên các tham số quan trọng với
task cũ. Mức "quan trọng" được xấp xỉ bằng đường chéo của ma trận Fisher information
`F`. Khi học task B sau task A, loss trở thành:

```latex
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i \left(\theta_i - \theta^{*}_{A,i}\right)^2
```

trong đó `θ*_A` là tham số sau khi học xong A và `λ` điều chỉnh mức ưu tiên tri thức cũ.

**Online EWC.** Schwarz et al. (2018) [10] dùng một phiên bản "online" chỉ giữ một bộ
Fisher và một điểm neo tham số, cập nhật Fisher dạng trung bình động `F ← γ F_cũ + F_mới`
thay vì lưu một số hạng phạt riêng cho mỗi task.

**Empirical Fisher.** Nhiều cài đặt thay Fisher thật (kỳ vọng theo phân phối dự đoán của mô
hình) bằng *empirical Fisher* (bình phương gradient với nhãn thật). Kunstner, Balles &
Hennig (2019) [11] chỉ ra rằng empirical Fisher nói chung không nắm bắt được thông tin bậc
hai như Fisher thật, và không phải là ước lượng Monte Carlo của Fisher.

**Cài đặt trong đồ án** (`src/continual_dl/strategies/ewc.py`, `configs/ewc.yaml`):

- Online EWC với `λ = 100`, `γ = 1` (Fisher được **cộng dồn**, không suy giảm), ước lượng
  trên tối đa 50 batch của dữ liệu stage vừa học.
- Dùng **empirical Fisher**: bình phương gradient của cross-entropy với nhãn thật.
- Số hạng phạt là `0.5 · λ · Σ F_i (θ_i − θ*_i)²`, khớp với công thức [9].

Một hệ quả đáng chú ý (giả thuyết, **chưa được đo** trong đồ án): ở Stage 1–3, dữ liệu chỉ
có một class, nên sau khi huấn luyện, mô hình dự đoán class đó gần như chắc chắn. Gradient
của cross-entropy khi đó rất nhỏ, dẫn tới Fisher ước lượng từ stage này gần bằng 0. Nếu đúng
vậy, EWC gần như không bảo vệ được tham số học ở các stage một-class.

## 4. Learning without Forgetting (LwF)

Li & Hoiem (2018) [12] đề xuất học task mới **mà không cần dữ liệu cũ**: trước khi học,
ghi lại đầu ra của mô hình cũ trên chính dữ liệu mới, rồi khi huấn luyện, buộc đầu ra tương
ứng của mô hình mới giống đầu ra cũ bằng **knowledge distillation**. Distillation dùng
softmax làm mềm bằng nhiệt độ `T` (Hinton, Vinyals & Dean, 2015 [13]), và loss distillation
được nhân với `T²` để độ lớn gradient không phụ thuộc vào `T`.

Thiết kế gốc của LwF có **head riêng cho mỗi task** (bối cảnh task-incremental), nên việc
giữ đầu ra của head cũ đủ để bảo toàn task cũ.

**Cài đặt trong đồ án** (`strategies/lwf.py`, `configs/lwf.yaml`, `α = 1`, `T = 2`):

- Teacher là bản sao đóng băng của mô hình **ngay trước** stage đang học.
- Loss distillation là KL-divergence giữa softmax có nhiệt độ của **riêng các logit class
  cũ** (teacher và student), cộng với cross-entropy trên các logit đã thấy.

**Vì sao LwF có thể sụp trong class-incremental một-head.** Softmax chỉ trên các logit class
cũ không đổi khi cộng cùng một hằng số vào mọi logit cũ. Vì vậy loss distillation giữ được
**thứ tự tương đối giữa các class cũ** (ví dụ Dog so với Cat), nhưng **không ràng buộc** việc
logit class mới vượt lên trên tất cả logit cũ. Đây đúng là hiện tượng mà Zhao et al. (2020)
[19] mô tả. Ma trận nhầm lẫn của đồ án minh họa rõ điều này (xem báo cáo, mục 5.2).

## 5. Replay / rehearsal

Ý tưởng lưu lại và luyện lại dữ liệu cũ đã có từ Robins (1995) [14], người phân tích
rehearsal và pseudorehearsal như cách giảm catastrophic forgetting. Trong học sâu, Rolnick et
al. (2019) [15] cho thấy experience replay đơn giản là một baseline mạnh, và Chaudhry et al.
(2019) [16] cho thấy ngay cả bộ nhớ episodic rất nhỏ cũng giúp đáng kể. iCaRL (Rebuffi et
al., 2017) [17] kết hợp exemplar rehearsal với distillation và chọn exemplar bằng
**herding**, để trung bình đặc trưng của tập exemplar xấp xỉ tốt trung bình của class.

**Cài đặt trong đồ án** (`strategies/replay.py`, `configs/replay.yaml`):

- Bộ nhớ cố định **200 ảnh**, chia đều theo class đã thấy (ví dụ 50 ảnh/class khi học
  Stage 3). Ảnh được chọn **ngẫu nhiên** theo seed chạy, không dùng herding.
- Mỗi batch gồm 16 ảnh mới và 16 ảnh từ bộ nhớ (`replay_ratio = 0.5`).
- Replay vi phạm giả định "không lưu dữ liệu cũ". Trong ứng dụng có ràng buộc riêng tư
  hoặc lưu trữ, đây là chi phí thật chứ không chỉ là chi tiết kỹ thuật.

## 6. Nearest Class Mean (NCM) trên đặc trưng pretrained đóng băng

Mensink et al. (2013) [20] biểu diễn mỗi class bằng **trung bình đặc trưng** của nó và
phân loại theo khoảng cách tới các trung bình. Nhờ vậy có thể thêm class mới gần như không
tốn chi phí huấn luyện. iCaRL [17] dùng quy tắc **nearest-mean-of-exemplars** để phân loại.

Gần đây, nhiều công trình cho thấy bộ đặc trưng pretrained đóng băng cộng với bộ phân loại
prototype là một baseline rất mạnh cho class-incremental:

- Janson et al. (2022) [21] cho thấy NCM trên đặc trưng pretrained đóng băng vượt nhiều
  phương pháp continual learning phức tạp trên các benchmark chuẩn, từ đó đặt câu hỏi về
  sự cần thiết của các cơ chế phức tạp.
- Zhou et al. (2025) [22] lập luận rằng embedding của mô hình pretrained đóng băng đã đủ khái
  quát, và một bộ phân loại prototype đơn giản có thể vượt các phương pháp phức tạp. Họ đề
  xuất APER để kết hợp mô hình đóng băng với mô hình được thích nghi.

Vì vậy kết quả mạnh của NCM trong đồ án **phù hợp với tài liệu** và **không phải là đóng góp
mới** của nhóm. Điểm đóng góp là đo nó cùng một giao thức, cùng phần cứng với các phương
pháp khác, kèm chi phí thời gian và bộ nhớ.

**Cài đặt trong đồ án** (`strategies/ncm.py`): backbone ViT-Tiny đóng băng, trích đặc trưng
token `[CLS]`, chuẩn hóa L2, cộng dồn tổng đặc trưng theo class. Dự đoán theo **cosine
similarity** lớn nhất với prototype. Lưu ý: prototype được tính trên ảnh train **có** data
augmentation ngẫu nhiên (transform huấn luyện), đây là nguồn biến thiên giữa các seed.

## 7. Vision Transformer và trọng số pretrained

- **ViT** (Dosovitskiy et al., 2021) [23] chia ảnh thành các patch 16×16, coi mỗi patch như
  một token và đưa vào Transformer encoder chuẩn.
- Cấu hình "Tiny" (embedding 192 chiều, 12 block, 3 attention head) được dùng trong DeiT
  (Touvron et al., 2021) [24].
- Trọng số mặc định của `vit_tiny_patch16_224` trong timm 1.0.30 (phiên bản trong `.venv`
  của máy local) là tag `augreg_in21k_ft_in1k`, tức **pretrain trên ImageNet-21k rồi
  fine-tune trên ImageNet-1k**, theo quy trình "AugReg" của Steiner et al. (2022) [25].
  Backbone có 5.524.416 tham số (đếm bằng timm local, không tính head 5 class).
  Phiên bản timm và tag trọng số của các run Colab **không được ghi lại và không thể khôi
  phục chính xác từ artifact hiện có**; vì vậy thông tin tag trên chỉ mô tả môi trường local,
  không được xem là metadata đã xác nhận của benchmark Colab.

Ý nghĩa cho đồ án: ImageNet-21k chứa nhiều khái niệm gần với Dog, Cat, Car, Person và
Building, nên bộ đặc trưng đã phân tách tốt năm class này trước khi đồ án huấn luyện gì.
Đây là lý do chính khiến NCM mạnh, và cũng là một mối đe dọa với tính khái quát của kết
luận.

## 8. Các metric đánh giá

Gọi `a_{k,j}` là accuracy trên class `j` sau khi học xong stage `k` (k = 0..3).

| Metric | Định nghĩa trong tài liệu | Cách đồ án tính (`metrics/classification.py`) |
|---|---|---|
| Average accuracy `A_k` | Trung bình accuracy trên các task đã thấy sau task k [27] | Trung bình **accuracy từng class** trên các class đã thấy. Test set cân bằng (50 ảnh/class) nên bằng overall accuracy. |
| Final average accuracy | `A_T` | `A_3`, trên cả 5 class |
| Average incremental accuracy | Trung bình accuracy sau mỗi batch class [17] | `(A_0 + A_1 + A_2 + A_3) / 4`, gồm cả Stage 0 |
| Forgetting | `f_j^k = max_{l ∈ {1..k−1}} a_{l,j} − a_{k,j}`, trung bình trên các task cũ [27] | Theo **class**: `max(mọi a_{l,j} kể cả cuối) − a_{3,j}`, trung bình trên Dog, Cat, Car, Person |
| Backward transfer (BWT) | `1/(T−1) Σ_{i<T} (R_{T,i} − R_{i,i})` [26], Phương trình (3) | Theo **class**: `a_{3,j} − a_{(stage học j), j}`, trung bình trên Dog, Cat, Car, Person |

Các khác biệt cần nêu rõ trong báo cáo:

1. Đồ án tính theo **class** thay vì theo **task**. Dog và Cat cùng thuộc Stage 0 nên Stage
   0 được đếm hai lần trong trung bình forgetting/BWT.
2. Forgetting của đồ án lấy max **bao gồm cả lần đánh giá cuối**, nên luôn ≥ 0. Định nghĩa
   [27] chỉ lấy max trên các lần trước.
3. Forgetting và `−BWT` trùng nhau khi accuracy của mỗi class chỉ giảm dần sau khi học
   (Naive, EWC, LwF, NCM). Chúng khác nhau khi accuracy tăng lên ở giữa rồi giảm (Replay,
   Joint), ví dụ Joint có forgetting 3,00 nhưng BWT −1,83.

## 9. Dữ liệu

Open Images (Kuznetsova et al., 2020) [28] là bộ dữ liệu quy mô lớn với nhãn phân loại ảnh,
bounding box và quan hệ thị giác. Đồ án dùng Open Images V7 qua tích hợp FiftyOne, cắt
(crop) ảnh theo bounding box để tạo bài toán phân loại một nhãn. Trang chính thức của V7:
<https://storage.googleapis.com/openimages/web/download_v7.html>.

---

## Ma trận tài liệu

| # | Tài liệu | Ý chính dùng trong đồ án | Dùng ở đâu | Kiểm chứng |
|---|---|---|---|---|
| 1 | McCloskey & Cohen 1989 | Mô tả sớm catastrophic interference | Giới thiệu | Crossref + trang nhà xuất bản |
| 2 | Ratcliff 1990 | Ràng buộc học/quên trong mô hình connectionist | Giới thiệu | Crossref |
| 3 | French 1999 | Tổng quan nguyên nhân forgetting | Giới thiệu | Crossref |
| 4 | Parisi et al. 2019 | Tổng quan lifelong learning | Phân loại phương pháp | Crossref + arXiv |
| 5 | De Lange et al. 2022 | Taxonomy, stability–plasticity | Phân loại phương pháp | Crossref (bản online) + tìm kiếm cho tập/số |
| 6 | van de Ven & Tolias 2019 | Ba kịch bản, EWC thất bại ở class-IL | Thảo luận kết quả | arXiv API |
| 7 | van de Ven et al. 2022 | Ba loại incremental learning | Định nghĩa bài toán | Crossref |
| 8 | Masana et al. 2023 | Survey class-IL, 13 phương pháp | Thảo luận | Crossref + arXiv |
| 9 | Kirkpatrick et al. 2017 | EWC | Phương pháp | Crossref + arXiv |
| 10 | Schwarz et al. 2018 | Online EWC | Phương pháp | PMLR |
| 11 | Kunstner et al. 2019 | Giới hạn của empirical Fisher | Thảo luận EWC | arXiv + NeurIPS |
| 12 | Li & Hoiem 2018 | LwF | Phương pháp | Crossref + arXiv |
| 13 | Hinton et al. 2015 | Knowledge distillation, nhiệt độ | Phương pháp | arXiv API |
| 14 | Robins 1995 | Rehearsal / pseudorehearsal | Phương pháp | Crossref |
| 15 | Rolnick et al. 2019 | Experience replay | Phương pháp | arXiv API (journal ref NeurIPS 2019) |
| 16 | Chaudhry et al. 2019 | Bộ nhớ episodic nhỏ | Thảo luận Replay | arXiv API |
| 17 | Rebuffi et al. 2017 | iCaRL, NME, herding, avg. incremental acc. | Phương pháp, metric | CVF + nguyên văn PDF |
| 18 | Wu et al. 2019 | Bias về class mới | Thảo luận | CVF |
| 19 | Zhao et al. 2020 | Distillation không chống được bias class mới | Thảo luận LwF | arXiv API (trang CVPR chưa xác minh) |
| 20 | Mensink et al. 2013 | NCM | Phương pháp | Crossref |
| 21 | Janson et al. 2022 | Frozen pretrained + NCM là baseline mạnh | Thảo luận NCM | arXiv API |
| 22 | Zhou et al. 2025 | Prototype trên PTM đóng băng, APER | Thảo luận NCM | Springer (qua tìm kiếm) + arXiv |
| 23 | Dosovitskiy et al. 2021 | ViT | Mô hình | OpenReview + arXiv |
| 24 | Touvron et al. 2021 | DeiT, cấu hình Tiny | Mô hình | PMLR |
| 25 | Steiner et al. 2022 | AugReg, trọng số IN-21k | Mô hình | arXiv API (TMLR) |
| 26 | Lopez-Paz & Ranzato 2017 | Định nghĩa BWT | Metric | NeurIPS proceedings, Phương trình (3) + arXiv |
| 27 | Chaudhry et al. 2018 | Forgetting measure | Metric | Crossref + nguyên văn HTML |
| 28 | Kuznetsova et al. 2020 | Open Images | Dữ liệu | Crossref + arXiv |

---

## Tài liệu tham khảo

[1] M. McCloskey and N. J. Cohen, "Catastrophic interference in connectionist networks: The
sequential learning problem," *Psychology of Learning and Motivation*, vol. 24, pp. 109–165,
1989. doi:10.1016/S0079-7421(08)60536-8

[2] R. Ratcliff, "Connectionist models of recognition memory: Constraints imposed by learning
and forgetting functions," *Psychological Review*, vol. 97, no. 2, pp. 285–308, 1990.
doi:10.1037/0033-295X.97.2.285

[3] R. M. French, "Catastrophic forgetting in connectionist networks," *Trends in Cognitive
Sciences*, vol. 3, no. 4, pp. 128–135, 1999. doi:10.1016/S1364-6613(99)01294-2

[4] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, "Continual lifelong
learning with neural networks: A review," *Neural Networks*, vol. 113, pp. 54–71, 2019.
doi:10.1016/j.neunet.2019.01.012

[5] M. De Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and
T. Tuytelaars, "A continual learning survey: Defying forgetting in classification tasks,"
*IEEE TPAMI*, vol. 44, no. 7, pp. 3366–3385, 2022. doi:10.1109/TPAMI.2021.3057446

[6] G. M. van de Ven and A. S. Tolias, "Three scenarios for continual learning,"
arXiv:1904.07734, 2019.

[7] G. M. van de Ven, T. Tuytelaars, and A. S. Tolias, "Three types of incremental learning,"
*Nature Machine Intelligence*, vol. 4, no. 12, pp. 1185–1197, 2022.
doi:10.1038/s42256-022-00568-3

[8] M. Masana, X. Liu, B. Twardowski, M. Menta, A. D. Bagdanov, and J. van de Weijer,
"Class-incremental learning: Survey and performance evaluation on image classification,"
*IEEE TPAMI*, vol. 45, no. 5, pp. 5513–5533, 2023. doi:10.1109/TPAMI.2022.3213473

[9] J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," *PNAS*,
vol. 114, no. 13, pp. 3521–3526, 2017. doi:10.1073/pnas.1611835114

[10] J. Schwarz, J. Luketina, W. M. Czarnecki, A. Grabska-Barwinska, Y. W. Teh, R. Pascanu,
and R. Hadsell, "Progress & Compress: A scalable framework for continual learning," in
*Proc. ICML*, PMLR 80, pp. 4528–4537, 2018.

[11] F. Kunstner, L. Balles, and P. Hennig, "Limitations of the empirical Fisher
approximation for natural gradient descent," in *Advances in NeurIPS 32*, 2019.
arXiv:1905.12558

[12] Z. Li and D. Hoiem, "Learning without Forgetting," *IEEE TPAMI*, vol. 40, no. 12,
pp. 2935–2947, 2018. doi:10.1109/TPAMI.2017.2773081

[13] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network,"
arXiv:1503.02531, 2015.

[14] A. Robins, "Catastrophic forgetting, rehearsal and pseudorehearsal," *Connection
Science*, vol. 7, no. 2, pp. 123–146, 1995. doi:10.1080/09540099550039318

[15] D. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, and G. Wayne, "Experience replay for
continual learning," in *Advances in NeurIPS 32*, 2019. arXiv:1811.11682

[16] A. Chaudhry, M. Rohrbach, M. Elhoseiny, T. Ajanthan, P. K. Dokania, P. H. S. Torr, and
M. Ranzato, "On tiny episodic memories in continual learning," arXiv:1902.10486, 2019.

[17] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, "iCaRL: Incremental
classifier and representation learning," in *Proc. CVPR*, pp. 2001–2010, 2017.

[18] Y. Wu, Y. Chen, L. Wang, Y. Ye, Z. Liu, Y. Guo, and Y. Fu, "Large scale incremental
learning," in *Proc. CVPR*, pp. 374–382, 2019.

[19] B. Zhao, X. Xiao, G. Gan, B. Zhang, and S.-T. Xia, "Maintaining discrimination and
fairness in class incremental learning," in *Proc. CVPR*, 2020. arXiv:1911.07053

[20] T. Mensink, J. Verbeek, F. Perronnin, and G. Csurka, "Distance-based image
classification: Generalizing to new classes at near-zero cost," *IEEE TPAMI*, vol. 35,
no. 11, pp. 2624–2637, 2013. doi:10.1109/TPAMI.2013.83

[21] P. Janson, W. Zhang, R. Aljundi, and M. Elhoseiny, "A simple baseline that questions the
use of pretrained-models in continual learning," Workshop on Distribution Shifts, 2022.
arXiv:2210.04428

[22] D.-W. Zhou, Z.-W. Cai, H.-J. Ye, D.-C. Zhan, and Z. Liu, "Revisiting class-incremental
learning with pre-trained models: Generalizability and adaptivity are all you need,"
*International Journal of Computer Vision*, vol. 133, no. 3, pp. 1012–1032, 2025.
doi:10.1007/s11263-024-02218-0

[23] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image
recognition at scale," in *Proc. ICLR*, 2021. arXiv:2010.11929

[24] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, "Training
data-efficient image transformers & distillation through attention," in *Proc. ICML*,
PMLR 139, pp. 10347–10357, 2021.

[25] A. Steiner, A. Kolesnikov, X. Zhai, R. Wightman, J. Uszkoreit, and L. Beyer, "How to
train your ViT? Data, augmentation, and regularization in vision transformers,"
*Transactions on Machine Learning Research*, 2022. arXiv:2106.10270

[26] D. Lopez-Paz and M. Ranzato, "Gradient episodic memory for continual learning," in
*Advances in NIPS 30*, 2017. arXiv:1706.08840. Official paper:
<https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf>

[27] A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. S. Torr, "Riemannian walk for
incremental learning: Understanding forgetting and intransigence," in *Proc. ECCV*,
pp. 556–572, 2018. doi:10.1007/978-3-030-01252-6_33

[28] A. Kuznetsova et al., "The Open Images Dataset V4," *International Journal of Computer
Vision*, vol. 128, no. 7, pp. 1956–1981, 2020. doi:10.1007/s11263-020-01316-z
