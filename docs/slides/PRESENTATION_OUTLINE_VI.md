# Dàn ý thuyết trình (13 slide, khoảng 13–16 phút)

Số liệu lấy từ `RESULTS.md` và `outputs/*.csv`. Hình lấy từ `outputs/final_figures/`.
`[TODO]` là chỗ nhóm cần điền.

---

## Slide 1. Tiêu đề

- Phân loại ảnh liên tục không quên thảm họa
- So sánh Naive, EWC, LwF, Replay, NCM và Joint trên giao thức class-incremental 4 stage
- [TODO: tên nhóm, thành viên, môn học, ngày]

**Ghi chú người nói:** Giới thiệu nhóm, đề tài số 26 trong danh sách.

---

## Slide 2. Vấn đề: catastrophic forgetting

- Dữ liệu đến dần theo thời gian; không thể (hoặc không nên) lưu lại hết dữ liệu cũ
- Học tiếp trên dữ liệu mới → mạng quên nhanh kiến thức cũ [1, 3]
- Hình minh họa: một đường accuracy class cũ rơi mạnh sau khi học class mới (dùng Naive trong
  `stage_accuracy.png`)

**Ghi chú:** Nêu ví dụ thực tế: hệ thống nhận diện thêm loại vật thể mới mà không muốn huấn
luyện lại từ đầu.

---

## Slide 3. Bài toán: class-incremental 4 stage

| Stage | Class mới | Đánh giá trên |
|---|---|---|
| 0 | Dog, Cat | 2 class |
| 1 | Car | 3 class |
| 2 | Person | 4 class |
| 3 | Building | 5 class |

- Một head 5 output dùng chung; khi test, mô hình **không biết** ảnh thuộc stage nào
- Đây là kịch bản khó nhất trong ba kịch bản của van de Ven & Tolias [6]
- Điểm đặc biệt: mỗi stage sau chỉ có **một** class mới

**Ghi chú:** Nhấn mạnh điểm "một class mỗi stage", vì nó giải thích phần lớn kết quả ở slide 8.

---

## Slide 4. Dữ liệu

- Open Images V7, crop theo bounding box, 500 ảnh mỗi class, tổng 2.500
- Chia 400 / 50 / 50 mỗi class (train / val / test)
- Đã kiểm tra trùng lặp, rò rỉ ảnh gốc giữa các tập, rà soát bằng contact sheet
- [TODO: chèn 1 hàng ảnh ví dụ mỗi class, ghi Open Images image ID; tránh ảnh nhận diện được
  cá nhân]

**Ghi chú:** Giải thích vì sao gộp split validation/test chính thức rồi chia lại (metadata split
train quá lớn so với RAM máy sinh viên).

---

## Slide 5. Sáu phương pháp

| Nhóm | Phương pháp | Ý tưởng một dòng |
|---|---|---|
| Mốc dưới | Naive | Fine-tune tiếp, không làm gì thêm |
| Regularization | EWC | Phạt thay đổi tham số quan trọng (Fisher) |
| Regularization | LwF | Giữ đầu ra cũ bằng distillation, không lưu ảnh |
| Rehearsal | Replay | Lưu 200 ảnh cũ, trộn 50/50 vào mỗi batch |
| Biểu diễn cố định | Frozen ViT-Tiny + NCM | Đóng băng backbone, mỗi class là một prototype |
| Mốc trên | Joint | Học trên toàn bộ dữ liệu đã thấy (không phải continual) |

- Chung: ViT-Tiny pretrained, AdamW lr 1e-4, batch 32, 20 + 15 epoch, 3 seed, Tesla T4

**Ghi chú:** Nói rõ Joint chỉ là mốc so sánh, dùng lại toàn bộ dữ liệu cũ.

---

## Slide 6. Metric

- **Final accuracy**: accuracy trên 5 class sau Stage 3
- **Average incremental accuracy**: trung bình accuracy sau mỗi stage [17]
- **Forgetting**: accuracy cao nhất từng đạt − accuracy cuối, trung bình theo class cũ [27]
- **BWT**: accuracy cuối − accuracy ngay sau khi học class đó [26]
- **Chi phí**: thời gian thực, peak bộ nhớ PyTorch cấp phát trên GPU

**Ghi chú:** Nói trước rằng đồ án tính forgetting/BWT theo class, không theo task, để tránh bị
hỏi ở phần phản biện.

---

## Slide 7. Kết quả chính

| Phương pháp | Final acc. (%) | Forgetting (%) | Thời gian (s) | Peak GPU (MiB) |
|---|---:|---:|---:|---:|
| Naive | 20,00 ± 0,00 | 96,67 | 501 | 646 |
| EWC | 20,00 ± 0,00 | 96,83 | 552 | 1039 |
| LwF | 20,00 ± 0,00 | 97,67 | 491 | 683 |
| Replay | 91,20 ± 5,01 | 8,50 | 763 | 646 |
| NCM | 92,67 ± 0,46 | 2,67 | 42 | 104 |
| Joint | 95,20 ± 1,39 | 3,00 | 1247 | 646 |

- Hình: `final_accuracy.png` hoặc `stage_accuracy.png`

**Ghi chú:** Ba phương pháp đầu kết thúc ở đúng 20% = đoán ngẫu nhiên trên 5 class. Replay và
NCM sát mốc trên Joint.

---

## Slide 8. Vì sao Naive, EWC, LwF sụp về 20%?

- Sau Stage 3: Building 100%, mọi class khác 0%, ở cả 9 run → mô hình đoán mọi ảnh là class vừa
  học
- Mỗi stage chỉ có một class → cross-entropy giảm bằng cách luôn đoán class mới
- LwF chỉ giữ thứ tự giữa các class cũ, không chặn class mới vượt lên [19]
- Ma trận nhầm lẫn LwF (seed 2026, sau Stage 1): 29 / 30 lỗi trên Dog/Cat là đoán thành Car

| Thực tế \ Dự đoán | Dog | Cat | Car |
|---|---:|---:|---:|
| Dog | 37 | 1 | 12 |
| Cat | 0 | 33 | 17 |
| Car | 0 | 0 | 50 |

**Ghi chú:** Đây là slide quan trọng nhất về mặt phân tích. Kết luận thận trọng: thất bại **dưới
siêu tham số đã thử và giao thức một-class-mỗi-stage**, phù hợp với tài liệu [6, 19], nhưng không
chứng minh EWC/LwF luôn thất bại.

---

## Slide 9. Replay và NCM: hai cách giữ trí nhớ

- **Replay**: 200 ảnh (10% tập train) giảm forgetting từ ~97% xuống 8,5%; nhưng biến thiên lớn
  (86,0 / 91,6 / 96,0 theo seed)
- **NCM**: 92,67%, forgetting 2,67%, nhanh hơn Replay ~18 lần, bộ nhớ GPU thấp hơn ~84%
- NCM mạnh vì đặc trưng pretrain ImageNet-21k đã tách tốt 5 khái niệm phổ biến này [21, 22]
- Chênh lệch NCM − Replay (1,47 điểm) **nhỏ hơn** độ lệch chuẩn của Replay → không kết luận được
  bên nào tốt hơn

**Ghi chú:** Nói rõ NCM không phải phát hiện mới; tài liệu đã ghi nhận frozen pretrained + NCM là
baseline mạnh.

---

## Slide 10. NCM mạnh vì đặc trưng pretrained, không phải vì luật NCM

- Cùng luật NCM, cùng dữ liệu, cùng seed; chỉ đổi backbone:

| Backbone | Final accuracy (%) | Forgetting (%) |
|---|---:|---:|
| Pretrained ImageNet-21k | 92,67 ± 0,46 | 2,67 ± 0,58 |
| Random init, Naive trên Stage 0 (800 ảnh) | 25,47 ± 1,40 | 21,83 ± 4,80 |
| Random init, không huấn luyện | 23,47 ± 1,01 | 22,67 ± 2,02 |

- Hình: `backbone_source_final_accuracy.png` và `backbone_source_stage_accuracy.png`
- Arm Stage 0 tự huấn luyện backbone bằng đúng thuật toán Naive trên Dog + Cat; NCM không tạo
  optimizer nên phải huấn luyện riêng rồi nạp `backbone.*` (bỏ head 5 lớp)
- **Kết luận được:** 92,67% không do bản thân luật NCM tạo ra. Hai backbone không có trọng số
  ImageNet-21k nằm gần mức ngẫu nhiên 20%
- **Không kết luận được:** arm Stage 0 là giám sát rẻ trong miền (800 ảnh, 2 class, 500 bước),
  không phải "không pretrain"; và nó underfit — run Naive của nó chỉ đạt 57–59% ở Stage 0. Không
  nói "ViT không train được từ đầu"
- Chênh lệch 2,00 điểm giữa hai arm cuối **nằm trong nhiễu** (khoảng seed có chồng lấn)

**Ghi chú:** Đây là slide trả lời trực tiếp câu hỏi "NCM có ăn gian không?". Nói thẳng: NCM
không gian lận giao thức, nhưng kết quả của nó phụ thuộc giả định về đặc trưng pretrained.

---

## Slide 11. Chi phí

- Hình: `runtime.png` và `gpu_memory.png`
- Thời gian phụ thuộc cấu hình epoch; NCM không có bước huấn luyện nào
- Peak memory cao của EWC đến từ bước ước lượng Fisher (chạy không dùng mixed precision), không
  phải bản chất EWC
- Bộ nhớ phụ: Replay lưu 200 ảnh; NCM lưu 5 vector 192 chiều; EWC lưu 2 bản sao tham số; LwF giữ
  một teacher

**Ghi chú:** Peak memory ở đây là bộ nhớ PyTorch cấp phát, không phải toàn bộ VRAM.

---

## Slide 12. Hạn chế

- Huấn luyện GPU không tất định: Stage 0 giống hệt nhau mà dao động tới 8 điểm (seed 42)
- 3 seed, test 50 ảnh/class, không kiểm định thống kê
- Không tinh chỉnh siêu tham số EWC/LwF/Replay
- Một thứ tự class; giao thức một-class-mỗi-stage là trường hợp cực đoan
- Ảnh crop, miền gần ImageNet → có lợi cho NCM; lợi thế này đã đo ở slide 10 nhưng chưa thử
  trên miền xa ImageNet, class tinh hay class đa dạng hình thức

---

## Slide 13. Kết luận và hướng phát triển

- Naive quên hoàn toàn; EWC, LwF (cấu hình đã thử) không cứu được trong class-incremental một-head
- Replay hiệu quả nhưng tốn lưu trữ và biến thiên; NCM rẻ, ổn định, **nhưng phụ thuộc miền
  pretrain**
- Hướng tiếp: tinh chỉnh `λ`, `α`; thêm hiệu chỉnh bias (BiC, WA); herding cho Replay; miền dữ
  liệu xa ImageNet; chạy tất định và nhiều seed hơn

**Ghi chú:** Kết bằng thông điệp: "Không có phương pháp miễn phí: chống quên luôn đổi bằng lưu
trữ dữ liệu, tính toán, hoặc giả định về đặc trưng pretrained."

---

## Câu hỏi phản biện dự kiến

**1. Vì sao EWC không tốt hơn Naive chút nào? Có phải cài sai không?**
Công thức phạt khớp Kirkpatrick et al. [9]. Nguyên nhân có thể là `λ = 100` quá nhỏ, và empirical
Fisher trên dữ liệu một-class gần bằng 0 sau khi mô hình khớp dữ liệu. Quan trọng hơn, EWC không
xử lý thiên lệch ở lớp phân loại cuối trong class-incremental; tài liệu [6] cũng báo EWC thất bại
ở kịch bản này. Nhóm chưa quét `λ`, nên đây là hạn chế đã nêu.

**2. LwF không cần ảnh cũ mà vẫn hay được dùng. Sao ở đây lại sụp?**
LwF gốc có head riêng cho mỗi task [12]. Ở đây chỉ có một head chung và loss distillation chỉ
trên logit class cũ, nên nó giữ Dog so với Cat nhưng không ngăn Car vượt lên. Ma trận nhầm lẫn ở
slide 8 cho thấy đúng điều đó.

**3. NCM gần như không học gì, vậy có phải "ăn gian" không?**
NCM tuân thủ giao thức: không dùng ảnh cũ, chỉ lưu vector trung bình. Nhóm đã đo trực tiếp (slide
10): giữ nguyên luật NCM và dữ liệu, chỉ đổi backbone, accuracy rơi từ 92,67% xuống 23,47% khi
backbone random init. Vậy NCM không gian lận giao thức, nhưng toàn bộ sức mạnh của nó đến từ giả
định về đặc trưng pretrained. Đó là lý do nhóm trình bày nó là baseline mạnh cho miền gần
ImageNet, không phải lời giải tổng quát.

**4. Có thể nói NCM tốt hơn Replay không?**
Không. Chênh lệch 1,47 điểm nhỏ hơn độ lệch chuẩn 5,01 của Replay, thứ hạng đổi chiều theo seed,
và chỉ có 3 seed. Kết luận đúng là tương đương về accuracy nhưng rẻ và ổn định hơn.

**5. Joint là gì, sao có forgetting?**
Joint fine-tune tiếp trên toàn bộ dữ liệu đã thấy ở mỗi stage, nên vẫn có thể giảm accuracy một
chút ở class cũ. Nó chỉ là mốc trên gần đúng, không phải phương pháp continual learning.

**6. Vì sao forgetting bằng đúng −BWT ở nhiều phương pháp?**
Hai metric trùng nhau khi accuracy mỗi class chỉ giảm sau khi học. Chúng khác nhau khi accuracy
tăng rồi giảm, như Joint (3,00 so với −1,83).

**7. Sao accuracy của NCM tăng từ Stage 0 lên Stage 1?**
Accuracy là trung bình theo class. Dog/Cat là cặp khó nhất; thêm Car dễ phân biệt làm trung bình
tăng. Không có nghĩa là mô hình học ngược.

**8. Kết quả có tái lập được không?**
Có mã, config, manifest dữ liệu và 18 file summary gốc (`outputs/colab_runs`). Tuy vậy huấn luyện
GPU không tất định: Stage 0 giống nhau về thuật toán vẫn dao động tới 8 điểm, nên chạy lại có thể
cho số hơi khác.

**9. Vì sao chọn ViT-Tiny?**
[TODO: nhóm trả lời, ví dụ: nhỏ, chạy được trên T4 miễn phí của Colab trong thời gian đồ án, có
trọng số pretrained sẵn trong timm.]

**10. Nếu có thêm thời gian, nhóm sẽ làm gì trước tiên?**
Quét siêu tham số EWC/LwF trên validation, bật chế độ tất định và dùng chung checkpoint Stage 0,
rồi thử giao thức nhiều class mỗi stage để kiểm tra thất bại của EWC/LwF có phụ thuộc giao thức
hay không.
