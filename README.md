XLAS - Nhận dạng tế bào máu bất thường (Blood Cell Classification)
     - mô tả: Phân loại tế bào máu/ 
     - Recognition


## Mục tiêu

- Nhiệm vụ chính của project là **Recognition (Classification)**.
- Train classifier để phân loại tế bào máu, trọng tâm hiện tại là các lớp WBC trong `dataset2-master`.
- Tạo vùng crop tự động từ ảnh kính hiển vi bằng xử lý ảnh số và/hoặc detector hỗ trợ.
- Dùng RBC detector như một bước phụ để nhận diện và loại bớt vùng hồng cầu khi tìm vùng nghi WBC.
- Tách code core khỏi notebook để có thể tái sử dụng cho demo web upload ảnh.

## Tình trạng dataset

Trong `dataset-master`, các file XML hiện tại chỉ có bounding box cho đối tượng
`RBC`.

Vì vậy, phần annotation XML này nên được dùng để train model phát hiện RBC,
không nên dùng để train trực tiếp detector nhiều lớp như `NEUTROPHIL`,
`LYMPHOCYTE`, `MONOCYTE`, `EOSINOPHIL`.

Nếu train detector chỉ bằng annotation RBC, các vùng WBC chưa được annotate sẽ
bị xem là background trong quá trình train detection.

Nhãn lớp WBC nằm ở `dataset2-master`, phù hợp hơn cho bài toán phân loại ảnh
single-cell.

## Pipeline đề xuất

### 1. Classifier WBC là phần chính

Dữ liệu vào:

- `dataset2-master`

Nhiệm vụ:

- Train model phân loại ảnh WBC dạng single-cell.

Một số class đầu ra có thể có:

```python
["NEUTROPHIL", "LYMPHOCYTE", "MONOCYTE", "EOSINOPHIL", "BASOPHIL"]
```

Kết quả:

- Tên loại WBC cho mỗi crop.
- Điểm tin cậy của classifier.

### 2. WBC proposal và crop tự động

Dữ liệu vào:

- Ảnh kính hiển vi đầy đủ từ `dataset-master` hoặc ảnh upload bên ngoài.

Nhiệm vụ:

- Tạo box ứng viên WBC bằng xử lý ảnh số.
- Crop các vùng ứng viên để đưa vào classifier.

Lưu ý quan trọng: không giả định tất cả WBC đều có màu tím. Màu tím/xanh của
nhân tế bào chỉ là một cue phụ. Hướng chính là dựa vào:

- Độ tương phản với nền RBC.
- Texture và mật độ biên.
- Độ sáng, độ bão hòa và khác biệt màu tổng quát.
- Diện tích, tỉ lệ khung hình, hình dạng contour.
- Ưu tiên các vùng lớn trước vì WBC thường lớn hơn RBC.

Kết quả:

- Các box có khả năng chứa WBC.
- Các crop WBC candidate để classifier nhận dạng.

Quy trình xử lý ảnh có thể gồm:

1. Đọc ảnh RGB/BGR.
2. Tăng tương phản cục bộ bằng CLAHE.
3. Tạo mask ứng viên dựa trên contrast, texture, adaptive threshold và edge.
4. Làm sạch mask bằng morphology như opening và closing.
5. Tìm contour hoặc connected components.
6. Lọc vùng theo diện tích, hình dạng và overlap với RBC.
7. Trả về các bounding box ứng viên WBC.

### 3. Detector RBC là phần hỗ trợ

Dữ liệu vào:

- `dataset-master/JPEGImages`
- `dataset-master/Annotations`

Nhiệm vụ:

- Phát hiện bounding box của RBC.
- Dùng box RBC để loại bỏ hoặc bỏ qua vùng hồng cầu khi tạo WBC proposal.

Danh sách class detection:

```python
["RBC"]
```

Kết quả:

- Box RBC.
- Điểm tin cậy của RBC.

### 4. Inference kết hợp

Với mỗi ảnh kính hiển vi:

1. Chạy detector RBC nếu đã train weights.
2. Dùng box RBC để giảm nhiễu trong bước tìm WBC candidate.
3. Chạy module WBC proposal bằng xử lý ảnh số.
4. Crop từng box ứng viên WBC.
5. Đưa mỗi crop vào classifier WBC.
6. Vẽ box RBC, box WBC candidate và nhãn WBC dự đoán.

Luồng tổng thể:

```text
image
  |-- optional RBC detector
  |     `-- RBC boxes để loại vùng hồng cầu
  |
  `-- WBC proposal + crop tự động
        `-- candidate WBC crops
              `-- WBC classifier
                    `-- NEUTROPHIL / LYMPHOCYTE / MONOCYTE / EOSINOPHIL / ...
```

## Cấu trúc code chính

- `src/model/wbc_classifier.py`: model classifier WBC và hàm predict class.
- `src/wbc_proposal.py`: tạo box ứng viên WBC và crop tự động.
- `src/pipeline.py`: điều phối detector, proposal và classifier.
- `src/visualization.py`: vẽ RBC boxes, WBC candidate boxes và nhãn dự đoán.
- `src/preprocessing.py`: tách `ClassificationPreprocessing` và `DetectionPreprocessing`.
- `src/evaluate.py`: metric riêng cho classifier và detector hỗ trợ.
- `src/train.py`: train WBC classifier hoặc RBC detector hỗ trợ.

## Kế hoạch train

### Train WBC classification

1. Load `dataset2-master`.
2. Train CNN classification hoặc transfer learning.
3. Đánh giá bằng accuracy, precision, recall, F1 và confusion matrix.

Đây là phần chính vì nhiệm vụ của đề tài là Recognition (Classification).

### Train RBC detection hỗ trợ

1. Convert XML annotation với class list `["RBC"]`.
2. Train YOLO trên box RBC.
3. Đánh giá bằng IoU, precision, recall và F1 cho RBC.
4. Dùng output RBC để loại bớt vùng hồng cầu khi tìm WBC candidate.

### Xây dựng module WBC proposal

1. Dùng các kỹ thuật xử lý ảnh số cổ điển.
2. Ưu tiên contrast, texture và hình dạng so với nền RBC, không bám cứng vào màu tím.
3. Ưu tiên các candidate lớn trước.
4. Kiểm tra bằng mắt xem các box ứng viên có bao được vùng WBC hay không.

## Đánh giá

WBC classifier:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

RBC detector hỗ trợ:

- IoU
- Precision
- Recall
- F1-score

Demo end-to-end:

- Hiển thị box RBC.
- Hiển thị box ứng viên WBC.
- Hiển thị kết quả classifier trên từng crop WBC.

## Ghi chú quan trọng

Không nên kỳ vọng detector RBC sẽ tìm ra box WBC. Box WBC cần đến từ một bước
đề xuất vùng riêng bằng xử lý ảnh số, sau đó classifier WBC sẽ gán nhãn cuối
cùng cho từng crop.

## Cách run project để train đầy đủ

Chạy các lệnh dưới đây trong terminal tại thư mục gốc project:

```text
F:\py\XLAS\HeaWeb
```

### Bước 1: Cài thư viện cần thiết

```bash
pip install -r requirements.txt
```

Nếu máy đã cài đủ thư viện thì có thể bỏ qua bước này.

### Bước 2: Train WBC classifier

```bash
python -m src.train --task classification --train_root dataset2-master/dataset2-master/images/TRAIN --val_root dataset2-master/dataset2-master/images/TEST --backbone resnet18 --img_size 224 --batch_size 16 --epochs 10 --save_dir saved_models
```

Sau khi train xong, weights WBC classifier nằm ở:

```text
saved_models/wbc_classifier_best.pth
```

Đường dẫn dùng trong notebook:

```python
WBC_WEIGHTS = PROJECT_ROOT / "saved_models" / "wbc_classifier_best.pth"
```

### Bước 3: Convert annotation RBC cho detector hỗ trợ

```bash
python -m src.annotation_converter
```

Sau bước này, file annotation CSV sẽ được tạo tại:

```text
dataset-master/dataset-master/YOLO_labels/annotations_converted.csv
```

### Bước 4: Train RBC detector hỗ trợ

```bash
python -m src.train --task detection --model yolov5 --img_dir dataset-master/dataset-master/JPEGImages --ann_file dataset-master/dataset-master/YOLO_labels/annotations_converted.csv --classes RBC --img_size 640 --batch_size 4 --epochs 20 --save_dir saved_models
```

Sau khi train xong, weights RBC thường nằm ở:

```text
saved_models/rbc_yolo/weights/best.pt
```

Đường dẫn dùng trong notebook:

```python
RBC_WEIGHTS = PROJECT_ROOT / "saved_models" / "rbc_yolo" / "weights" / "best.pt"
```

### Bước 5: Chạy lại notebook sau khi có weights

Trong notebook, điền:

```python
RBC_WEIGHTS = PROJECT_ROOT / "saved_models" / "rbc_yolo" / "weights" / "best.pt"
WBC_WEIGHTS = PROJECT_ROOT / "saved_models" / "wbc_classifier_best.pth"
```

Sau đó chạy lại các cell:

1. Load WBC classifier.
2. Load RBC detector.
3. Vẽ RBC boxes để kiểm tra.
4. Chạy WBC proposal sau khi loại RBC.
5. Chạy pipeline end-to-end.

Nếu chưa train model và chỉ muốn test xử lý ảnh số, giữ:

```python
RBC_WEIGHTS = None
WBC_WEIGHTS = None
```
