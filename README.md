XLAS - Nhận dạng tế bào máu bất thường (Blood Cell Classification)
     - mô tả: Phân loại tế bào máu/ 
     - Recognition


## Mục tiêu

- Phát hiện hồng cầu (`RBC`) trên ảnh kính hiển vi.
- Dùng vùng RBC đã phát hiện để loại bỏ hoặc bỏ qua hồng cầu khi tìm vùng nghi là bạch cầu.
- Tạo vùng ứng viên WBC bằng xử lý ảnh số, không phụ thuộc cứng vào màu tím.
- Crop vùng ứng viên WBC và phân loại bằng classifier đã train từ `dataset2-master`.
- Chuẩn bị pipeline để sau này bọc thành web upload ảnh và hiển thị kết quả.

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

### 1. Detector RBC

Dữ liệu vào:

- `dataset-master/JPEGImages`
- `dataset-master/Annotations`

Nhiệm vụ:

- Phát hiện bounding box của RBC.

Danh sách class detection:

```python
["RBC"]
```

Kết quả:

- Box RBC.
- Điểm tin cậy của RBC.

### 2. Classifier WBC

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

### 3. Đề xuất vùng WBC bằng xử lý ảnh số

Vì `dataset-master` không có bounding box cho WBC, nên box ứng viên WBC cần
được tạo bằng các phương pháp xử lý ảnh số trước khi đưa vào classifier.

Lưu ý quan trọng: không giả định tất cả WBC đều có màu tím. Màu tím của nhân tế
bào có thể là một dấu hiệu hữu ích, nhưng không nên là điều kiện duy nhất.

Nên kết hợp nhiều đặc trưng ảnh:

- Tương phản màu trong HSV hoặc LAB khi nhân tế bào hiện rõ.
- Độ bão hòa màu và độ sáng khác với RBC/background.
- Texture và mật độ biên.
- Diện tích, tỉ lệ khung hình, độ tròn, hình dạng contour.
- Lọc bớt các vùng trùng lặp với box RBC đã detect.

Quy trình xử lý ảnh có thể gồm:

1. Đọc ảnh RGB/BGR.
2. Chuyển sang HSV và/hoặc LAB.
3. Tạo mask ứng viên dựa trên màu, saturation, brightness và texture.
4. Làm sạch mask bằng morphology như opening và closing.
5. Tìm contour hoặc connected components.
6. Lọc vùng theo diện tích, hình dạng và overlap với RBC.
7. Trả về các bounding box ứng viên WBC.

Kết quả:

- Các box có khả năng chứa WBC.

### 4. Inference kết hợp

Với mỗi ảnh kính hiển vi:

1. Chạy detector RBC.
2. Vẽ box RBC từ output của detector.
3. Chạy module đề xuất vùng WBC bằng xử lý ảnh số.
4. Crop từng box ứng viên WBC.
5. Đưa mỗi crop vào classifier WBC.
6. Vẽ box WBC kèm nhãn class dự đoán.

Luồng tổng thể:

```text
image
  |-- RBC detector
  |     `-- RBC boxes
  |
  `-- WBC proposal bằng xử lý ảnh số
        `-- candidate WBC boxes
              `-- crop
                    `-- WBC classifier
                          `-- NEUTROPHIL / LYMPHOCYTE / MONOCYTE / EOSINOPHIL / ...
```

## Kế hoạch train

### Train RBC detection

1. Convert XML annotation với class list `["RBC"]`.
2. Train Sparse-RCNN hoặc YOLO trên box RBC.
3. Đánh giá bằng IoU, precision, recall và mAP cho RBC.

### Train WBC classification

1. Load `dataset2-master`.
2. Train CNN classification hoặc transfer learning.
3. Đánh giá bằng accuracy, precision, recall, F1 và confusion matrix.

### Xây dựng module WBC proposal

1. Dùng các kỹ thuật xử lý ảnh số cổ điển.
2. Tune rule trên các ảnh mẫu.
3. Kiểm tra bằng mắt xem các box ứng viên có bao được vùng WBC hay không.

## Đánh giá

RBC detector:

- IoU
- Precision
- Recall
- mAP

WBC classifier:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

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

### Bước 1: Convert annotation RBC

```bash
python -m src.annotation_converter
```

Sau bước này, file annotation CSV sẽ được tạo tại:

```text
dataset-master/dataset-master/YOLO_labels/annotations_converted.csv
```

### Bước 2: Train RBC detector

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

### Bước 3: Train WBC classifier

```bash
python -m src.train ^
  --task classification ^
  --train_root dataset2-master/dataset2-master/images/TRAIN ^
  --val_root dataset2-master/dataset2-master/images/TEST ^
  --backbone resnet18 ^
  --img_size 224 ^
  --batch_size 16 ^
  --epochs 10 ^
  --save_dir saved_models
```

Sau khi train xong, weights WBC classifier nằm ở:

```text
saved_models/wbc_classifier_best.pth
```

Đường dẫn dùng trong notebook:

```python
WBC_WEIGHTS = PROJECT_ROOT / "saved_models" / "wbc_classifier_best.pth"
```

### Bước 4: Chạy lại notebook sau khi có weights

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