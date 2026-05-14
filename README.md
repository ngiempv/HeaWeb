# XLAS - Nhận dạng tế bào máu bất thường

## Mục tiêu

Đề tài tập trung vào **Recognition (Classification)**.

- `dataset2-master` dùng để train **WBC classifier**.
- `dataset-master` dùng làm phần **hỗ trợ**:
  - RBC detector
  - WBC proposal / crop trên ảnh kính hiển vi lớn
  - demo pipeline end-to-end

Ý tưởng chính:

1. Nhận ảnh kính hiển vi.
2. Dùng RBC detector để giảm nhiễu từ vùng hồng cầu.
3. Dùng WBC proposal để khoanh vùng nghi bạch cầu.
4. Crop từng vùng nghi ngờ.
5. Dùng CNN classifier để gán loại WBC:
   - `NEUTROPHIL`
   - `EOSINOPHIL`
   - `LYMPHOCYTE`
   - `MONOCYTE`

## Tại sao không train detector đa lớp từ XML hiện có

Trong `dataset-master`, XML hiện có chỉ gắn nhãn cho `RBC`. Vì vậy:

- XML này phù hợp để train **RBC detector**.
- Không phù hợp để train trực tiếp detector đa lớp `WBC/RBC/Platelet`.
- WBC trong ảnh lớn chưa được annotate đầy đủ, nên nếu dùng cho detection đa lớp thì dễ coi WBC là background.

Vì vậy, hướng hợp lý hơn là:

- **WBC classification** là phần chính.
- **RBC detection** và **WBC proposal** là phần hỗ trợ để crop đúng vùng.

## Cấu trúc source

- `src/train.py`: train classifier WBC hoặc detector RBC hỗ trợ.
- `src/pipeline.py`: ghép `RBC -> WBC proposal -> WBC classifier`.
- `src/wbc_proposal.py`: xử lý ảnh số để đề xuất vùng nghi WBC.
- `src/visualization.py`: vẽ box và kết quả pipeline.
- `src/evaluate.py`: metric cho classifier và detector.
- `src/model/wbc_classifier.py`: model classifier WBC.
- `src/model/yolov5.py`: wrapper cho YOLO hỗ trợ RBC detector.

## Cách chạy

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Train WBC classifier

```bash
python -m src.train --task classification --train_root dataset2-master/dataset2-master/images/TRAIN --val_root dataset2-master/dataset2-master/images/TEST --backbone resnet18 --img_size 224 --batch_size 16 --epochs 10 --save_dir saved_models
```

Sau khi train xong, weights nằm tại:

```text
saved_models/wbc_classifier_best.pth
```

### 3. Convert annotation RBC

```bash
python -m src.annotation_converter
```

File kết quả:

```text
dataset-master/dataset-master/YOLO_labels/annotations_converted.csv
```

### 4. Train RBC detector hỗ trợ

```bash
python -m src.train --task detection --model yolov5 --img_dir dataset-master/dataset-master/JPEGImages --ann_file dataset-master/dataset-master/YOLO_labels/annotations_converted.csv --classes RBC --img_size 640 --batch_size 4 --epochs 20 --save_dir saved_models
```

Sau khi train xong, weights nằm tại:

```text
saved_models/rbc_yolo/weights/best.pt
```

### 5. Chạy notebook demo

Mở:

```text
src/notebooks/classification_demo.ipynb
```

Notebook này dùng để:

- xem ảnh single-cell
- test WBC classifier
- xem XML RBC
- xem WBC proposal
- test pipeline end-to-end

## Ghi chú quan trọng

- Không nên kỳ vọng RBC detector sẽ tự tìm được WBC.
- WBC proposal là bước xử lý ảnh số để đề xuất vùng nghi ngờ, không phải detector học sâu đa lớp.
- Nếu muốn test pipeline đúng, hãy train WBC classifier trước, rồi mới chạy notebook.

## Đánh giá

### WBC classifier

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### RBC detector hỗ trợ

- IoU
- Precision
- Recall
- F1-score

### Pipeline demo

- Số box RBC
- Số box WBC candidate
- Nhãn WBC dự đoán cho từng crop

