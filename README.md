XLAS - Phân biệt tế bào máu bất thường
Step 1: Pretrain / Baseline (optional)
  └─ Task: Classification CNN / Transfer Learning
  └─ Input: dataset2-master
  └─ Output: Model nhận dạng tế bào bình thường

Step 2: Preprocessing
  └─ Resize, normalize, augment
  └─ Lọc ảnh nhòe / lỗi

Step 3: Dataset Loader
  ├─ Classification loader (dataset2-master)
  └─ Detection loader (dataset-master)
      └─ Output: image tensor + boxes + labels

Step 4: Model
  ├─ Sparse-RCNN / YOLO (detection + abnormal classification)
  └─ Optional: Transfer Learning từ pretrain CNN (Step 1)

Step 5: Training
  └─ Forward pass → Loss (classification + bbox regression) → Backprop → Optimizer → Checkpoint

Step 6: Evaluation
  ├─ mAP, IoU → bounding boxes
  ├─ Precision, Recall, F1 → class abnormal
  └─ Confusion matrix → visual inspection

Step 7: Prediction
  └─ Load image mới → Detect all cells → Classify normal / abnormal → visualize bounding box + class + score

Step 8: Demo
  └─ Jupyter Notebook / run_demo.py
      └─ Upload image → Detect + Highlight abnormal cells → Count cells
