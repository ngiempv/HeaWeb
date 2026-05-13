import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from ..preprocessing import Preprocessing

class YOLOv5Wrapper:
    """
    Wrapper cho YOLO qua package ultralytics: train, detect, export.
    Ten file giu la yolov5.py de khong lam vo cac import cu trong project.
    """
    def __init__(self, weights=None, img_size=640, device=None, num_classes=5):
        """
        :param weights: path đến pretrained weights (.pt) hoặc None
        :param img_size: size input YOLOv5
        :param device: 'cpu' hoặc 'cuda'
        :param num_classes: số class
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Keep Ultralytics config inside the project so notebook/web runs do not
        # fail on restricted Windows profile directories.
        project_root = Path(__file__).resolve().parents[2]
        yolo_config_dir = project_root / ".ultralytics"
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))

        # Load model YOLO from ultralytics.
        from ultralytics import YOLO
        if weights is not None:
            self.model = YOLO(weights)
        else:
            self.model = YOLO('yolov8n.yaml')
        self.model.to(self.device)

    def train(self, data_yaml, epochs=10, lr=0.01, batch=4, project="runs", name="rbc_yolo"):
        """
        Train YOLO bang format dataset cua Ultralytics.
        data_yaml la file .yaml co train/val/images/labels.
        """
        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.img_size,
            batch=batch,
            lr0=lr,
            device=self.device,
            project=project,
            name=name,
        )

    def detect(self, imgs):
        """
        Detect cells trong ảnh mới
        :param imgs: list of PIL Image hoặc tensor [C,H,W]
        :return: predictions YOLOv5
        """
        results = []
        self.model.eval()
        with torch.no_grad():
            for img in imgs:
                if torch.is_tensor(img):
                    img = (img.permute(1,2,0).cpu().numpy()*255).astype('uint8')
                pred = self.model.predict(img)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                if hasattr(pred, 'boxes'):
                    boxes = pred.boxes
                    if len(boxes) == 0:
                        results.append(np.zeros((0, 6), dtype=np.float32))
                    else:
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls = boxes.cls.cpu().numpy().reshape(-1, 1)
                        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
                        arr = np.concatenate([cls, xyxy, conf], axis=1).astype(np.float32)
                        results.append(arr)
                else:
                    results.append(pred)
        return results

    def save(self, path):
        """Lưu weights"""
        self.model.save(path)

    def load(self, path):
        """Load weights"""
        self.model = torch.load(path, map_location=self.device)
