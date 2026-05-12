import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from ..preprocessing import Preprocessing

class YOLOv5Wrapper:
    """
    Wrapper cho YOLOv5: train, detect, export
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

        # Load model YOLOv5 từ ultralytics
        from yolov5 import YOLO  # pip install yolov5
        if weights is not None:
            self.model = YOLO(weights)
        else:
            # khởi tạo model mới với num_classes
            self.model = YOLO('yolov5s.yaml')  
            self.model.model.nc = num_classes
        self.model.to(self.device)

    def train(self, train_loader, val_loader=None, epochs=10, lr=0.01):
        """
        Train YOLOv5 trên CellDataset
        """
        for epoch in range(epochs):
            self.model.train()
            for imgs, targets in train_loader:
                # imgs: [B,C,H,W], targets: dict boxes + labels
                # YOLOv5 expects list of images in numpy uint8
                imgs_np = [img.permute(1,2,0).cpu().numpy()*255 for img in imgs]
                labels_list = []
                for t in targets:
                    boxes = t['boxes'].cpu().numpy()
                    labels = t['labels'].cpu().numpy()
                    # YOLOv5 format: [class, x_center, y_center, w, h] normalized
                    H,W = imgs_np[0].shape[:2]
                    boxes_norm = []
                    for i in range(len(labels)):
                        x_center = (boxes[i,0]+boxes[i,2])/2 / W
                        y_center = (boxes[i,1]+boxes[i,3])/2 / H
                        w = (boxes[i,2]-boxes[i,0])/W
                        h = (boxes[i,3]-boxes[i,1])/H
                        boxes_norm.append([labels[i], x_center, y_center, w, h])
                    labels_list.append(boxes_norm)
                # forward + loss
                self.model.train_step(imgs_np, labels_list)

            print(f"Epoch {epoch+1}/{epochs} done")

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
                results.append(pred)
        return results

    def save(self, path):
        """Lưu weights"""
        self.model.save(path)

    def load(self, path):
        """Load weights"""
        self.model = torch.load(path, map_location=self.device)