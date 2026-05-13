import torch
from .model.yolov5 import YOLOv5Wrapper
from .preprocessing import Preprocessing
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class Predictor:
    """
    Class để dự đoán tế bào mới với SparseRCNN hoặc YOLOv5
    """
    def __init__(self, model_type='sparse_rcnn', weights=None, img_size=224, device=None, classes=None):
        """
        :param model_type: 'sparse_rcnn' hoặc 'yolov5'
        :param weights: checkpoint path
        :param img_size: resize ảnh
        :param device: 'cuda' hoặc 'cpu'
        :param classes: list class name
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.classes = classes or ['RBC']

        # Load model
        if model_type == 'sparse_rcnn':
            try:
                from .model.sparse_rcnn import SparseRCNNModel
            except ImportError as e:
                raise ImportError(
                    "SparseRCNN is unavailable in this environment. "
                    "Install a torchvision version that supports SparseRCNN, or use model_type='yolov5'."
                ) from e

            self.model = SparseRCNNModel(num_classes=len(self.classes))
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = YOLOv5Wrapper(weights=weights, num_classes=len(self.classes), img_size=img_size)

        self.preprocessor = Preprocessing(img_size=(img_size, img_size))

    def predict_image(self, img_path):
        """
        Predict 1 ảnh
        :param img_path: đường dẫn ảnh
        :return: predictions dict: boxes, labels, scores
        """
        img_tensor, boxes, labels = self.preprocessor.preprocess_detection(img_path)
        if img_tensor is None:
            return None

        if self.model_type == 'sparse_rcnn':
            with torch.no_grad():
                pred = self.model([img_tensor.to(self.device)])[0]
        else:
            pred = self.model.detect([img_tensor])[0]

        return img_tensor, pred

    def visualize_prediction(self, img_tensor, predictions, save_path=None, show=True):
        """
        Hiển thị bounding boxes + labels + scores
        :param img_tensor: tensor [C,H,W]
        :param predictions: dict hoặc numpy array (tùy YOLO/SparseRCNN)
        """
        img = img_tensor.permute(1,2,0).cpu().numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        if self.model_type == 'sparse_rcnn':
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
        else:
            boxes = predictions[:,1:5]
            labels = predictions[:,0].astype(int)
            scores = predictions[:,5] if predictions.shape[1]>5 else [1.0]*len(labels)

        for box, label, score in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin-5, f"{self.classes[label]}:{score:.2f}", color='red', fontsize=8)

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    def predict_folder(self, folder_path, save_dir=None):
        """
        Predict tất cả ảnh trong folder
        :param folder_path: folder chứa ảnh
        :param save_dir: lưu ảnh output nếu muốn
        """
        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png','jpg','jpeg'))]
        os.makedirs(save_dir, exist_ok=True) if save_dir else None

        for fname in img_files:
            img_path = os.path.join(folder_path, fname)
            img_tensor, pred = self.predict_image(img_path)
            if img_tensor is None:
                print(f"[WARN] Skipping {fname}, image invalid or blurry")
                continue
            out_path = os.path.join(save_dir, fname) if save_dir else None
            self.visualize_prediction(img_tensor, pred, save_path=out_path)
            print(f"[INFO] Processed {fname}")

    def predict_image_path(self, img_path):
        """
        Alias nho de notebook goi nhanh.
        """
        return self.predict_image(img_path)
