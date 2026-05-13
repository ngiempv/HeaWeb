import torch
from torch.utils.data import DataLoader
from .model.yolov5 import YOLOv5Wrapper
from .model.wbc_classifier import WBCClassifier, build_wbc_transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import datasets

class Evaluator:
    """
    Class để đánh giá mô hình SparseRCNN hoặc YOLOv5
    """
    def __init__(self, model_type='sparse_rcnn', model_weights=None, img_size=224, device=None, classes=None):
        """
        :param model_type: 'sparse_rcnn' hoặc 'yolov5'
        :param model_weights: đường dẫn checkpoint
        :param img_size: resize ảnh
        :param device: 'cuda' hoặc 'cpu'
        :param classes: list class name
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        if classes is not None:
            self.classes = classes
        elif model_type == 'classification':
            self.classes = ['NEUTROPHIL','LYMPHOCYTE','MONOCYTE','EOSINOPHIL','BASOPHIL']
        else:
            self.classes = ['RBC']

        if model_type == 'classification':
            self.model = None
        elif model_type == 'sparse_rcnn':
            try:
                from .model.sparse_rcnn import SparseRCNNModel
            except ImportError as e:
                raise ImportError(
                    "SparseRCNN is unavailable in this environment. "
                    "Install a torchvision version that supports SparseRCNN, or use model_type='yolov5'."
                ) from e

            self.model = SparseRCNNModel(num_classes=len(self.classes))
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = YOLOv5Wrapper(weights=model_weights, num_classes=len(self.classes), img_size=img_size)

    @classmethod
    def for_wbc_classifier(cls, weights_path, img_size=224, device=None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(weights_path, map_location=device)
        class_names = ckpt.get('class_names')
        backbone = ckpt.get('backbone', 'resnet18')
        model = WBCClassifier(num_classes=len(class_names), backbone=backbone, pretrained=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        obj = cls.__new__(cls)
        obj.model_type = 'classification'
        obj.device = device
        obj.img_size = img_size
        obj.classes = class_names
        obj.model = model
        obj.class_names = class_names
        obj.backbone = backbone
        return obj

    def evaluate_dataset(self, dataset, batch_size=2):
        """
        Đánh giá toàn bộ dataset
        :param dataset: CellDataset detection
        """
        if self.model_type == 'classification':
            raise ValueError('Use evaluate_wbc_classifier for classification tasks.')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        all_preds = []
        all_labels = []

        for imgs, targets in dataloader:
            if self.model_type == 'sparse_rcnn':
                imgs = [img.to(self.device) for img in imgs]
                targets_cuda = [{k: v.to(self.device) for k,v in t.items()} for t in targets]
                with torch.no_grad():
                    predictions = self.model(imgs)
                # predictions: list of dicts {'boxes', 'labels', 'scores'}
                for pred, target in zip(predictions, targets):
                    all_preds.extend(pred['labels'].cpu().numpy())
                    all_labels.extend(target['labels'].cpu().numpy())
            else:
                # YOLOv5
                imgs_np = []
                for img in imgs:
                    imgs_np.append((img.permute(1,2,0).cpu().numpy()*255).astype('uint8'))
                with torch.no_grad():
                    predictions = self.model.detect(imgs_np)
                for pred, target in zip(predictions, targets):
                    if len(pred) == 0:
                        continue
                    all_preds.extend(pred[:,0].astype(int))  # class index
                    all_labels.extend(target['labels'].cpu().numpy())

        # Tính metrics
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        return precision, recall, f1, cm

    def visualize_prediction(self, img_tensor, target=None, show=True, save_path=None):
        """
        Hiển thị bounding boxes + class
        :param img_tensor: tensor [C,H,W]
        :param target: dict {'boxes', 'labels'} optional
        """
        if self.model_type == 'classification':
            raise ValueError('visualize_prediction is not intended for classification tasks.')
        if self.model_type == 'sparse_rcnn':
            self.model.eval()
            with torch.no_grad():
                pred = self.model([img_tensor.to(self.device)])[0]
        else:
            img_np = (img_tensor.permute(1,2,0).cpu().numpy()*255).astype('uint8')
            pred = self.model.detect([img_tensor])[0]

        img = img_tensor.permute(1,2,0).cpu().numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw GT boxes
        if target is not None:
            for box, label in zip(target['boxes'], target['labels']):
                xmin, ymin, xmax, ymax = box.cpu().numpy()
                rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin-5, f"GT:{self.classes[label]}", color='green', fontsize=8)

        # Draw predicted boxes
        if self.model_type == 'sparse_rcnn':
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
        else:
            boxes = pred[:,1:5]
            labels = pred[:,0].astype(int)
            scores = pred[:,5] if pred.shape[1]>5 else np.ones(len(labels))

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

    def evaluate_wbc_classifier(self, imagefolder_root, batch_size=16):
        """
        Danh gia classifier tren ImageFolder.
        """
        if self.model_type not in ['classification', 'wbc_classifier']:
            raise ValueError('evaluate_wbc_classifier requires a classification evaluator')

        ds = datasets.ImageFolder(imagefolder_root, transform=build_wbc_transforms(self.img_size, train=False))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y_true = []
        y_pred = []

        self.model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(imgs)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        return acc, precision, recall, f1, cm
