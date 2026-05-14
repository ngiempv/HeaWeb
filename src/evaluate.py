import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .model.wbc_classifier import WBCClassifier, build_wbc_transforms
from .model.yolov5 import YOLOv5Wrapper


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def classification_metrics_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    true_sum = cm.sum(axis=1).astype(np.float64)

    precision_per_class = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    recall_per_class = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0)
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(tp),
        where=(precision_per_class + recall_per_class) > 0,
    )

    total = cm.sum()
    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": float(precision_per_class.mean()) if len(precision_per_class) else 0.0,
        "recall": float(recall_per_class.mean()) if len(recall_per_class) else 0.0,
        "f1": float(f1_per_class.mean()) if len(f1_per_class) else 0.0,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm,
    }


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)

    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, ix2 - ix1)
    inter_h = np.maximum(0, iy2 - iy1)
    inter = inter_w * inter_h

    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_detection_boxes(pred_boxes, gt_boxes, pred_scores=None, iou_threshold=0.5):
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []

    if pred_scores is None:
        order = np.arange(len(pred_boxes))
    else:
        order = np.argsort(-np.asarray(pred_scores))

    matched_gt = set()
    matched_ious = []
    tp = 0
    fp = 0

    for pred_idx in order:
        ious = box_iou(pred_boxes[pred_idx], gt_boxes)
        best_gt = int(np.argmax(ious))
        best_iou = float(ious[best_gt])
        if best_iou >= iou_threshold and best_gt not in matched_gt:
            tp += 1
            matched_gt.add(best_gt)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matched_ious


class Evaluator:
    """
    Danh gia rieng classification va detection.
    Classification WBC la metric chinh cua de tai; detection RBC la metric ho tro.
    """

    def __init__(self, model_type="yolov5", model_weights=None, img_size=224, device=None, classes=None):
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.classes = classes or ["RBC"]

        if model_type == "classification":
            self.model = None
        elif model_type == "sparse_rcnn":
            try:
                from .model.sparse_rcnn import SparseRCNNModel
            except ImportError as exc:
                raise ImportError(
                    "SparseRCNN is unavailable in this environment. "
                    "Install a torchvision version that supports SparseRCNN, or use model_type='yolov5'."
                ) from exc

            self.model = SparseRCNNModel(num_classes=len(self.classes))
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = YOLOv5Wrapper(weights=model_weights, num_classes=len(self.classes), img_size=img_size)

    @classmethod
    def for_wbc_classifier(cls, weights_path, img_size=224, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights_path, map_location=device)
        class_names = ckpt.get("class_names")
        backbone = ckpt.get("backbone", "resnet18")
        model = WBCClassifier(num_classes=len(class_names), backbone=backbone, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        obj = cls.__new__(cls)
        obj.model_type = "classification"
        obj.device = device
        obj.img_size = img_size
        obj.classes = class_names
        obj.model = model
        obj.class_names = class_names
        obj.backbone = backbone
        return obj

    def evaluate_wbc_classifier(self, imagefolder_root, batch_size=16):
        if self.model_type != "classification":
            raise ValueError("evaluate_wbc_classifier requires Evaluator.for_wbc_classifier(...)")

        ds = datasets.ImageFolder(imagefolder_root, transform=build_wbc_transforms(self.img_size, train=False))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y_true = []
        y_pred = []

        self.model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(preds.cpu().tolist())

        cm = confusion_matrix_np(y_true, y_pred, num_classes=len(ds.classes))
        metrics = classification_metrics_from_cm(cm)
        metrics["class_names"] = ds.classes

        print(
            "Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            "Recall: {recall:.4f}, F1-score: {f1:.4f}".format(**metrics)
        )
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
        return metrics

    def evaluate_rbc_detector(self, dataset, batch_size=2, iou_threshold=0.5, score_threshold=0.25):
        if self.model_type == "classification":
            raise ValueError("evaluate_rbc_detector requires a detection evaluator.")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_ious = []

        for imgs, targets in dataloader:
            if self.model_type == "sparse_rcnn":
                imgs_device = [img.to(self.device) for img in imgs]
                with torch.no_grad():
                    predictions = self.model(imgs_device)
                pred_items = []
                for pred in predictions:
                    scores = pred.get("scores", torch.ones(len(pred["boxes"]))).detach().cpu().numpy()
                    keep = scores >= score_threshold
                    pred_items.append((pred["boxes"].detach().cpu().numpy()[keep], scores[keep]))
            else:
                imgs_np = [(img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8") for img in imgs]
                with torch.no_grad():
                    predictions = self.model.detect(imgs_np)
                pred_items = []
                for pred in predictions:
                    if pred is None or len(pred) == 0:
                        pred_items.append((np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)))
                        continue
                    scores = pred[:, 5] if pred.shape[1] > 5 else np.ones((len(pred),), dtype=np.float32)
                    keep = scores >= score_threshold
                    pred_items.append((pred[:, 1:5][keep], scores[keep]))

            for (pred_boxes, pred_scores), target in zip(pred_items, targets):
                gt_boxes = target["boxes"].cpu().numpy()
                tp, fp, fn, ious = match_detection_boxes(
                    pred_boxes,
                    gt_boxes,
                    pred_scores=pred_scores,
                    iou_threshold=iou_threshold,
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn
                matched_ious.extend(ious)

        precision = total_tp / max(1, total_tp + total_fp)
        recall = total_tp / max(1, total_tp + total_fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

        metrics = {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "iou_threshold": iou_threshold,
            "score_threshold": score_threshold,
        }

        print(
            "RBC detector | Precision: {precision:.4f}, Recall: {recall:.4f}, "
            "F1-score: {f1:.4f}, mean IoU: {mean_iou:.4f}".format(**metrics)
        )
        print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
        return metrics

    def evaluate_dataset(self, dataset, batch_size=2):
        """
        Backward-compatible alias cho detection evaluation.
        """
        return self.evaluate_rbc_detector(dataset, batch_size=batch_size)
