import cv2
import numpy as np
import torch
from PIL import Image
from .preprocessing import Preprocessing
from .wbc_proposal import WBCProposal
from .model.wbc_classifier import predict_class
from .model.yolov5 import YOLOv5Wrapper


def load_rbc_detector(model_type='sparse_rcnn', weights=None, classes=None, img_size=224, device=None):
    """
    Helper nap detector RBC cho notebook.
    """
    classes = classes or ['RBC']
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'sparse_rcnn':
        try:
            from .model.sparse_rcnn import SparseRCNNModel
        except ImportError as exc:
            raise ImportError(
                "SparseRCNN is unavailable in this environment. "
                "Use model_type='yolov5' or install a torchvision version that includes SparseRCNN."
            ) from exc
        model = SparseRCNNModel(num_classes=len(classes))
        if weights:
            state_dict = torch.load(weights, map_location=device)
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    if model_type == 'yolov5':
        return YOLOv5Wrapper(weights=weights, num_classes=len(classes), img_size=img_size)

    raise ValueError(f"Unsupported RBC detector type: {model_type}")


class CellPipeline:
    """
    Pipeline muc tieu cho notebook:
    - RBC detector
    - WBC proposal
    - WBC classifier
    """

    def __init__(
        self,
        rbc_detector=None,
        wbc_classifier=None,
        wbc_class_names=None,
        device=None,
        img_size=224,
    ):
        self.rbc_detector = rbc_detector
        self.wbc_classifier = wbc_classifier
        self.wbc_class_names = wbc_class_names or [
            "NEUTROPHIL",
            "LYMPHOCYTE",
            "MONOCYTE",
            "EOSINOPHIL",
            "BASOPHIL",
        ]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.preprocessor = Preprocessing(img_size=(img_size, img_size))
        self.wbc_proposal = WBCProposal()

    @staticmethod
    def _load_bgr(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return image

    def run(self, image_path):
        """
        Tra ve dict gom:
        - image_bgr
        - rbc_boxes
        - wbc_boxes
        - wbc_predictions
        """
        image_bgr = self._load_bgr(image_path)
        result = {
            "image_bgr": image_bgr,
            "rbc_boxes": [],
            "wbc_boxes": [],
            "wbc_predictions": [],
        }

        if self.rbc_detector is not None:
            rbc_preds = self._detect_rbc(image_bgr)
            result["rbc_boxes"] = rbc_preds
        else:
            rbc_preds = []

        wbc_boxes = self.wbc_proposal.propose(image_bgr, exclude_boxes=rbc_preds)
        result["wbc_boxes"] = wbc_boxes

        if self.wbc_classifier is not None and len(wbc_boxes) > 0:
            crops = self.wbc_proposal.crop_boxes(image_bgr, wbc_boxes)
            preds = []
            for crop in crops:
                preds.append(
                    predict_class(
                        self.wbc_classifier,
                        crop,
                        self.wbc_class_names,
                        device=self.device,
                        img_size=self.img_size,
                    )
                )
            result["wbc_predictions"] = preds

        return result

    def _detect_rbc(self, image_bgr):
        """
        Dung detector hien co de lay bbox RBC.
        Support cac wrapper detect hoac predict trong repo.
        """
        if self.rbc_detector is None:
            return []

        if isinstance(self.rbc_detector, torch.nn.Module):
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            tensor = self.preprocessor.to_tensor(self.preprocessor.resize(pil_img))
            orig_h, orig_w = image_bgr.shape[:2]
            scale_x = orig_w / float(self.img_size)
            scale_y = orig_h / float(self.img_size)
            self.rbc_detector.eval()
            with torch.no_grad():
                preds = self.rbc_detector([tensor.to(self.device)])[0]
            if preds is None:
                return []
            boxes = preds.get("boxes")
            if boxes is None:
                return []
            boxes = boxes.detach().cpu().numpy()
            scaled = []
            for x1, y1, x2, y2 in boxes:
                scaled.append([
                    float(x1 * scale_x),
                    float(y1 * scale_y),
                    float(x2 * scale_x),
                    float(y2 * scale_y),
                ])
            return scaled

        if hasattr(self.rbc_detector, "detect"):
            preds = self.rbc_detector.detect([image_bgr])[0]
            if preds is None or len(preds) == 0:
                return []
            boxes = []
            for row in preds:
                if len(row) >= 5:
                    boxes.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
            return boxes

        if hasattr(self.rbc_detector, "predict"):
            preds = self.rbc_detector.predict(image_bgr)
            if preds is None:
                return []
            if isinstance(preds, dict) and "boxes" in preds:
                return preds["boxes"]

        return []

    @staticmethod
    def draw_result(result):
        image = result["image_bgr"].copy()

        for x1, y1, x2, y2 in result["rbc_boxes"]:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, "RBC", (int(x1), max(0, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for i, box in enumerate(result["wbc_boxes"]):
            x1, y1, x2, y2 = box
            label = "WBC"
            if i < len(result["wbc_predictions"]):
                label = result["wbc_predictions"][i]["label"]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(image, label, (int(x1), max(0, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return image
