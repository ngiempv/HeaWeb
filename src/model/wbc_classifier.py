import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class WBCClassifier(nn.Module):
    """
    Classifier WBC dua tren transfer learning.
    Phu hop de test trong notebook va sau do boc web.
    """

    def __init__(self, num_classes, backbone="resnet18", pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.model(x)


def build_wbc_transforms(img_size=224, train=False):
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            *base,
        ])
    return transforms.Compose(base)


def predict_class(model, image_bgr, class_names, device="cpu", img_size=224):
    """
    Predict 1 crop WBC tu anh BGR.
    Tra ve dict: label, score, probs.
    """
    model.eval()
    img_rgb = Image.fromarray(image_bgr[:, :, ::-1].copy())
    tfm = build_wbc_transforms(img_size=img_size, train=False)
    tensor = tfm(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        score, pred_idx = torch.max(probs, dim=0)
    return {
        "label": class_names[int(pred_idx)],
        "score": float(score),
        "probs": probs.cpu().tolist(),
        "class_index": int(pred_idx),
    }


def load_wbc_classifier(weights_path, device="cpu"):
    """
    Load checkpoint classifier da train.
    Tra ve: model, class_names, backbone, img_size.
    """
    ckpt = torch.load(weights_path, map_location=device)
    class_names = ckpt.get("class_names")
    backbone = ckpt.get("backbone", "resnet18")
    img_size = ckpt.get("img_size", 224)
    model = WBCClassifier(num_classes=len(class_names), backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, backbone, img_size
