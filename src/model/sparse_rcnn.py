import torch
import torch.nn as nn
from torchvision.models.detection import SparseRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.sparse_rcnn import SparseRCNNHead, SparseRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss

class SparseRCNNModel(nn.Module):
    """
    Sparse R-CNN model cho detection + classification tế bào
    """
    def __init__(self, num_classes, backbone_pretrained=True, backbone_weights=None, num_queries=100):
        """
        :param num_classes: số class bao gồm các tế bào bình thường + abnormal
        :param backbone_pretrained: dùng pretrained ImageNet hay không
        :param backbone_weights: đường dẫn checkpoint pretrained CNN (optional)
        :param num_queries: số proposals cố định
        """
        super(SparseRCNNModel, self).__init__()

        # 1️⃣ Backbone (ResNet50 + FPN)
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=backbone_pretrained)
        if backbone_weights is not None:
            # load pretrained CNN từ dataset2-master
            print(f"[INFO] Loading backbone weights from {backbone_weights}")
            state_dict = torch.load(backbone_weights, map_location='cpu')
            self.backbone.body.load_state_dict(state_dict, strict=False)

        # 2️⃣ Sparse R-CNN Head
        # output_size = 256 là dimension của hidden feature
        self.model = SparseRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            num_proposals=num_queries,
        )

        # 3️⃣ Custom head (nếu muốn fine-tune)
        in_channels = self.model.roi_heads.box_head.fc7.out_features
        self.model.roi_heads.box_predictor = SparseRCNNPredictor(in_channels, num_classes)

    def forward(self, images, targets=None):
        """
        :param images: list of tensors [C,H,W], 0-1 normalized
        :param targets: list of dicts [{'boxes': tensor[N,4], 'labels': tensor[N]}], optional for training
        :return: predictions hoặc loss dictionary
        """
        if self.training:
            assert targets is not None, "Targets required for training"
            loss_dict = self.model(images, targets)
            return loss_dict
        else:
            # inference
            predictions = self.model(images)
            return predictions