import os
import argparse
import torch
from torch.utils.data import DataLoader
from .dataloader import CellDataset
from .model.yolov5 import YOLOv5Wrapper

def train_sparse_rcnn(args):
    try:
        from .model.sparse_rcnn import SparseRCNNModel
    except ImportError as e:
        raise ImportError(
            "SparseRCNN is unavailable in this environment. "
            "Install a torchvision version that supports SparseRCNN, or use --model yolov5."
        ) from e
    # Dataset
    classes = args.classes.split(',')
    dataset = CellDataset(
        dataset_type='detection',
        img_dir=args.img_dir,
        annotations_file=args.ann_file,
        classes=classes,
        img_size=(args.img_size,args.img_size)
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SparseRCNNModel(num_classes=len(classes), backbone_pretrained=True, backbone_weights=args.backbone_weights)
    model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"sparse_rcnn_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[INFO] Checkpoint saved to {checkpoint_path}")

def train_yolov5(args):
    classes = args.classes.split(',')
    dataset = CellDataset(
        dataset_type='detection',
        img_dir=args.img_dir,
        annotations_file=args.ann_file,
        classes=classes,
        img_size=(args.img_size,args.img_size)
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    yolo = YOLOv5Wrapper(weights=args.weights, num_classes=len(classes), img_size=args.img_size)
    yolo.train(dataloader, epochs=args.epochs)
    # Save final model
    yolo.save(os.path.join(args.save_dir, "yolov5_final.pt"))
    print(f"[INFO] YOLOv5 model saved to {os.path.join(args.save_dir, 'yolov5_final.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SparseRCNN or YOLOv5 on Blood Cell Dataset")
    parser.add_argument('--model', type=str, default='sparse_rcnn', choices=['sparse_rcnn','yolov5'], help='Model type')
    parser.add_argument('--img_dir', type=str, required=True, help='Image folder')
    parser.add_argument('--ann_file', type=str, default=None, help='Annotations CSV for detection')
    parser.add_argument('--backbone_weights', type=str, default=None, help='Pretrained backbone for SparseRCNN')
    parser.add_argument('--weights', type=str, default=None, help='Pretrained weights for YOLOv5')
    parser.add_argument('--classes', type=str, default='NEUTROPHIL,LYMPHOCYTE,MONOCYTE,EOSINOPHIL,ABNORMAL', help='Comma separated classes')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.model == 'sparse_rcnn':
        train_sparse_rcnn(args)
    else:
        train_yolov5(args)