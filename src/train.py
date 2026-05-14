import os
import argparse
import time
import platform
import sys
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .dataloader import CellDataset
from .model.yolov5 import YOLOv5Wrapper
from .model.wbc_classifier import WBCClassifier, build_wbc_transforms
from .yolo_dataset import prepare_yolo_detection_dataset
from torchvision import datasets
from torch import nn

def _format_duration(seconds):
    seconds = int(round(seconds))
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m{secs:02d}s"
    if mins > 0:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


def _progress_line(prefix, current, total, elapsed, loss=None, width=26):
    total = max(1, total)
    current = min(current, total)
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled + "-" * (width - filled)
    elapsed_txt = _format_duration(elapsed)
    if current >= total and elapsed > 0:
        rate = elapsed / total
        remaining = 0.0
    else:
        rate = elapsed / current if current > 0 else 0.0
        remaining = max(0.0, rate * (total - current))
    remaining_txt = _format_duration(remaining)
    extra = f", loss={loss:.3f}" if loss is not None else ""
    return f"{prefix}: {current:>3}/{total} |{bar}| {pct*100:5.1f}% [{elapsed_txt}<{remaining_txt}{extra}]"

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
    if not args.ann_file and not args.yolo_data:
        raise ValueError('--ann_file or --yolo_data is required for YOLO detection training')

    if args.yolo_data:
        yolo_data = args.yolo_data
    else:
        yolo_data = prepare_yolo_detection_dataset(
            annotations_csv=args.ann_file,
            image_dir=args.img_dir,
            output_dir=args.yolo_output_dir,
            class_names=classes,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(f"[INFO] YOLO dataset prepared at {yolo_data}")

    yolo = YOLOv5Wrapper(weights=args.weights, num_classes=len(classes), img_size=args.img_size)
    yolo.train(
        data_yaml=yolo_data,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch_size,
        project=args.save_dir,
        name='rbc_yolo',
    )
    fixed_weight_dir = Path(args.save_dir) / "rbc_yolo" / "weights"
    fixed_weight_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = sorted(
        Path("runs").glob("detect/**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidate_paths:
        src_best = candidate_paths[0]
        dst_best = fixed_weight_dir / "best.pt"
        shutil.copy2(src_best, dst_best)
        last_src = src_best.with_name("last.pt")
        if last_src.exists():
            shutil.copy2(last_src, fixed_weight_dir / "last.pt")
        print(f"[INFO] YOLO training finished. Best weights copied to {dst_best}")
    else:
        print(f"[WARN] YOLO training finished but no best.pt found under runs/detect")

def train_wbc_classifier(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_root = args.train_root
    val_root = args.val_root or args.train_root
    num_workers = 0 if platform.system().lower().startswith("win") else 4

    print(
        f"[INFO] Start WBC training | device={device} | "
        f"train_root={train_root} | val_root={val_root}",
        flush=True,
    )

    print("[INFO] Loading datasets...", flush=True)
    train_ds = datasets.ImageFolder(train_root, transform=build_wbc_transforms(args.img_size, train=True))
    val_ds = datasets.ImageFolder(val_root, transform=build_wbc_transforms(args.img_size, train=False))
    print(f"[INFO] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}", flush=True)
    print(f"[INFO] Classes: {train_ds.classes}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    print("[INFO] Building model...", flush=True)
    class_names = train_ds.classes
    model = WBCClassifier(num_classes=len(class_names), backbone=args.backbone, pretrained=args.pretrained)
    model.to(device)
    print("[INFO] Model ready. Starting training loop...", flush=True)
    class_text = ", ".join(class_names)
    print(f"[INFO] Data  : train={train_root} | val={val_root}", flush=True)
    print(f"[INFO] Class : {class_text}", flush=True)
    print(f"[INFO] Model : WBC classifier ({args.backbone})", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}", flush=True)
        print(
            f"WBC classifier | train={train_root} | val={val_root} | classes={class_names}",
            flush=True,
        )
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            line = _progress_line("Train", batch_idx, total_batches, time.time() - epoch_start, loss.item())
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        total_val_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                line = _progress_line("Val  ", batch_idx, total_val_batches, time.time() - epoch_start)
                sys.stdout.write("\r" + line)
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

        acc = correct / max(1, total)
        epoch_time = time.time() - epoch_start
        print(
            f"epoch_time={_format_duration(epoch_time)} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'backbone': args.backbone,
                'img_size': args.img_size,
            }
            path = os.path.join(args.save_dir, 'wbc_classifier_best.pth')
            torch.save(ckpt, path)
            print(f"[INFO] Best checkpoint -> {path}", flush=True)

    total_time = time.time() - total_start
    print(f"\n[INFO] Training finished in {_format_duration(total_time)}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train blood cell models")
    parser.add_argument('--task', type=str, default='classification', choices=['detection', 'classification'], help='Task type')
    parser.add_argument('--model', type=str, default='sparse_rcnn', choices=['sparse_rcnn','yolov5'], help='Detection model type')
    parser.add_argument('--img_dir', type=str, default=None, help='Image folder for detection')
    parser.add_argument('--ann_file', type=str, default=None, help='Annotations CSV for detection')
    parser.add_argument('--yolo_data', type=str, default=None, help='Existing YOLO data yaml for detection')
    parser.add_argument('--yolo_output_dir', type=str, default='dataset-master/rbc_yolo_dataset', help='Output folder for generated YOLO dataset')
    parser.add_argument('--train_root', type=str, default=None, help='Training root for classification')
    parser.add_argument('--val_root', type=str, default=None, help='Validation root for classification')
    parser.add_argument('--backbone_weights', type=str, default=None, help='Pretrained backbone for SparseRCNN')
    parser.add_argument('--weights', type=str, default=None, help='Pretrained weights for YOLOv5')
    parser.add_argument('--classes', type=str, default='RBC', help='Comma separated classes for detection')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone for WBC classifier')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone for WBC classifier when available')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio for generated YOLO dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset split')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.task == 'classification':
        if not args.train_root:
            raise ValueError('--train_root is required when --task classification')
        train_wbc_classifier(args)
    elif args.model == 'sparse_rcnn':
        if not args.img_dir:
            raise ValueError('--img_dir is required for detection training')
        train_sparse_rcnn(args)
    else:
        if not args.img_dir:
            raise ValueError('--img_dir is required for detection training')
        train_yolov5(args)
