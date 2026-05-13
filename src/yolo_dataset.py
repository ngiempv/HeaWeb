import os
import shutil
from pathlib import Path

import cv2
import pandas as pd


def prepare_yolo_detection_dataset(
    annotations_csv,
    image_dir,
    output_dir,
    class_names=None,
    val_ratio=0.2,
    seed=42,
):
    """
    Tao dataset YOLO tu CSV detection cua RBC.

    CSV can co cac cot:
    image, class_id, xmin, ymin, xmax, ymax
    """
    class_names = class_names or ["RBC"]
    annotations_csv = Path(annotations_csv)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    df = pd.read_csv(annotations_csv)
    required_cols = {"image", "class_id", "xmin", "ymin", "xmax", "ymax"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in annotation CSV: {sorted(missing)}")

    images = sorted(df["image"].dropna().unique().tolist())
    if not images:
        raise ValueError("Annotation CSV does not contain any images.")

    shuffled = pd.Series(images).sample(frac=1.0, random_state=seed).tolist()
    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_images = set(shuffled[:val_count])

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_name in images:
        split = "val" if image_name in val_images else "train"
        src_img = image_dir / image_name
        if not src_img.exists():
            continue

        img = cv2.imread(str(src_img))
        if img is None:
            continue

        h, w = img.shape[:2]
        dst_img = output_dir / "images" / split / image_name
        shutil.copy2(src_img, dst_img)

        rows = df[df["image"] == image_name]
        label_lines = []
        for _, row in rows.iterrows():
            x1 = float(row["xmin"])
            y1 = float(row["ymin"])
            x2 = float(row["xmax"])
            y2 = float(row["ymax"])
            cls_id = int(row["class_id"])

            x_center = ((x1 + x2) / 2.0) / w
            y_center = ((y1 + y2) / 2.0) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            label_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        label_path = output_dir / "labels" / split / f"{src_img.stem}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

    yaml_path = output_dir / "rbc.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return str(yaml_path)
