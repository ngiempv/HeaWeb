import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from .preprocessing import ClassificationPreprocessing, DetectionPreprocessing

class CellDataset(Dataset):
    """
    Dataset cho cả Classification (dataset2-master) và Detection (dataset-master)
    """
    def __init__(self, dataset_type, img_dir, annotations_file=None, classes=None,
                 img_size=(224,224), augment=False):
        """
        :param dataset_type: 'classification' hoặc 'detection'
        :param img_dir: folder chứa ảnh
        :param annotations_file: CSV hoặc None nếu classification
        :param classes: list class
        :param img_size: resize size
        :param augment: True nếu muốn augment ảnh classification
        """
        assert dataset_type in ['classification','detection']
        self.dataset_type = dataset_type
        self.img_dir = img_dir
        self.augment = augment
        self.classes = classes
        if dataset_type == 'classification':
            self.preprocessor = ClassificationPreprocessing(img_size=img_size)
        else:
            self.preprocessor = DetectionPreprocessing(img_size=img_size)
        
        if dataset_type == 'classification':
            # Tạo list (img_path, label)
            self.samples = []
            for cls in os.listdir(img_dir):
                cls_path = os.path.join(img_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith(('png','jpg','jpeg')):
                        self.samples.append((os.path.join(cls_path,fname), classes.index(cls)))
        else:
            # Detection dataset
            self.df = pd.read_csv(annotations_file)
            self.samples = self.df['image'].unique().tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.dataset_type == 'classification':
            img_path, label = self.samples[idx]
            img_tensor = self.preprocessor.preprocess(img_path, augment=self.augment)
            if img_tensor is None:
                # Nếu ảnh bị blur hoặc lỗi, random chọn ảnh khác
                return self.__getitem__((idx+1)%len(self))
            return img_tensor, label
        else:
            # Detection
            img_name = self.samples[idx]
            img_path = os.path.join(self.img_dir, img_name)
            df_img = self.df[self.df['image']==img_name]
            boxes = df_img[['xmin','ymin','xmax','ymax']].to_numpy()
            labels = df_img['class_id'].to_numpy()

            attempts = 0
            img_tensor, boxes_scaled, labels_out = self.preprocessor.preprocess(img_path, boxes, labels)
            while img_tensor is None and attempts < len(self):
                attempts += 1
                idx = (idx + 1) % len(self)
                img_name = self.samples[idx]
                img_path = os.path.join(self.img_dir, img_name)
                df_img = self.df[self.df['image']==img_name]
                boxes = df_img[['xmin','ymin','xmax','ymax']].to_numpy()
                labels = df_img['class_id'].to_numpy()
                img_tensor, boxes_scaled, labels_out = self.preprocessor.preprocess(img_path, boxes, labels)

            if img_tensor is None:
                raise RuntimeError('Detection dataset contains no valid images after preprocessing.')

            target = {
                'boxes': torch.tensor(boxes_scaled, dtype=torch.float32),
                'labels': torch.tensor(labels_out, dtype=torch.int64)
            }
            return img_tensor, target
