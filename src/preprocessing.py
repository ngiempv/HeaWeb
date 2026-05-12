import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import random
'''dùng với sigle cell, mỗi ảnh 1 tế bào'''
class Preprocessing:
    def __init__(self, img_size=(224, 224), normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)):
        """
        param img_size: Kích thước resize ảnh (width, height)
        param normalize_mean: Mean để chuẩn hóa
        param normalize_std: Std để chuẩn hóa
        """
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)

    def load_image(self, path):
        """Load image và convert sang RGB"""
        img = Image.open(path).convert("RGB")
        return img

    def resize(self, img):
        """Resize ảnh về img_size"""
        return img.resize(self.img_size)

    def to_tensor(self, img):
        """Chuyển PIL Image sang Tensor"""
        img = transforms.ToTensor()(img)
        img = self.normalize(img)
        return img

    def augment(self, img):
        """Một số augmentation cơ bản"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
        # Random vertical flip
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)
        # Random rotation
        angle = random.choice([0, 90, 180, 270])
        img = transforms.functional.rotate(img, angle)
        return img

    def filter_blur(self, img, threshold=100):
        """Lọc ảnh nhòe bằng Laplacian variance"""
        img_cv = np.array(img.convert("L"))
        variance = cv2.Laplacian(img_cv, cv2.CV_64F).var()
        return variance > threshold

    def preprocess_classification(self, path, augment=True):
        """
        Tiền xử lý ảnh cho classification
        :param path: đường dẫn ảnh
        :param augment: có augment hay không
        """
        img = self.load_image(path)
        if augment:
            img = self.augment(img)
        if not self.filter_blur(img):
            return None
        img = self.resize(img)
        img = self.to_tensor(img)
        return img

    def preprocess_detection(self, path, boxes=None, labels=None):
        """
        Tiền xử lý ảnh cho detection
        :param path: đường dẫn ảnh
        :param boxes: bounding boxes (n,4)
        :param labels: labels (n,)
        """
        img = self.load_image(path)
        if not self.filter_blur(img):
            return None, None, None
        orig_size = img.size
        img = self.resize(img)
        img_tensor = self.to_tensor(img)

        # Scale bounding boxes
        if boxes is not None:
            scale_x = self.img_size[0] / orig_size[0]
            scale_y = self.img_size[1] / orig_size[1]
            boxes = boxes.copy()
            boxes[:, 0] = boxes[:, 0] * scale_x
            boxes[:, 1] = boxes[:, 1] * scale_y
            boxes[:, 2] = boxes[:, 2] * scale_x
            boxes[:, 3] = boxes[:, 3] * scale_y

        return img_tensor, boxes, labels