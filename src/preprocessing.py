import cv2
import numpy as np
import random
from PIL import Image
from torchvision import transforms


class Preprocessing:
    """
    Preprocessing chung, giu lai cac ham cu de tuong thich ngoc.
    """

    def __init__(self, img_size=(224, 224), normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)):
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)

    def load_image(self, path):
        return Image.open(path).convert("RGB")

    def resize(self, img):
        return img.resize(self.img_size)

    def to_tensor(self, img):
        img = transforms.ToTensor()(img)
        return self.normalize(img)

    def augment(self, img):
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)
        angle = random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(img, angle)

    def filter_blur(self, img, threshold=100):
        img_cv = np.array(img.convert("L"))
        variance = cv2.Laplacian(img_cv, cv2.CV_64F).var()
        return variance > threshold

    def preprocess_classification(self, path, augment=True):
        img = self.load_image(path)
        if augment:
            img = self.augment(img)
        if not self.filter_blur(img):
            return None
        img = self.resize(img)
        return self.to_tensor(img)

    def preprocess_detection(self, path, boxes=None, labels=None):
        img = self.load_image(path)
        orig_size = img.size
        img = self.resize(img)
        img_tensor = self.to_tensor(img)

        if boxes is not None:
            scale_x = self.img_size[0] / orig_size[0]
            scale_y = self.img_size[1] / orig_size[1]
            boxes = boxes.copy()
            boxes[:, 0] = boxes[:, 0] * scale_x
            boxes[:, 1] = boxes[:, 1] * scale_y
            boxes[:, 2] = boxes[:, 2] * scale_x
            boxes[:, 3] = boxes[:, 3] * scale_y

        return img_tensor, boxes, labels


class ClassificationPreprocessing(Preprocessing):
    def preprocess(self, path, augment=True):
        return self.preprocess_classification(path, augment=augment)


class DetectionPreprocessing(Preprocessing):
    def preprocess(self, path, boxes=None, labels=None):
        return self.preprocess_detection(path, boxes=boxes, labels=labels)
