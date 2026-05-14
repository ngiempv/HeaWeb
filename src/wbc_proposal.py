import cv2
import numpy as np


class WBCProposal:
    """
    Tao box ung vien WBC bang xu ly anh so.
    Khong gia dinh WBC phai co mau tim; uu tien dua vao do tuong phan,
    cau truc va su khac biet so voi nen RBC. Mau chi la cue phu.
    Neu co box RBC tu detector, cac vung RBC se bi loai tru truoc khi lay contour.
    """

    def __init__(
        self,
        min_area=200,
        max_area=60000,
        max_candidates=30,
        hsv_ranges=None,
        lab_ranges=None,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.max_candidates = max_candidates
        self.hsv_ranges = hsv_ranges or [
            ((0, 0, 0), (180, 120, 255)),
            ((90, 20, 20), (170, 255, 255)),
        ]
        self.lab_ranges = lab_ranges or [
            ((0, 120, 120), (255, 180, 220)),
        ]

    def _mask_from_ranges(self, img, ranges, color_space):
        if color_space == "hsv":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == "lab":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        else:
            raise ValueError(f"Unsupported color_space: {color_space}")

        mask = np.zeros(converted.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(converted, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        return mask

    def propose(self, image_bgr, exclude_boxes=None):
        """
        Tra ve list box [x1, y1, x2, y2] ung vien WBC.
        """
        if image_bgr is None:
            return []

        exclude_mask = self._build_exclude_mask(image_bgr.shape[:2], exclude_boxes)
        enhanced = self._enhance_contrast(image_bgr)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        contrast_mask = self._contrast_mask(enhanced)
        texture_mask = self._texture_mask(gray_blur)
        _, otsu_mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive_mask = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )
        edges = cv2.Canny(gray_blur, 40, 120)
        color_mask = self._weak_color_contrast_mask(enhanced)

        kernel = np.ones((5, 5), np.uint8)
        boxes = []
        candidate_masks = [
            contrast_mask,
            texture_mask,
            otsu_mask,
            adaptive_mask,
            edges,
            color_mask,
        ]

        for mask in candidate_masks:
            clean = self._apply_exclude_mask(mask, exclude_mask)
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
            clean = cv2.dilate(clean, kernel, iterations=1)
            boxes.extend(self._boxes_from_mask(clean))

        if not boxes:
            fallback = cv2.bitwise_or(contrast_mask, adaptive_mask)
            fallback = cv2.bitwise_or(fallback, otsu_mask)
            fallback = cv2.bitwise_or(fallback, texture_mask)
            fallback = cv2.morphologyEx(fallback, cv2.MORPH_OPEN, kernel)
            fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, kernel)
            fallback = self._apply_exclude_mask(fallback, exclude_mask)
            boxes.extend(self._boxes_from_mask(fallback))

        if exclude_boxes:
            boxes = [box for box in boxes if not self._overlaps_excluded(box, exclude_boxes)]

        boxes = self._deduplicate_boxes(boxes)
        boxes.sort(key=lambda b: (-(b[2] - b[0]) * (b[3] - b[1]), b[1], b[0]))
        return boxes[: self.max_candidates]

    def _boxes_from_mask(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 8:
                continue
            boxes.append([x, y, x + w, y + h])
        return boxes

    def _deduplicate_boxes(self, boxes, iou_threshold=0.45):
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept = []
        for box in boxes:
            if all(self._box_iou(box, kept_box) < iou_threshold for kept_box in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _box_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _enhance_contrast(image_bgr):
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _contrast_mask(image_bgr):
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_blur = cv2.GaussianBlur(l, (5, 5), 0)

        blackhat = cv2.morphologyEx(l_blur, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
        tophat = cv2.morphologyEx(l_blur, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))

        _, l_otsu = cv2.threshold(l_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, a_otsu = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, b_otsu = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask = cv2.bitwise_or(l_otsu, cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        mask = cv2.bitwise_or(mask, cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        mask = cv2.bitwise_or(mask, a_otsu)
        mask = cv2.bitwise_or(mask, b_otsu)
        return mask

    @staticmethod
    def _texture_mask(gray_blur):
        lap = cv2.Laplacian(gray_blur, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        _, mask = cv2.threshold(lap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    @staticmethod
    def _weak_color_contrast_mask(image_bgr, saturation_thresh=18):
        b, g, r = cv2.split(image_bgr)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]

        chroma = np.maximum.reduce([b, g, r]).astype(np.int16) - np.minimum.reduce([b, g, r]).astype(np.int16)
        mask = np.zeros(b.shape, dtype=np.uint8)
        cond = (chroma > 20) & (s > saturation_thresh)
        mask[cond] = 255
        mask = cv2.medianBlur(mask, 5)
        return mask

    @staticmethod
    def _build_exclude_mask(shape_hw, exclude_boxes=None, padding=4):
        mask = np.zeros(shape_hw, dtype=np.uint8)
        if not exclude_boxes:
            return mask

        h, w = shape_hw
        for box in exclude_boxes:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = box[:4]
            x1 = max(0, min(w, int(x1) - padding))
            y1 = max(0, min(h, int(y1) - padding))
            x2 = max(0, min(w, int(x2) + padding))
            y2 = max(0, min(h, int(y2) + padding))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def _apply_exclude_mask(mask, exclude_mask):
        if exclude_mask is None or not np.any(exclude_mask):
            return mask
        filtered = mask.copy()
        filtered[exclude_mask > 0] = 0
        return filtered

    @staticmethod
    def _overlaps_excluded(box, exclude_boxes, iou_threshold=0.25):
        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area == 0:
            return True
        for ex in exclude_boxes:
            ex_x1, ex_y1, ex_x2, ex_y2 = ex
            ix1 = max(x1, ex_x1)
            iy1 = max(y1, ex_y1)
            ix2 = min(x2, ex_x2)
            iy2 = min(y2, ex_y2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter == 0:
                continue
            ex_area = max(0, ex_x2 - ex_x1) * max(0, ex_y2 - ex_y1)
            union = area + ex_area - inter
            if union > 0 and inter / union >= iou_threshold:
                return True
        return False

    @staticmethod
    def crop_boxes(image_bgr, boxes):
        crops = []
        h, w = image_bgr.shape[:2]
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, min(w, int(x1)))
            x2 = max(0, min(w, int(x2)))
            y1 = max(0, min(h, int(y1)))
            y2 = max(0, min(h, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image_bgr[y1:y2, x1:x2].copy())
        return crops
