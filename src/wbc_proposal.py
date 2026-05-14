import cv2
import numpy as np


class WBCProposal:
    """
    Tao box ung vien WBC bang xu ly anh so.
    Khong coi mau tim la dau hieu duy nhat; uu tien do tuong phan va cau truc.
    Neu co box RBC, se loai tru cac vung RBC truoc khi tim proposal WBC.
    """

    def __init__(
        self,
        min_area=600,
        max_area=120000,
        max_candidates=15,
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

    def propose(self, image_bgr, exclude_boxes=None):
        """
        Tra ve list box [x1, y1, x2, y2] ung vien WBC.
        """
        if image_bgr is None:
            return []

        h, w = image_bgr.shape[:2]
        dynamic_min_area = max(self.min_area, int(h * w * 0.0015))
        dynamic_max_area = max(self.max_area, int(h * w * 0.25))

        exclude_mask = self._build_exclude_mask((h, w), exclude_boxes)
        enhanced = self._enhance_contrast(image_bgr)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        purple_mask = self._purple_mask(enhanced)
        contrast_mask = self._contrast_mask(enhanced)
        texture_mask = self._texture_mask(gray_blur)
        color_mask = self._weak_color_contrast_mask(enhanced)
        edges = cv2.Canny(gray_blur, 40, 120)

        candidate_masks = [
            purple_mask,
            contrast_mask,
            texture_mask,
            color_mask,
            edges,
        ]

        _, otsu_mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive_mask = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )

        boxes = []
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        kernel_dilate = np.ones((3, 3), np.uint8)

        for mask in candidate_masks:
            clean = self._apply_exclude_mask(mask, exclude_mask)
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_open)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close)
            clean = cv2.dilate(clean, kernel_dilate, iterations=1)
            boxes.extend(self._boxes_from_mask(clean, dynamic_min_area, dynamic_max_area))

        if not boxes:
            fallback = cv2.bitwise_or(otsu_mask, adaptive_mask)
            fallback = cv2.bitwise_or(fallback, texture_mask)
            fallback = self._apply_exclude_mask(fallback, exclude_mask)
            fallback = cv2.morphologyEx(fallback, cv2.MORPH_OPEN, kernel_open)
            fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, kernel_close)
            fallback = cv2.dilate(fallback, kernel_dilate, iterations=1)
            boxes = self._boxes_from_mask(fallback, dynamic_min_area, dynamic_max_area)

        if exclude_boxes:
            boxes = [box for box in boxes if not self._overlaps_excluded(box, exclude_boxes, iou_threshold=0.18)]

        boxes = self._expand_boxes(boxes, image_bgr.shape[:2])
        boxes = self._rank_boxes_by_purple_score(boxes, enhanced, top_k=self.max_candidates * 3)
        boxes = self._keep_strong_purple_boxes(boxes, enhanced, min_ratio=0.55)
        boxes = self._merge_overlapping_boxes(boxes, iou_threshold=0.22)
        boxes = self._remove_nested_small_boxes(boxes)
        boxes.sort(key=lambda b: (-(b[2] - b[0]) * (b[3] - b[1]), b[1], b[0]))
        return boxes[: self.max_candidates]

    def _boxes_from_mask(self, mask, min_area=None, max_area=None):
        min_area = self.min_area if min_area is None else min_area
        max_area = self.max_area if max_area is None else max_area
        img_h, img_w = mask.shape[:2]
        img_area = img_h * img_w
        max_box_area = img_area * 0.18
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            if w < 12 or h < 12:
                continue
            box_area = w * h
            if box_area > max_box_area:
                continue
            if w > img_w * 0.65 or h > img_h * 0.75:
                continue
            fill_ratio = area / max(1, box_area)
            if fill_ratio < 0.06:
                continue
            aspect = max(w / max(1, h), h / max(1, w))
            if aspect > 5.0:
                continue
            boxes.append([x, y, x + w, y + h])
        return boxes

    def _merge_overlapping_boxes(self, boxes, iou_threshold=0.22):
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        merged = []
        for box in boxes:
            placed = False
            for idx, kept in enumerate(merged):
                if self._box_iou(box, kept) >= iou_threshold:
                    merged[idx] = [
                        min(box[0], kept[0]),
                        min(box[1], kept[1]),
                        max(box[2], kept[2]),
                        max(box[3], kept[3]),
                    ]
                    placed = True
                    break
            if not placed:
                merged.append(list(box))

        changed = True
        while changed:
            changed = False
            next_merged = []
            for box in merged:
                absorbed = False
                for idx, kept in enumerate(next_merged):
                    if self._box_iou(box, kept) >= iou_threshold:
                        next_merged[idx] = [
                            min(box[0], kept[0]),
                            min(box[1], kept[1]),
                            max(box[2], kept[2]),
                            max(box[3], kept[3]),
                        ]
                        absorbed = True
                        changed = True
                        break
                if not absorbed:
                    next_merged.append(box)
            merged = next_merged
        return merged

    def _remove_nested_small_boxes(self, boxes):
        if len(boxes) <= 1:
            return boxes

        filtered = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = max(0, x2 - x1) * max(0, y2 - y1)
            keep = True
            for j, other in enumerate(boxes):
                if i == j:
                    continue
                ox1, oy1, ox2, oy2 = other
                other_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
                if other_area <= area:
                    continue
                ix1 = max(x1, ox1)
                iy1 = max(y1, oy1)
                ix2 = min(x2, ox2)
                iy2 = min(y2, oy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if area > 0 and inter / area >= 0.80:
                    keep = False
                    break
            if keep:
                filtered.append(box)
        return filtered

    @staticmethod
    def _rank_boxes_by_purple_score(boxes, image_bgr, top_k=None):
        if not boxes:
            return []

        scored = []
        h, w = image_bgr.shape[:2]
        b, g, r = cv2.split(image_bgr)
        purple_score = (r.astype(np.int16) + b.astype(np.int16) - 2 * g.astype(np.int16))
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, min(w, int(x1)))
            y1 = max(0, min(h, int(y1)))
            x2 = max(0, min(w, int(x2)))
            y2 = max(0, min(h, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            patch = purple_score[y1:y2, x1:x2]
            mean_score = float(np.mean(patch)) if patch.size else -999.0
            scored.append((mean_score, [x1, y1, x2, y2]))

        scored.sort(key=lambda item: item[0], reverse=True)
        ranked = [box for _, box in scored]
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    @staticmethod
    def _keep_strong_purple_boxes(boxes, image_bgr, min_ratio=0.55):
        if not boxes:
            return []

        h, w = image_bgr.shape[:2]
        b, g, r = cv2.split(image_bgr)
        purple_score = (r.astype(np.int16) + b.astype(np.int16) - 2 * g.astype(np.int16))

        scored = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, min(w, int(x1)))
            y1 = max(0, min(h, int(y1)))
            x2 = max(0, min(w, int(x2)))
            y2 = max(0, min(h, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            patch = purple_score[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            mean_score = float(np.mean(patch))
            scored.append((mean_score, [x1, y1, x2, y2]))

        if not scored:
            return []

        max_score = max(score for score, _ in scored)
        threshold = max_score * min_ratio
        kept = [box for score, box in scored if score >= threshold]
        if not kept:
            kept = [box for _, box in scored[: max(1, min(3, len(scored)))]]
        return kept

    @staticmethod
    def _purple_mask(image_bgr):
        b, g, r = cv2.split(image_bgr)
        purple_score = (r.astype(np.int16) + b.astype(np.int16) - 2 * g.astype(np.int16))
        purple_score = np.clip(purple_score + 128, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(purple_score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.medianBlur(mask, 5)
        return mask

    @staticmethod
    def _expand_boxes(boxes, shape_hw, pad_ratio=0.35, min_pad=16, max_pad=56):
        h, w = shape_hw
        expanded = []
        for x1, y1, x2, y2 in boxes:
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            pad = int(min(max_pad, max(min_pad, max(bw, bh) * pad_ratio)))
            expanded.append([
                max(0, int(x1) - pad),
                max(0, int(y1) - pad),
                min(w, int(x2) + pad),
                min(h, int(y2) + pad),
            ])
        return expanded

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
