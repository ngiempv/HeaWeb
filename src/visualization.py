import cv2


def draw_boxes(image_bgr, boxes, label="BOX", color=(0, 255, 0), thickness=2):
    image = image_bgr.copy()
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(
            image,
            label,
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return image


def draw_pipeline_result(result):
    image = result["image_bgr"].copy()

    for box in result.get("rbc_boxes", []):
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            image,
            "RBC",
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    wbc_predictions = result.get("wbc_predictions", [])
    for idx, box in enumerate(result.get("wbc_boxes", [])):
        x1, y1, x2, y2 = box[:4]
        label = "WBC"
        if idx < len(wbc_predictions):
            label = wbc_predictions[idx].get("label", "WBC")
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(
            image,
            label,
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    return image
