import cv2


def rotate_frame(frame, angle: int):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def draw_fancy_box(img, box, label, conf):
    """Draw a visible bounding box for the current object."""
    x1, y1, x2, y2 = map(int, box)
    h, w, _ = img.shape
    y1 = max(0, y1)
    x1 = max(0, x1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    color = (0, 255, 127)
    thickness = 3

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 50, 0), thickness + 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    label_text = f"{label} {conf:.2f}"
    font_scale = max(1.2, min(3, w / 500))
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
    text_x = x1
    text_y = max(text_h + 15, y1 - 10)

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (text_x - 5, text_y - text_h - 10),
        (text_x + text_w + 10, text_y + 5),
        color,
        -1,
    )
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(
        img,
        label_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        3,
    )
    return img


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0
