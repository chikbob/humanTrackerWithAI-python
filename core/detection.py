import time

import cv2
import streamlit as st
from ultralytics import YOLO
from typing import Optional


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return YOLO(model_path)


def build_class_meta(model_names: dict, animal_classes: dict):
    all_class_names = list(model_names.values())
    class_meta = {name: {"is_animal": False, "animal_group": None} for name in all_class_names}
    for group_name, names in animal_classes.items():
        for name in names:
            if name in class_meta:
                class_meta[name]["is_animal"] = True
                class_meta[name]["animal_group"] = group_name
    return all_class_names, class_meta


def get_class_meta(class_meta: dict, cls_name: str):
    meta = class_meta.get(cls_name, {})
    return meta.get("is_animal", False), meta.get("animal_group")


def class_allowed(cls_name: str, animal_filter: str, animal_classes: dict, track_classes: list[str]) -> bool:
    if animal_filter != "всё":
        allowed_animals = animal_classes.get(animal_filter, [])
        if cls_name not in allowed_animals:
            return False
    if track_classes and cls_name not in track_classes:
        return False
    return True


def get_roi_rect(frame_w: int, frame_h: int, roi_config: dict):
    x1 = int(frame_w * (roi_config["roi_x"] / 100.0))
    y1 = int(frame_h * (roi_config["roi_y"] / 100.0))
    x2 = int(frame_w * min(1.0, (roi_config["roi_x"] + roi_config["roi_w"]) / 100.0))
    y2 = int(frame_h * min(1.0, (roi_config["roi_y"] + roi_config["roi_h"]) / 100.0))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def is_inside_roi(cx: float, cy: float, frame_w: int, frame_h: int, roi_config: dict) -> bool:
    if not roi_config["enable_roi"]:
        return True
    x1, y1, x2, y2 = get_roi_rect(frame_w, frame_h, roi_config)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def draw_roi_overlay(frame_rgb, roi_config: dict):
    if not roi_config["enable_roi"]:
        return frame_rgb
    h, w, _ = frame_rgb.shape
    x1, y1, x2, y2 = get_roi_rect(w, h, roi_config)
    overlay = frame_rgb.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 120, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame_rgb, 0.85, 0, frame_rgb)
    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (60, 120, 255), 2)
    cv2.putText(frame_rgb, "ROI", (x1 + 6, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 120, 255), 2)
    return frame_rgb


def detect_and_annotate(
    frame_bgr,
    frame_index: int,
    source_type: str,
    *,
    use_tracking: bool,
    model,
    conf_threshold: float,
    inference_size: int,
    session: Optional[dict],
    class_meta: dict,
    animal_filter: str,
    animal_classes: dict,
    track_classes: list[str],
    roi_config: dict,
    event_settings: dict,
    register_event_fn,
    process_disappeared_fn,
    draw_box_fn,
    warning_callback=None,
):
    t0 = time.time()
    if use_tracking:
        try:
            results = model.track(
                frame_bgr,
                imgsz=inference_size,
                conf=conf_threshold,
                iou=0.5,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
        except ModuleNotFoundError:
            if warning_callback is not None:
                warning_callback("Трекинг-недоступен: отсутствуют зависимости. Выполняется только детекция.")
            results = model.predict(frame_bgr, imgsz=inference_size, conf=conf_threshold, verbose=False)
    else:
        results = model.predict(frame_bgr, imgsz=inference_size, conf=conf_threshold, verbose=False)
    processing_time_ms = (time.time() - t0) * 1000

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame_rgb.shape
    detections_meta = []

    for result in results:
        boxes = result.boxes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None
        xyxy = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()

        for i, box in enumerate(xyxy):
            cls_id = int(cls_arr[i])
            cls_name = model.names[cls_id]
            conf = float(conf_arr[i])
            track_id = int(ids[i]) if ids is not None else None
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            roi_inside = is_inside_roi(cx, cy, frame_w, frame_h, roi_config)
            is_animal, animal_group = get_class_meta(class_meta, cls_name)
            track_key = f"{session['id']}:{track_id}:{cls_name}" if (session and track_id is not None) else None
            prev_inside = session["track_inside_roi"].get(track_key, False) if track_key else False
            roi_enter = bool(roi_inside and (not prev_inside))
            roi_exit = bool((not roi_inside) and prev_inside)
            if track_key and session is not None:
                session["track_inside_roi"][track_key] = roi_inside
                session["track_last_seen"][track_key] = time.time()
                session["track_class_by_key"][track_key] = cls_name
                if track_key in session["disappeared_track_keys"]:
                    session["disappeared_track_keys"].remove(track_key)

            detection = {
                "class_id": cls_id,
                "class_name": cls_name,
                "is_animal": is_animal,
                "animal_group": animal_group,
                "confidence": conf,
                "box": [x1, y1, x2, y2],
                "track_id": track_id,
                "center_x": cx,
                "center_y": cy,
                "frame_width": frame_w,
                "frame_height": frame_h,
                "roi_inside": roi_inside,
                "roi_enter": roi_enter if track_id is not None else roi_inside,
                "roi_exit": roi_exit if track_id is not None else False,
            }
            detections_meta.append(detection)

            if session is not None:
                register_event_fn(frame_index=frame_index, detection=detection, source_type=source_type, session=session)

            if not class_allowed(cls_name, animal_filter, animal_classes, track_classes):
                continue
            if roi_config["enable_roi"] and not roi_inside:
                continue

            label = f"{cls_name} id:{track_id}" if track_id is not None else cls_name
            frame_rgb = draw_box_fn(frame_rgb, box, label, conf)

    if session is not None and use_tracking:
        process_disappeared_fn(
            frame_index=frame_index,
            source_type=source_type,
            session=session,
            frame_width=frame_w,
            frame_height=frame_h,
        )

    frame_rgb = draw_roi_overlay(frame_rgb, roi_config)
    return frame_rgb, detections_meta, processing_time_ms


def detect_and_draw_live(
    frame_bgr,
    *,
    model,
    conf_threshold: float,
    inference_size: int,
    class_meta: dict,
    animal_filter: str,
    animal_classes: dict,
    track_classes: list[str],
    roi_config: dict,
    draw_box_fn,
):
    results = model.predict(frame_bgr, imgsz=inference_size, conf=conf_threshold, verbose=False)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame_rgb.shape

    for result in results:
        for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
            cls_id = int(result.boxes.cls[i])
            cls_name = model.names[cls_id]
            conf = float(result.boxes.conf[i])
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if not class_allowed(cls_name, animal_filter, animal_classes, track_classes):
                continue
            if roi_config["enable_roi"] and not is_inside_roi(cx, cy, frame_w, frame_h, roi_config):
                continue
            frame_rgb = draw_box_fn(frame_rgb, box, cls_name, conf)

    frame_rgb = draw_roi_overlay(frame_rgb, roi_config)
    return frame_rgb
