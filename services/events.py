import time
import uuid
from typing import Optional


def add_notification(session_state, text: str, *, enabled: bool, toast_callback=None):
    session_state.notifications.append({"timestamp": time.time(), "text": text})
    if len(session_state.notifications) > 200:
        session_state.notifications = session_state.notifications[-200:]
    if enabled and toast_callback is not None:
        toast_callback(text)


def create_event(
    session_state,
    db_insert_event,
    *,
    session: dict,
    event_type: str,
    source_type: str,
    frame_index: int,
    class_name: str = "",
    confidence: Optional[float] = None,
    track_id: Optional[int] = None,
    animal_group: Optional[str] = None,
    is_animal: bool = False,
    roi_inside: bool = False,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    message: str = "",
):
    event = {
        "event_id": str(uuid.uuid4())[:8],
        "session_id": session["id"],
        "event_type": event_type,
        "source_type": source_type,
        "frame_index": frame_index,
        "timestamp": time.time(),
        "class_name": class_name,
        "confidence": confidence if confidence is not None else 0.0,
        "track_id": track_id,
        "animal_group": animal_group,
        "is_animal": is_animal,
        "roi_inside": roi_inside,
        "center_x": center_x,
        "center_y": center_y,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "message": message,
    }
    session_state.events.append(event)
    session["events_count"] += 1
    db_insert_event(event)
    return event


def register_detection_event(
    session_state,
    db_insert_event,
    *,
    session: dict,
    frame_index: int,
    detection: dict,
    source_type: str,
    settings: dict,
    notify_callback,
):
    track_id = detection.get("track_id")
    track_key = None
    is_new_track = False
    if track_id is not None:
        track_key = f"{session['id']}:{track_id}:{detection['class_name']}"
        if track_key not in session["seen_track_keys"]:
            session["seen_track_keys"].add(track_key)
            is_new_track = True

    should_store_event = (track_id is None) or is_new_track
    if should_store_event:
        create_event(
            session_state,
            db_insert_event,
            session=session,
            event_type="object_detected",
            source_type=source_type,
            frame_index=frame_index,
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            track_id=track_id,
            animal_group=detection["animal_group"],
            is_animal=detection["is_animal"],
            roi_inside=detection.get("roi_inside", False),
            center_x=detection.get("center_x"),
            center_y=detection.get("center_y"),
            frame_width=detection.get("frame_width"),
            frame_height=detection.get("frame_height"),
            message=f"Обнаружен объект {detection['class_name']}",
        )

    if detection.get("roi_enter"):
        create_event(
            session_state,
            db_insert_event,
            session=session,
            event_type="roi_enter",
            source_type=source_type,
            frame_index=frame_index,
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            track_id=track_id,
            animal_group=detection["animal_group"],
            is_animal=detection["is_animal"],
            roi_inside=True,
            center_x=detection.get("center_x"),
            center_y=detection.get("center_y"),
            frame_width=detection.get("frame_width"),
            frame_height=detection.get("frame_height"),
            message=f"Вход в ROI: {detection['class_name']}",
        )

    if settings["rule_count_enabled"] and should_store_event and detection["class_name"] == settings["rule_class"]:
        ts_now = time.time()
        bucket = session["class_event_times"].get(settings["rule_class"], [])
        bucket = [ts for ts in bucket if ts_now - ts <= float(settings["rule_t"])]
        bucket.append(ts_now)
        session["class_event_times"][settings["rule_class"]] = bucket
        last_alert = session["rule_last_alert_ts"].get(settings["rule_class"], 0)
        if len(bucket) >= int(settings["rule_n"]) and (ts_now - last_alert) > float(settings["rule_t"]):
            session["rule_last_alert_ts"][settings["rule_class"]] = ts_now
            msg = (
                f"Правило N/T: {len(bucket)} объектов класса {settings['rule_class']} "
                f"за {int(settings['rule_t'])} сек"
            )
            create_event(
                session_state,
                db_insert_event,
                session=session,
                event_type="rule_count",
                source_type=source_type,
                frame_index=frame_index,
                class_name=settings["rule_class"],
                confidence=detection["confidence"],
                track_id=track_id,
                animal_group=detection["animal_group"],
                is_animal=detection["is_animal"],
                roi_inside=detection.get("roi_inside", False),
                center_x=detection.get("center_x"),
                center_y=detection.get("center_y"),
                frame_width=detection.get("frame_width"),
                frame_height=detection.get("frame_height"),
                message=msg,
            )
            if settings["enable_notifications"]:
                notify_callback(msg)

    should_notify_detection = (
        settings["enable_notifications"]
        and detection["confidence"] >= settings["notify_conf_threshold"]
        and detection["class_name"] in settings["notify_classes"]
        and (not settings["enable_roi"] or detection.get("roi_enter", False))
    )
    if should_notify_detection:
        if track_key is not None:
            if track_key in session["notified_track_keys"]:
                return
            session["notified_track_keys"].add(track_key)
        notify_callback(f"Событие: {detection['class_name']} (conf={detection['confidence']:.2f}), кадр {frame_index}")


def process_disappeared_tracks(
    session_state,
    db_insert_event,
    *,
    session: dict,
    frame_index: int,
    source_type: str,
    frame_width: int,
    frame_height: int,
    rule_disappear_enabled: bool,
    rule_disappear_seconds: float,
    enable_notifications: bool,
    notify_callback,
):
    if not rule_disappear_enabled:
        return

    now_ts = time.time()
    for track_key, last_seen in list(session["track_last_seen"].items()):
        if now_ts - last_seen > float(rule_disappear_seconds):
            if track_key in session["disappeared_track_keys"]:
                continue
            session["disappeared_track_keys"].add(track_key)
            disappeared_class = session["track_class_by_key"].get(track_key, "")
            msg = f"Объект исчез > {int(rule_disappear_seconds)} сек: {disappeared_class}"
            create_event(
                session_state,
                db_insert_event,
                session=session,
                event_type="object_disappeared",
                source_type=source_type,
                frame_index=frame_index,
                class_name=disappeared_class,
                confidence=0.0,
                track_id=None,
                animal_group=None,
                is_animal=False,
                roi_inside=False,
                center_x=None,
                center_y=None,
                frame_width=frame_width,
                frame_height=frame_height,
                message=msg,
            )
            if enable_notifications:
                notify_callback(msg)

    # Clean old tracking keys so session state does not grow forever.
    for track_key, last_seen in list(session["track_last_seen"].items()):
        if time.time() - last_seen > float(rule_disappear_seconds) * 20:
            session["track_last_seen"].pop(track_key, None)
            session["track_inside_roi"].pop(track_key, None)
            session["track_class_by_key"].pop(track_key, None)
            session["disappeared_track_keys"].discard(track_key)
