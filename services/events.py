import time
import uuid
from typing import Optional

from services.identity_service import get_identity_placeholder_result


def add_notification(session_state, text: str, *, enabled: bool, toast_callback=None):
    session_state.notifications.append({"timestamp": time.time(), "text": text})
    if len(session_state.notifications) > 200:
        session_state.notifications = session_state.notifications[-200:]
    if enabled and toast_callback is not None:
        toast_callback(text)


def create_persisted_event(
    session_state,
    db_insert_event,
    *,
    session: dict,
    event_scope: str,
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
    employee_id: Optional[int] = None,
    access_point_id: Optional[int] = None,
    access_log_id: Optional[int] = None,
    identified_employee_id: Optional[int] = None,
    identification_confidence: Optional[float] = None,
    identification_status: Optional[str] = None,
):
    """Persist either a raw CV event or a domain event into the unified event journal."""
    identity_state = getattr(session_state, "identity_gallery_state", None)
    identity_result = get_identity_placeholder_result(identity_state)
    event = {
        "event_id": str(uuid.uuid4())[:8],
        "session_id": session["id"],
        "event_scope": event_scope,
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
        "employee_id": employee_id,
        "access_point_id": access_point_id,
        "access_log_id": access_log_id,
        # These fields prepare the journal for future employee identification.
        "identified_employee_id": identified_employee_id,
    }
    event["identified_employee_id"] = (
        identified_employee_id
        if identified_employee_id is not None
        else identity_result["identified_employee_id"]
    )
    event["identification_confidence"] = (
        identification_confidence
        if identification_confidence is not None
        else identity_result["identification_confidence"]
    )
    event["identification_status"] = (
        identification_status
        if identification_status is not None
        else identity_result["identification_status"]
    )
    session_state.events.append(event)
    session["events_count"] += 1
    db_insert_event(event)
    return event


def create_raw_detection_event(session_state, db_insert_event, **kwargs):
    """Store low-level computer-vision telemetry independently from business semantics."""
    return create_persisted_event(
        session_state,
        db_insert_event,
        event_scope="raw",
        **kwargs,
    )


def create_domain_entry_event(session_state, db_insert_event, **kwargs):
    """Store interpreted entry-zone events for the enterprise access domain."""
    return create_persisted_event(
        session_state,
        db_insert_event,
        event_scope="domain",
        **kwargs,
    )


def _get_track_key(session: dict, detection: dict):
    track_id = detection.get("track_id")
    if track_id is None:
        return None
    return f"{session['id']}:{track_id}:{detection['class_name']}"


def _is_person_detection(detection: dict) -> bool:
    return detection.get("class_name") == "person"


def _get_domain_flags(session: dict, track_key: Optional[str]):
    if track_key is None:
        return {}
    return session["track_domain_flags"].setdefault(track_key, {})


def _notify_once_for_track(session: dict, track_key: Optional[str], flag_name: str) -> bool:
    if track_key is None:
        return True
    flags = _get_domain_flags(session, track_key)
    if flags.get(flag_name):
        return False
    flags[flag_name] = True
    return True


def _remember_entry_attempt(session: dict, track_key: Optional[str], timestamp_value: float):
    if track_key is None:
        return []
    attempts = session["track_entry_timestamps"].setdefault(track_key, [])
    attempts = [entry_ts for entry_ts in attempts if timestamp_value - entry_ts <= 60.0]
    attempts.append(timestamp_value)
    session["track_entry_timestamps"][track_key] = attempts
    return attempts


def register_raw_detection_events(
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
    """Write raw detection-layer events that come directly from model output and tracking."""
    track_id = detection.get("track_id")
    track_key = _get_track_key(session, detection)
    is_new_track = False
    if track_key is not None and track_key not in session["seen_track_keys"]:
        session["seen_track_keys"].add(track_key)
        is_new_track = True

    should_store_event = (track_id is None) or is_new_track
    if should_store_event:
        create_raw_detection_event(
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
        create_raw_detection_event(
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

    if detection.get("roi_exit"):
        create_raw_detection_event(
            session_state,
            db_insert_event,
            session=session,
            event_type="roi_exit",
            source_type=source_type,
            frame_index=frame_index,
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            track_id=track_id,
            animal_group=detection["animal_group"],
            is_animal=detection["is_animal"],
            roi_inside=False,
            center_x=detection.get("center_x"),
            center_y=detection.get("center_y"),
            frame_width=detection.get("frame_width"),
            frame_height=detection.get("frame_height"),
            message=f"Выход из ROI: {detection['class_name']}",
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
            create_raw_detection_event(
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


def register_entry_zone_domain_events(
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
    """
    Translate raw person tracking signals into enterprise entry-zone events.

    ROI is treated as the entry zone itself. Events are intentionally business-oriented
    and separate from low-level detector telemetry.
    """
    if not _is_person_detection(detection):
        return

    track_key = _get_track_key(session, detection)
    now_ts = time.time()
    if track_key is not None:
        session["track_first_seen"].setdefault(track_key, now_ts)

    # First appearance near the controlled entrance area.
    if _notify_once_for_track(session, track_key, "person_detected_near_entry"):
        create_domain_entry_event(
            session_state,
            db_insert_event,
            session=session,
            event_type="person_detected_near_entry",
            source_type=source_type,
            frame_index=frame_index,
            class_name="person",
            confidence=detection["confidence"],
            track_id=detection.get("track_id"),
            roi_inside=detection.get("roi_inside", False),
            center_x=detection.get("center_x"),
            center_y=detection.get("center_y"),
            frame_width=detection.get("frame_width"),
            frame_height=detection.get("frame_height"),
            message="Человек обнаружен рядом со входной зоной",
            access_point_id=settings.get("default_access_point_id"),
        )
    if _notify_once_for_track(session, track_key, "unknown_person_detected"):
        create_domain_entry_event(
            session_state,
            db_insert_event,
            session=session,
            event_type="unknown_person_detected",
            source_type=source_type,
            frame_index=frame_index,
            class_name="person",
            confidence=detection["confidence"],
            track_id=detection.get("track_id"),
            roi_inside=detection.get("roi_inside", False),
            center_x=detection.get("center_x"),
            center_y=detection.get("center_y"),
            frame_width=detection.get("frame_width"),
            frame_height=detection.get("frame_height"),
            message="Личность человека не установлена; требуется последующее сопоставление с сотрудником",
            access_point_id=settings.get("default_access_point_id"),
            identification_status="unknown",
        )

    # Transition into the entry zone.
    if detection.get("roi_enter") and _notify_once_for_track(session, track_key, "person_entered_entry_zone"):
        entry_attempts = _remember_entry_attempt(session, track_key, now_ts)
        create_domain_entry_event(
            session_state,
            db_insert_event,
            session=session,
            event_type="person_entered_entry_zone",
            source_type=source_type,
            frame_index=frame_index,
            class_name="person",
            confidence=detection["confidence"],
            track_id=detection.get("track_id"),
            roi_inside=True,
            center_x=detection.get("center_x"),
            center_y=detection.get("center_y"),
            frame_width=detection.get("frame_width"),
            frame_height=detection.get("frame_height"),
            message="Человек вошел во входную зону предприятия",
            access_point_id=settings.get("default_access_point_id"),
        )
        if settings["enable_notifications"]:
            notify_callback("Событие проходной: человек вошел во входную зону")
        repeated_entry_cooldown = float(settings.get("event_cooldown", 5))
        if len(entry_attempts) >= 2 and (entry_attempts[-1] - entry_attempts[-2]) <= repeated_entry_cooldown:
            if _notify_once_for_track(session, track_key, "repeated_entry_attempt"):
                create_domain_entry_event(
                    session_state,
                    db_insert_event,
                    session=session,
                    event_type="repeated_entry_attempt",
                    source_type=source_type,
                    frame_index=frame_index,
                    class_name="person",
                    confidence=detection["confidence"],
                    track_id=detection.get("track_id"),
                    roi_inside=True,
                    center_x=detection.get("center_x"),
                    center_y=detection.get("center_y"),
                    frame_width=detection.get("frame_width"),
                    frame_height=detection.get("frame_height"),
                    message="Зафиксирована повторная попытка входа в контролируемую зону за короткий интервал",
                    access_point_id=settings.get("default_access_point_id"),
                )

    # Explicit exit from the entry zone while the track is still visible.
    if detection.get("roi_exit") and _notify_once_for_track(session, track_key, "person_left_entry_zone"):
        create_domain_entry_event(
            session_state,
            db_insert_event,
            session=session,
            event_type="person_left_entry_zone",
            source_type=source_type,
            frame_index=frame_index,
            class_name="person",
            confidence=detection["confidence"],
            track_id=detection.get("track_id"),
            roi_inside=False,
            center_x=detection.get("center_x"),
            center_y=detection.get("center_y"),
            frame_width=detection.get("frame_width"),
            frame_height=detection.get("frame_height"),
            message="Человек покинул входную зону предприятия",
            access_point_id=settings.get("default_access_point_id"),
        )

    # Long presence near the entrance is kept separate from access entry/exit.
    prolonged_seconds = float(settings.get("prolonged_presence_seconds", 10))
    if track_key is not None:
        first_seen_ts = session["track_first_seen"].get(track_key, now_ts)
        has_prolonged_presence = (now_ts - first_seen_ts) >= prolonged_seconds
        if has_prolonged_presence and _notify_once_for_track(session, track_key, "prolonged_presence_near_entry"):
            create_domain_entry_event(
                session_state,
                db_insert_event,
                session=session,
                event_type="prolonged_presence_near_entry",
                source_type=source_type,
                frame_index=frame_index,
                class_name="person",
                confidence=detection["confidence"],
                track_id=detection.get("track_id"),
                roi_inside=detection.get("roi_inside", False),
                center_x=detection.get("center_x"),
                center_y=detection.get("center_y"),
                frame_width=detection.get("frame_width"),
                frame_height=detection.get("frame_height"),
                message=f"Человек находится рядом со входной зоной более {int(prolonged_seconds)} сек",
                access_point_id=settings.get("default_access_point_id"),
            )
            if settings["enable_notifications"]:
                notify_callback("Событие проходной: зафиксировано длительное присутствие у входа")


def register_detection_and_entry_events(
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
    """Entry point that explicitly splits raw telemetry from domain interpretation."""
    register_raw_detection_events(
        session_state,
        db_insert_event,
        session=session,
        frame_index=frame_index,
        detection=detection,
        source_type=source_type,
        settings=settings,
        notify_callback=notify_callback,
    )
    register_entry_zone_domain_events(
        session_state,
        db_insert_event,
        session=session,
        frame_index=frame_index,
        detection=detection,
        source_type=source_type,
        settings=settings,
        notify_callback=notify_callback,
    )


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
    default_access_point_id=None,
):
    """Finalize raw and domain events when a tracked object disappears from the scene."""
    if not rule_disappear_enabled:
        return

    now_ts = time.time()
    for track_key, last_seen in list(session["track_last_seen"].items()):
        if now_ts - last_seen > float(rule_disappear_seconds):
            if track_key in session["disappeared_track_keys"]:
                continue
            session["disappeared_track_keys"].add(track_key)
            disappeared_class = session["track_class_by_key"].get(track_key, "")
            was_inside_entry_zone = session["track_inside_roi"].get(track_key, False)
            msg = f"Объект исчез > {int(rule_disappear_seconds)} сек: {disappeared_class}"
            create_raw_detection_event(
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

            # If a tracked person disappears after being inside the entry zone,
            # treat this as leaving the zone to keep domain events consistent.
            if disappeared_class == "person":
                flags = _get_domain_flags(session, track_key)
                if was_inside_entry_zone and not flags.get("person_left_entry_zone"):
                    flags["person_left_entry_zone"] = True
                    create_domain_entry_event(
                        session_state,
                        db_insert_event,
                        session=session,
                        event_type="person_left_entry_zone",
                        source_type=source_type,
                        frame_index=frame_index,
                        class_name="person",
                        confidence=0.0,
                        track_id=None,
                        roi_inside=False,
                        center_x=None,
                        center_y=None,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        message="Человек покинул входную зону предприятия",
                        access_point_id=default_access_point_id,
                    )

    # Clean old tracking keys so session state does not grow forever.
    for track_key, last_seen in list(session["track_last_seen"].items()):
        if time.time() - last_seen > float(rule_disappear_seconds) * 20:
            session["track_last_seen"].pop(track_key, None)
            session["track_inside_roi"].pop(track_key, None)
            session["track_class_by_key"].pop(track_key, None)
            session["disappeared_track_keys"].discard(track_key)
            session["track_first_seen"].pop(track_key, None)
            session["track_domain_flags"].pop(track_key, None)
            session["track_entry_timestamps"].pop(track_key, None)
