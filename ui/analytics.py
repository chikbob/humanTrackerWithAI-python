import json
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils.vision import compute_iou


def render_status_panel(st, *, source_mode: str, model_name: str, conf_threshold: float, notify_conf_threshold: float, rotation_angle: int, animal_filter: str, track_classes: list[str], notifications: list[dict]):
    with st.container(border=True):
        st.subheader("🧭 Быстрый статус")
        st.write(f"Источник: **{source_mode}**")
        st.write(f"Модель: **{model_name}**")
        st.write(f"Порог детекции: **{conf_threshold:.2f}**")
        st.write(f"Порог уведомлений: **{notify_conf_threshold:.2f}**")
        st.write(f"Угол поворота: **{rotation_angle}°**")
        st.write(f"Фильтр животных: **{animal_filter}**")
        st.write(f"Фильтр классов: **{', '.join(track_classes) if track_classes else 'все'}**")

    with st.container(border=True):
        st.subheader("🔔 Алерты")
        if notifications:
            recent = notifications[-8:]
            for notification in reversed(recent):
                ts = datetime.fromtimestamp(notification["timestamp"]).strftime("%H:%M:%S")
                st.markdown(f"- `{ts}` {notification['text']}")
        else:
            st.caption("Уведомлений пока нет.")


def render_analytics(st, *, sessions: list[dict], events: list[dict], notifications: list[dict], show_advanced: bool, model):
    st.markdown("---")
    st.subheader("📈 Панель мониторинга и аналитика")

    total_frames = sum(len(session["frames"]) for session in sessions)
    total_events = len(events)
    class_counter = Counter(event["class_name"] for event in events)
    top_class = class_counter.most_common(1)[0][0] if class_counter else "—"

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Сеансов", len(sessions))
    met2.metric("Кадров", total_frames)
    met3.metric("Событий object detected", total_events)
    met4.metric("Топ-класс", top_class)

    if notifications:
        with st.expander("🔔 Последние уведомления", expanded=False):
            notif_df = pd.DataFrame(
                [
                    {
                        "Время": datetime.fromtimestamp(notification["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "Сообщение": notification["text"],
                    }
                    for notification in notifications[-20:]
                ]
            )
            st.dataframe(notif_df.iloc[::-1], use_container_width=True, hide_index=True)

    if show_advanced:
        tab_sessions, tab_events, tab_export, tab_kpi = st.tabs(
            ["Сеансы", "События и динамика", "Экспорт отчётов", "KPI модели"]
        )
    else:
        tab_sessions, tab_events = st.tabs(["Сеансы", "События"])
        tab_export = None
        tab_kpi = None

    with tab_sessions:
        _render_sessions_tab(st, sessions)

    with tab_events:
        _render_events_tab(st, events, show_advanced)

    if show_advanced and tab_export is not None:
        with tab_export:
            _render_export_tab(st, sessions, events)

    if show_advanced and tab_kpi is not None:
        with tab_kpi:
            _render_kpi_tab(st, model)


def _render_sessions_tab(st, sessions: list[dict]):
    if not sessions:
        st.info("Пока нет ни одного сеанса распознавания.")
        return

    sessions_summary = []
    for idx, session in enumerate(sessions, start=1):
        started = datetime.fromtimestamp(session["started_at"]).strftime("%Y-%m-%d %H:%M:%S")
        finished = (
            datetime.fromtimestamp(session["finished_at"]).strftime("%Y-%m-%d %H:%M:%S")
            if session["finished_at"] is not None
            else ""
        )
        duration = session["finished_at"] - session["started_at"] if session["finished_at"] is not None else None
        sessions_summary.append(
            {
                "№": idx,
                "ID (сокр.)": session["id"][:8],
                "Модель": session["model"],
                "Источник": session["source_type"],
                "Путь / камера": session["source_path"],
                "Фильтр животных": session["animal_filter"],
                "Фильтр классов": ", ".join(session["class_filter"]) if session["class_filter"] else "все",
                "Кадров в сеансе": len(session["frames"]),
                "Событий": session["events_count"],
                "Начало": started,
                "Конец": finished,
                "Длительность, с": round(duration, 2) if duration is not None else "",
            }
        )

    df_sessions = pd.DataFrame(sessions_summary)
    st.dataframe(df_sessions, use_container_width=True, hide_index=True)

    session_index = st.number_input(
        "Выберите номер сеанса для детализации",
        min_value=1,
        max_value=len(sessions),
        value=len(sessions),
        step=1,
    )
    selected_session = sessions[session_index - 1]
    frames = selected_session["frames"]
    if not frames:
        st.info("В выбранном сеансе нет кадров.")
        return

    df_frames = pd.DataFrame(
        [
            {
                "Кадр": frame["frame_index"],
                "Время кадра": datetime.fromtimestamp(frame["timestamp"]).strftime("%H:%M:%S"),
                "Размер (W×H)": f"{frame['width']}×{frame['height']}",
                "Угол": frame["rotation_angle"],
                "Время обработки, мс": round(frame["processing_time_ms"], 2),
                "Кол-во детекций": frame["detections_count"],
            }
            for frame in frames
        ]
    )
    st.dataframe(df_frames, use_container_width=True, hide_index=True)


def _render_events_tab(st, events: list[dict], show_advanced: bool):
    if not events:
        st.info("Журнал событий пока пуст.")
        return

    df_events = pd.DataFrame(
        [
            {
                "event_id": event["event_id"],
                "session_id": event["session_id"][:8],
                "event_type": event.get("event_type", "object_detected"),
                "source_type": event["source_type"],
                "frame_index": event["frame_index"],
                "timestamp": datetime.fromtimestamp(event["timestamp"]),
                "class_name": event["class_name"],
                "confidence": round(event["confidence"], 3),
                "track_id": event["track_id"] if event["track_id"] is not None else "",
                "animal_group": event["animal_group"] or "",
                "roi_inside": "да" if event.get("roi_inside") else "нет",
                "message": event.get("message", ""),
                "center_x": event.get("center_x"),
                "center_y": event.get("center_y"),
                "frame_width": event.get("frame_width"),
                "frame_height": event.get("frame_height"),
            }
            for event in events
        ]
    )

    if not show_advanced:
        simple_events = df_events[["timestamp", "class_name", "event_type", "message"]].copy()
        simple_events = simple_events.rename(
            columns={
                "timestamp": "Время",
                "class_name": "Класс",
                "event_type": "Тип события",
                "message": "Описание",
            }
        )
        st.dataframe(simple_events.sort_values("Время", ascending=False), use_container_width=True, hide_index=True)
        st.caption("Включите «Расширенные настройки», чтобы увидеть динамику, тепловую карту и фильтры.")
        return

    col_evt1, col_evt2, col_evt3 = st.columns(3)
    with col_evt1:
        selected_source = st.selectbox("Источник событий", options=["все"] + sorted(df_events["source_type"].unique().tolist()), index=0)
    with col_evt2:
        selected_classes = st.multiselect("Классы для динамики", options=sorted(df_events["class_name"].unique().tolist()), default=[])
    with col_evt3:
        selected_event_types = st.multiselect("Типы событий", options=sorted(df_events["event_type"].unique().tolist()), default=[])

    filtered_events = df_events.copy()
    if selected_source != "все":
        filtered_events = filtered_events[filtered_events["source_type"] == selected_source]
    if selected_classes:
        filtered_events = filtered_events[filtered_events["class_name"].isin(selected_classes)]
    if selected_event_types:
        filtered_events = filtered_events[filtered_events["event_type"].isin(selected_event_types)]

    st.dataframe(filtered_events.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

    timeline = filtered_events.copy()
    timeline["minute"] = timeline["timestamp"].dt.floor("min")
    timeline_series = timeline.groupby("minute").size().rename("events")
    st.caption("Динамика количества событий по минутам")
    st.line_chart(timeline_series)

    class_bar = filtered_events["class_name"].value_counts().rename_axis("class_name").to_frame("count")
    st.caption("Распределение событий по классам")
    st.bar_chart(class_bar)

    heat_df = filtered_events.dropna(subset=["center_x", "center_y", "frame_width", "frame_height"])
    if heat_df.empty:
        return

    heat_size = 96
    heat = np.zeros((heat_size, heat_size), dtype=np.float32)
    for _, row in heat_df.iterrows():
        fw = max(float(row["frame_width"]), 1.0)
        fh = max(float(row["frame_height"]), 1.0)
        nx = min(max(float(row["center_x"]) / fw, 0.0), 1.0)
        ny = min(max(float(row["center_y"]) / fh, 0.0), 1.0)
        xi = min(int(nx * (heat_size - 1)), heat_size - 1)
        yi = min(int(ny * (heat_size - 1)), heat_size - 1)
        heat[yi, xi] += 1.0

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=3, sigmaY=3)
    if float(heat.max()) <= 0:
        return

    heat_norm = (heat / heat.max() * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    st.caption("Тепловая карта движения/появлений объектов")
    st.image(heat_color, use_container_width=False)


def _render_export_tab(st, sessions: list[dict], events: list[dict]):
    if not sessions and not events:
        st.info("Нет данных для экспорта.")
        return

    sessions_export = []
    for session in sessions:
        sessions_export.append(
            {
                "session_id": session["id"],
                "model": session["model"],
                "source_type": session["source_type"],
                "source_path": session["source_path"],
                "animal_filter": session["animal_filter"],
                "class_filter": session["class_filter"],
                "rotation_angle": session["rotation_angle"],
                "started_at": datetime.fromtimestamp(session["started_at"]).isoformat(),
                "finished_at": datetime.fromtimestamp(session["finished_at"]).isoformat() if session["finished_at"] else None,
                "frames_count": len(session["frames"]),
                "events_count": session["events_count"],
            }
        )

    frames_export = []
    for session in sessions:
        for frame in session["frames"]:
            frames_export.append(
                {
                    "session_id": session["id"],
                    "frame_index": frame["frame_index"],
                    "timestamp": datetime.fromtimestamp(frame["timestamp"]).isoformat(),
                    "width": frame["width"],
                    "height": frame["height"],
                    "rotation_angle": frame["rotation_angle"],
                    "processing_time_ms": round(frame["processing_time_ms"], 2),
                    "detections_count": frame["detections_count"],
                }
            )

    events_export = []
    for event in events:
        events_export.append(
            {
                "event_id": event["event_id"],
                "session_id": event["session_id"],
                "event_type": event.get("event_type", "object_detected"),
                "source_type": event["source_type"],
                "frame_index": event["frame_index"],
                "timestamp": datetime.fromtimestamp(event["timestamp"]).isoformat(),
                "class_name": event["class_name"],
                "confidence": round(event["confidence"], 3),
                "track_id": event["track_id"],
                "animal_group": event["animal_group"],
                "is_animal": event["is_animal"],
                "roi_inside": event.get("roi_inside"),
                "center_x": event.get("center_x"),
                "center_y": event.get("center_y"),
                "frame_width": event.get("frame_width"),
                "frame_height": event.get("frame_height"),
                "message": event.get("message", ""),
            }
        )

    df_sessions_export = pd.DataFrame(sessions_export)
    df_frames_export = pd.DataFrame(frames_export)
    df_events_export = pd.DataFrame(events_export)

    st.download_button(
        "⬇️ Экспорт сессий (CSV)",
        data=df_sessions_export.to_csv(index=False).encode("utf-8"),
        file_name="sessions_report.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Экспорт кадров (CSV)",
        data=df_frames_export.to_csv(index=False).encode("utf-8"),
        file_name="frames_report.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Экспорт событий (CSV)",
        data=df_events_export.to_csv(index=False).encode("utf-8"),
        file_name="events_report.csv",
        mime="text/csv",
    )

    full_report = {
        "generated_at": datetime.now().isoformat(),
        "sessions": sessions_export,
        "frames": frames_export,
        "events": events_export,
    }
    st.download_button(
        "⬇️ Экспорт полного отчёта (JSON)",
        data=json.dumps(full_report, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="full_report.json",
        mime="application/json",
    )


def _render_kpi_tab(st, model):
    st.markdown("### KPI модели: Precision / Recall")
    st.caption("Загрузите изображения и CSV-разметку: image_name,class_name,x1,y1,x2,y2")
    kpi_conf = st.slider("Порог confidence для KPI", 0.1, 0.95, 0.25, 0.05, key="kpi_conf")
    kpi_iou = st.slider("Порог IoU для матчинга", 0.1, 0.95, 0.5, 0.05, key="kpi_iou")
    kpi_images = st.file_uploader(
        "Изображения для валидации",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="kpi_images",
    )
    kpi_labels = st.file_uploader("CSV разметки (ground truth)", type=["csv"], key="kpi_labels")

    if not st.button("Рассчитать Precision/Recall", key="run_kpi"):
        return
    if not kpi_images or not kpi_labels:
        st.warning("Нужно загрузить изображения и CSV-разметку.")
        return

    gt_df = pd.read_csv(kpi_labels)
    required_cols = {"image_name", "class_name", "x1", "y1", "x2", "y2"}
    if not required_cols.issubset(set(gt_df.columns)):
        st.error("CSV должен содержать колонки: image_name,class_name,x1,y1,x2,y2")
        return

    per_class = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for uploaded in kpi_images:
        image_name = uploaded.name
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gt_rows = gt_df[gt_df["image_name"] == image_name]
        gt_by_class = {}
        for _, row in gt_rows.iterrows():
            cls_name = str(row["class_name"])
            gt_by_class.setdefault(cls_name, []).append(
                [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            )

        pred_results = model.predict(img_bgr, imgsz=640, conf=kpi_conf, verbose=False)
        pred_by_class = {}
        for result in pred_results:
            for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                cls_id = int(result.boxes.cls[i])
                cls_name = model.names[cls_id]
                pred_by_class.setdefault(cls_name, []).append([float(v) for v in box.tolist()])

        all_classes_eval = set(gt_by_class.keys()) | set(pred_by_class.keys())
        for cls_name in all_classes_eval:
            gt_boxes = gt_by_class.get(cls_name, [])
            pr_boxes = pred_by_class.get(cls_name, [])
            matched_gt = set()
            tp = 0
            fp = 0

            for pred_box in pr_boxes:
                best_iou = 0.0
                best_gt_idx = None
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_gt_idx is not None and best_iou >= kpi_iou:
                    matched_gt.add(best_gt_idx)
                    tp += 1
                else:
                    fp += 1

            fn = len(gt_boxes) - len(matched_gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            entry = per_class.setdefault(cls_name, {"TP": 0, "FP": 0, "FN": 0})
            entry["TP"] += tp
            entry["FP"] += fp
            entry["FN"] += fn

    rows = []
    for cls_name, values in sorted(per_class.items()):
        tp = values["TP"]
        fp = values["FP"]
        fn = values["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append(
            {
                "class_name": cls_name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    met_a, met_b, met_c = st.columns(3)
    met_a.metric("Overall Precision", f"{overall_precision:.3f}")
    met_b.metric("Overall Recall", f"{overall_recall:.3f}")
    met_c.metric("TP / FP / FN", f"{total_tp} / {total_fp} / {total_fn}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
