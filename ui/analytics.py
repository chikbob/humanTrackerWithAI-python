import json
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils.vision import compute_iou


def render_status_panel(
    st,
    *,
    source_mode: str,
    model_name: str,
    conf_threshold: float,
    notify_conf_threshold: float,
    rotation_angle: int,
    animal_filter: str,
    track_classes: list[str],
    notifications: list[dict],
):
    with st.container(border=True):
        st.subheader("🧭 Состояние поста наблюдения")
        st.write(f"Источник видеопотока: **{source_mode}**")
        st.write(f"Модель анализа: **{model_name}**")
        st.write(f"Порог детекции: **{conf_threshold:.2f}**")
        st.write(f"Порог тревожных уведомлений: **{notify_conf_threshold:.2f}**")
        st.write(f"Поворот изображения: **{rotation_angle}°**")
        st.write(f"Фильтр объектов: **{animal_filter}**")
        st.write(f"Контроль по классам: **{', '.join(track_classes) if track_classes else 'все объекты'}**")

    with st.container(border=True):
        st.subheader("🔔 Оповещения проходной")
        if notifications:
            recent = notifications[-8:]
            for notification in reversed(recent):
                ts = datetime.fromtimestamp(notification["timestamp"]).strftime("%H:%M:%S")
                st.markdown(f"- `{ts}` {notification['text']}")
        else:
            st.caption("Оповещений по входной зоне пока нет.")


def render_analytics(
    st,
    *,
    sessions: list[dict],
    events: list[dict],
    notifications: list[dict],
    show_advanced: bool,
    model,
    employees: list[dict],
    access_logs: list[dict],
):
    st.markdown("---")
    st.subheader("📊 Оперативная панель предприятия")

    total_frames = sum(len(session["frames"]) for session in sessions)
    total_events = len(events)
    total_domain_events = sum(1 for event in events if event.get("event_scope") == "domain")
    top_event = Counter(event["event_type"] for event in events if event.get("event_scope") == "domain").most_common(1)
    top_event_name = top_event[0][0] if top_event else "—"

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Активных сеансов мониторинга", len(sessions))
    met2.metric("Обработано кадров", total_frames)
    met3.metric("Событий проходной", total_domain_events)
    met4.metric("Основной тип события", top_event_name)

    if notifications:
        with st.expander("Последние уведомления поста наблюдения", expanded=False):
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

    tab_monitoring, tab_journal, tab_employees, tab_stats = st.tabs(
        [
            "Онлайн-мониторинг входной зоны",
            "Журнал событий",
            "Список сотрудников",
            "Статистика проходов",
        ]
    )

    with tab_monitoring:
        _render_monitoring_tab(st, sessions)

    with tab_journal:
        _render_events_tab(st, events, access_logs, show_advanced)

    with tab_employees:
        _render_employees_tab(st, employees)

    with tab_stats:
        _render_access_stats_tab(st, events, access_logs)

    if show_advanced:
        with st.expander("Служебные разделы", expanded=False):
            subtab_export, subtab_kpi = st.tabs(["Экспорт отчетов", "KPI модели"])
            with subtab_export:
                _render_export_tab(st, sessions, events)
            with subtab_kpi:
                _render_kpi_tab(st, model)


def _render_monitoring_tab(st, sessions: list[dict]):
    st.markdown("**Онлайн-мониторинг входной зоны**")
    if not sessions:
        st.info("Сеансы мониторинга входной зоны пока не запускались.")
        return

    sessions_summary = []
    for idx, session in enumerate(sessions, start=1):
        started = datetime.fromtimestamp(session["started_at"]).strftime("%Y-%m-%d %H:%M:%S")
        finished = (
            datetime.fromtimestamp(session["finished_at"]).strftime("%Y-%m-%d %H:%M:%S")
            if session["finished_at"] is not None
            else "идет наблюдение"
        )
        duration = session["finished_at"] - session["started_at"] if session["finished_at"] is not None else None
        sessions_summary.append(
            {
                "№": idx,
                "Сеанс": session["id"][:8],
                "Источник": session["source_type"],
                "Камера / файл": session["source_path"],
                "Модель": session["model"],
                "Начало": started,
                "Завершение": finished,
                "Длительность, сек": round(duration, 2) if duration is not None else "",
                "Кадров": len(session["frames"]),
                "Событий": session["events_count"],
            }
        )

    df_sessions = pd.DataFrame(sessions_summary)
    st.dataframe(df_sessions, use_container_width=True, hide_index=True)

    session_index = st.number_input(
        "Выберите сеанс мониторинга для детализации",
        min_value=1,
        max_value=len(sessions),
        value=len(sessions),
        step=1,
        key="monitoring_session_index",
    )
    selected_session = sessions[session_index - 1]
    frames = selected_session["frames"]
    if not frames:
        st.info("В выбранном сеансе еще нет кадров.")
        return

    df_frames = pd.DataFrame(
        [
            {
                "Кадр": frame["frame_index"],
                "Время кадра": datetime.fromtimestamp(frame["timestamp"]).strftime("%H:%M:%S"),
                "Разрешение": f"{frame['width']}×{frame['height']}",
                "Поворот": frame["rotation_angle"],
                "Обработка, мс": round(frame["processing_time_ms"], 2),
                "Детекций": frame["detections_count"],
            }
            for frame in frames
        ]
    )
    st.dataframe(df_frames, use_container_width=True, hide_index=True)


def _render_events_tab(st, events: list[dict], access_logs: list[dict], show_advanced: bool):
    st.markdown("**Журнал событий**")
    if not events:
        st.info("Журнал проходов и событий входной зоны пока пуст.")
        return

    df_events = pd.DataFrame(
        [
            {
                "event_id": event["event_id"],
                "session_id": event["session_id"][:8],
                "scope": event.get("event_scope", "raw"),
                "event_type": event.get("event_type", "object_detected"),
                "source_type": event["source_type"],
                "frame_index": event["frame_index"],
                "timestamp": datetime.fromtimestamp(event["timestamp"]),
                "class_name": event["class_name"],
                "confidence": round(event["confidence"], 3),
                "track_id": event["track_id"] if event["track_id"] is not None else "",
                "roi_inside": "да" if event.get("roi_inside") else "нет",
                "message": event.get("message", ""),
            }
            for event in events
        ]
    )

    if access_logs:
        st.caption("Журнал проходов предприятия")
        df_access_logs = pd.DataFrame(
            [
                {
                    "ID": row["id"],
                    "Время": datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "Сотрудник": row["employee_name"] or "не определен",
                    "Точка прохода": row["access_point_name"] or "не задана",
                    "Тип события": row["event_type"],
                    "Уверенность": round(row["confidence"], 3) if row["confidence"] is not None else "",
                    "Примечание": row["note"] or "",
                }
                for row in access_logs
            ]
        )
        st.dataframe(df_access_logs, use_container_width=True, hide_index=True)
    else:
        st.caption("Журнал проходов пока пуст. Доменные события будут появляться здесь автоматически.")

    if not show_advanced:
        simple_events = df_events[["timestamp", "scope", "event_type", "message"]].copy()
        simple_events = simple_events.rename(
            columns={
                "timestamp": "Время",
                "scope": "Уровень",
                "event_type": "Тип события",
                "message": "Описание",
            }
        )
        st.caption("Общий журнал событий входной зоны")
        st.dataframe(simple_events.sort_values("Время", ascending=False), use_container_width=True, hide_index=True)
        return

    col_evt1, col_evt2, col_evt3 = st.columns(3)
    with col_evt1:
        selected_scope = st.selectbox(
            "Уровень события",
            options=["все"] + sorted(df_events["scope"].unique().tolist()),
            index=0,
        )
    with col_evt2:
        selected_event_types = st.multiselect(
            "Типы событий",
            options=sorted(df_events["event_type"].unique().tolist()),
            default=[],
        )
    with col_evt3:
        selected_source = st.selectbox(
            "Источник видеопотока",
            options=["все"] + sorted(df_events["source_type"].unique().tolist()),
            index=0,
        )

    filtered_events = df_events.copy()
    if selected_scope != "все":
        filtered_events = filtered_events[filtered_events["scope"] == selected_scope]
    if selected_event_types:
        filtered_events = filtered_events[filtered_events["event_type"].isin(selected_event_types)]
    if selected_source != "все":
        filtered_events = filtered_events[filtered_events["source_type"] == selected_source]

    st.dataframe(filtered_events.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

    timeline = filtered_events.copy()
    timeline["minute"] = timeline["timestamp"].dt.floor("min")
    timeline_series = timeline.groupby("minute").size().rename("events")
    st.caption("Динамика событий входной зоны по минутам")
    st.line_chart(timeline_series)

    event_bar = filtered_events["event_type"].value_counts().rename_axis("event_type").to_frame("count")
    st.caption("Распределение событий проходной")
    st.bar_chart(event_bar)


def _render_employees_tab(st, employees: list[dict]):
    st.markdown("**Список сотрудников**")
    if not employees:
        empty_df = pd.DataFrame(columns=["ID", "ФИО", "Подразделение", "Должность", "Статус", "Создан"])
        st.dataframe(empty_df, use_container_width=True, hide_index=True)
        st.caption("Справочник сотрудников пока не заполнен.")
        return

    df_employees = pd.DataFrame(
        [
            {
                "ID": employee["id"],
                "ФИО": employee["full_name"],
                "Подразделение": employee["department"] or "",
                "Должность": employee["position"] or "",
                "Статус": employee["status"] or "",
                "Создан": datetime.fromtimestamp(employee["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
                if employee["created_at"]
                else "",
            }
            for employee in employees
        ]
    )
    st.dataframe(df_employees, use_container_width=True, hide_index=True)


def _render_access_stats_tab(st, events: list[dict], access_logs: list[dict]):
    st.markdown("**Статистика проходов**")

    domain_events = [event for event in events if event.get("event_scope") == "domain"]
    entered_events = [event for event in domain_events if event.get("event_type") == "person_entered_entry_zone"]
    left_events = [event for event in domain_events if event.get("event_type") == "person_left_entry_zone"]
    prolonged_events = [event for event in domain_events if event.get("event_type") == "prolonged_presence_near_entry"]

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Входов в зону прохода", len(entered_events))
    met2.metric("Выходов из зоны прохода", len(left_events))
    met3.metric("Длительных присутствий", len(prolonged_events))
    met4.metric("Записей в журнале проходов", len(access_logs))

    if not domain_events:
        st.info("Статистика появится после первых событий во входной зоне предприятия.")
        return

    df_domain = pd.DataFrame(
        [
            {
                "timestamp": datetime.fromtimestamp(event["timestamp"]),
                "event_type": event["event_type"],
                "confidence": round(event["confidence"], 3),
            }
            for event in domain_events
        ]
    )
    df_domain["date"] = df_domain["timestamp"].dt.floor("D")
    st.caption("События проходной по дням")
    st.line_chart(df_domain.groupby("date").size().rename("events"))

    event_distribution = df_domain["event_type"].value_counts().rename_axis("event_type").to_frame("count")
    st.caption("Распределение доменных событий")
    st.bar_chart(event_distribution)


def _render_export_tab(st, sessions: list[dict], events: list[dict]):
    if not sessions and not events:
        st.info("Нет данных для выгрузки отчетов.")
        return

    sessions_export = []
    for session in sessions:
        sessions_export.append(
            {
                "session_id": session["id"],
                "model": session["model"],
                "source_type": session["source_type"],
                "source_path": session["source_path"],
                "rotation_angle": session["rotation_angle"],
                "started_at": datetime.fromtimestamp(session["started_at"]).isoformat(),
                "finished_at": datetime.fromtimestamp(session["finished_at"]).isoformat() if session["finished_at"] else None,
                "frames_count": len(session["frames"]),
                "events_count": session["events_count"],
            }
        )

    events_export = []
    for event in events:
        events_export.append(
            {
                "event_id": event["event_id"],
                "session_id": event["session_id"],
                "event_scope": event.get("event_scope", "raw"),
                "event_type": event.get("event_type", "object_detected"),
                "source_type": event["source_type"],
                "frame_index": event["frame_index"],
                "timestamp": datetime.fromtimestamp(event["timestamp"]).isoformat(),
                "class_name": event["class_name"],
                "confidence": round(event["confidence"], 3),
                "track_id": event["track_id"],
                "roi_inside": event.get("roi_inside"),
                "message": event.get("message", ""),
            }
        )

    df_sessions_export = pd.DataFrame(sessions_export)
    df_events_export = pd.DataFrame(events_export)

    st.download_button(
        "⬇️ Экспорт сеансов мониторинга (CSV)",
        data=df_sessions_export.to_csv(index=False).encode("utf-8"),
        file_name="monitoring_sessions.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Экспорт журнала событий (CSV)",
        data=df_events_export.to_csv(index=False).encode("utf-8"),
        file_name="entry_zone_events.csv",
        mime="text/csv",
    )

    full_report = {
        "generated_at": datetime.now().isoformat(),
        "sessions": sessions_export,
        "events": events_export,
    }
    st.download_button(
        "⬇️ Экспорт полного отчета (JSON)",
        data=json.dumps(full_report, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="enterprise_access_report.json",
        mime="application/json",
    )


def _render_kpi_tab(st, model):
    st.markdown("### KPI модели видеонаблюдения")
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
