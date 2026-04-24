"""Zone management UI for monitored cameras."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ZONE_TYPE_LABELS = {
    "entry": "Входная зона",
    "restricted": "Запретная зона",
    "service": "Служебная зона",
    "observation": "Зона наблюдения",
    "line_cross": "Линия контроля",
}


def render_zones(
    st,
    *,
    video_sources: list[dict],
    worker_statuses: list[dict],
    zones: list[dict],
    create_zone_fn,
    update_zone_fn,
    set_zone_active_fn,
):
    st.subheader("Камеры и контролируемые зоны")
    production_sources = [source for source in video_sources if source.get("source_type") != "browser_camera"]
    if not production_sources:
        st.info("Сначала добавьте хотя бы один серверный источник видео, чтобы создавать зоны контроля.")
        return

    statuses_by_source = {status["source_id"]: status for status in worker_statuses}
    source_labels = {f"{source['name']} [{source['id']}]": source for source in production_sources}
    zone_rows = []
    for zone in zones:
        source = next((source for source in production_sources if source["id"] == zone["source_id"]), None)
        zone_rows.append(
            {
                "ID": zone["id"],
                "Источник": source["name"] if source is not None else f"source:{zone['source_id']}",
                "Зона": zone["name"],
                "Тип": ZONE_TYPE_LABELS.get(zone["zone_type"], zone["zone_type"]),
                "Геометрия": f"{zone['x']:.0f}/{zone['y']:.0f}/{zone['w']:.0f}/{zone['h']:.0f}",
                "Активна": "да" if zone.get("is_active") else "нет",
                "Описание": zone.get("description") or "",
            }
        )

    top1, top2, top3 = st.columns(3)
    top1.metric("Камер с зонами", len({zone["source_id"] for zone in zones}))
    top2.metric("Всего зон", len(zones))
    top3.metric("Активных зон", sum(1 for zone in zones if zone.get("is_active")))

    left_col, right_col = st.columns([1.15, 1.0], gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown("### Реестр зон")
            if zone_rows:
                st.dataframe(pd.DataFrame(zone_rows), width="stretch", hide_index=True)
            else:
                st.info("Зоны пока не созданы. Начните с добавления первой зоны контроля.")

        with st.container(border=True):
            st.markdown("### Добавить зону")
            selected_source_label = st.selectbox("Камера", options=list(source_labels.keys()), key="zones_create_source")
            selected_source = source_labels[selected_source_label]
            with st.form("create_zone_form", clear_on_submit=True):
                name = st.text_input("Название зоны", value="Зона контроля")
                zone_type = st.selectbox(
                    "Тип зоны",
                    options=list(ZONE_TYPE_LABELS.keys()),
                    format_func=lambda key: ZONE_TYPE_LABELS[key],
                )
                zone_col1, zone_col2 = st.columns(2)
                with zone_col1:
                    x = st.slider("X, %", min_value=0, max_value=95, value=20)
                    w_max = max(1, 100 - x)
                    w = st.slider("W, %", min_value=1, max_value=w_max, value=min(60, w_max))
                with zone_col2:
                    y = st.slider("Y, %", min_value=0, max_value=95, value=20)
                    h_max = max(1, 100 - y)
                    h = st.slider("H, %", min_value=1, max_value=h_max, value=min(60, h_max))
                description = st.text_area("Описание", placeholder="Например: зона перед входной дверью или участок контроля склада.")
                is_active = st.checkbox("Активировать сразу", value=True)
                submitted = st.form_submit_button("Создать зону")
            if submitted:
                create_zone_fn(
                    source_id=selected_source["id"],
                    name=name,
                    zone_type=zone_type,
                    x=float(x),
                    y=float(y),
                    w=float(w),
                    h=float(h),
                    is_active=is_active,
                    description=description,
                )
                st.success("Зона контроля добавлена.")
                st.rerun()

    with right_col:
        with st.container(border=True):
            st.markdown("### Камера и snapshot")
            preview_source_label = st.selectbox("Источник для preview", options=list(source_labels.keys()), key="zones_preview_source")
            preview_source = source_labels[preview_source_label]
            preview_status = statuses_by_source.get(preview_source["id"], {})
            snapshot_path = preview_status.get("last_snapshot_path")
            if snapshot_path and Path(snapshot_path).exists():
                st.image(snapshot_path, width="stretch", caption=f"Snapshot: {preview_source['name']}")
            else:
                st.info("Для выбранной камеры пока нет snapshot от worker. Preview зоны будет доступен после появления кадров.")
            st.caption(
                f"Статус: {preview_status.get('status', 'idle')} · "
                f"Последний кадр: {_fmt_ts(preview_status.get('last_frame_at'))} · "
                f"Локация: {preview_source.get('location') or 'не указана'}"
            )

        source_zone_options = [zone for zone in zones if zone["source_id"] == preview_source["id"]]
        with st.container(border=True):
            st.markdown("### Редактирование зоны")
            if not source_zone_options:
                st.info("Для выбранной камеры зон пока нет.")
            else:
                zone_labels = {f"{zone['name']} [{zone['id']}]": zone for zone in source_zone_options}
                selected_zone_label = st.selectbox("Зона", options=list(zone_labels.keys()), key="zones_edit_select")
                selected_zone = zone_labels[selected_zone_label]
                with st.form("edit_zone_form"):
                    name = st.text_input("Название зоны", value=selected_zone["name"])
                    zone_type = st.selectbox(
                        "Тип зоны",
                        options=list(ZONE_TYPE_LABELS.keys()),
                        index=list(ZONE_TYPE_LABELS.keys()).index(selected_zone["zone_type"])
                        if selected_zone["zone_type"] in ZONE_TYPE_LABELS
                        else 3,
                        format_func=lambda key: ZONE_TYPE_LABELS[key],
                    )
                    zone_col1, zone_col2 = st.columns(2)
                    with zone_col1:
                        x = st.slider("X, %", min_value=0, max_value=95, value=int(selected_zone["x"]), key=f"edit_zone_x_{selected_zone['id']}")
                        w_max = max(1, 100 - x)
                        w = st.slider(
                            "W, %",
                            min_value=1,
                            max_value=w_max,
                            value=min(int(selected_zone["w"]), w_max),
                            key=f"edit_zone_w_{selected_zone['id']}",
                        )
                    with zone_col2:
                        y = st.slider("Y, %", min_value=0, max_value=95, value=int(selected_zone["y"]), key=f"edit_zone_y_{selected_zone['id']}")
                        h_max = max(1, 100 - y)
                        h = st.slider(
                            "H, %",
                            min_value=1,
                            max_value=h_max,
                            value=min(int(selected_zone["h"]), h_max),
                            key=f"edit_zone_h_{selected_zone['id']}",
                        )
                    description = st.text_area("Описание", value=selected_zone.get("description") or "")
                    save = st.form_submit_button("Сохранить изменения")
                if save:
                    update_zone_fn(
                        zone_id=selected_zone["id"],
                        source_id=preview_source["id"],
                        name=name,
                        zone_type=zone_type,
                        x=float(x),
                        y=float(y),
                        w=float(w),
                        h=float(h),
                        description=description,
                    )
                    st.success("Зона обновлена.")
                    st.rerun()

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("Активировать зону", key=f"activate_zone_{selected_zone['id']}"):
                        set_zone_active_fn(zone_id=selected_zone["id"], is_active=True)
                        st.success("Зона активирована.")
                        st.rerun()
                with action_col2:
                    if st.button("Деактивировать зону", key=f"deactivate_zone_{selected_zone['id']}"):
                        set_zone_active_fn(zone_id=selected_zone["id"], is_active=False)
                        st.warning("Зона деактивирована.")
                        st.rerun()


def _fmt_ts(timestamp_value):
    if not timestamp_value:
        return "—"
    return datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
