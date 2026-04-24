"""Zone management UI for monitored cameras."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from services.rules import RULE_SEVERITY_OPTIONS, ZONE_RULE_TYPES


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
    zone_rules: list[dict],
    access_context: dict,
    create_zone_fn,
    update_zone_fn,
    set_zone_active_fn,
    create_zone_rule_fn,
    update_zone_rule_fn,
    set_zone_rule_active_fn,
):
    can_manage = access_context.get("role") == "admin"
    if not can_manage:
        st.info("Управление зонами и правилами доступно только администратору. Экран открыт в режиме просмотра.")
    st.subheader("Камеры и контролируемые зоны")
    production_sources = [source for source in video_sources if source.get("source_type") != "browser_camera"]
    if not production_sources:
        st.info("Сначала добавьте хотя бы один серверный источник видео, чтобы создавать зоны контроля.")
        return

    statuses_by_source = {status["source_id"]: status for status in worker_statuses}
    source_labels = {f"{source['name']} [{source['id']}]": source for source in production_sources}
    zone_rows = []
    rule_rows = []
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
    for rule in zone_rules:
        zone = next((zone for zone in zones if zone["id"] == rule["zone_id"]), None)
        source = next((source for source in production_sources if zone is not None and source["id"] == zone["source_id"]), None)
        rule_rows.append(
            {
                "ID": rule["id"],
                "Источник": source["name"] if source is not None else "—",
                "Зона": zone["name"] if zone is not None else f"zone:{rule['zone_id']}",
                "Правило": ZONE_RULE_TYPES.get(rule["rule_type"], {}).get("label", rule["rule_type"]),
                "Порог, сек": rule["threshold_seconds"],
                "Порог, кол-во": rule["threshold_count"],
                "Cooldown, сек": rule["cooldown_seconds"],
                "Серьезность": RULE_SEVERITY_OPTIONS.get(rule["severity"], rule["severity"]),
                "Активно": "да" if rule.get("is_active") else "нет",
            }
        )

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Камер с зонами", len({zone["source_id"] for zone in zones}))
    top2.metric("Всего зон", len(zones))
    top3.metric("Активных зон", sum(1 for zone in zones if zone.get("is_active")))
    top4.metric("Активных правил", sum(1 for rule in zone_rules if rule.get("is_active")))

    zones_tab, rules_tab = st.tabs(["Зоны контроля", "Правила зон"])

    with zones_tab:
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
                    if not can_manage:
                        st.error("Недостаточно прав для добавления зоны.")
                    else:
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
            _render_zone_preview_and_editor(
                st,
                source_labels=source_labels,
                statuses_by_source=statuses_by_source,
                zones=zones,
                can_manage=can_manage,
                update_zone_fn=update_zone_fn,
                set_zone_active_fn=set_zone_active_fn,
            )

    with rules_tab:
        left_col, right_col = st.columns([1.1, 1.0], gap="large")
        with left_col:
            with st.container(border=True):
                st.markdown("### Реестр правил")
                if rule_rows:
                    st.dataframe(pd.DataFrame(rule_rows), width="stretch", hide_index=True)
                else:
                    st.info("Правила пока не созданы. Начните с базовых сценариев типа «Человек в зоне» или «Длительное присутствие».")
            with st.container(border=True):
                st.markdown("### Добавить правило")
                if not zones:
                    st.info("Сначала создайте хотя бы одну активную зону.")
                else:
                    zone_options = {f"{zone['name']} [{zone['id']}]": zone for zone in zones}
                    selected_zone_label = st.selectbox("Зона", options=list(zone_options.keys()), key="zone_rule_create_zone")
                    selected_zone = zone_options[selected_zone_label]
                    with st.form("create_zone_rule_form", clear_on_submit=True):
                        rule_type = st.selectbox(
                            "Тип правила",
                            options=list(ZONE_RULE_TYPES.keys()),
                            format_func=lambda key: ZONE_RULE_TYPES[key]["label"],
                        )
                        threshold_col1, threshold_col2, threshold_col3 = st.columns(3)
                        with threshold_col1:
                            threshold_seconds = st.number_input("Порог, сек", min_value=1, max_value=3600, value=10, step=1)
                        with threshold_col2:
                            threshold_count = st.number_input("Порог, кол-во", min_value=1, max_value=100, value=3, step=1)
                        with threshold_col3:
                            cooldown_seconds = st.number_input("Cooldown, сек", min_value=0, max_value=3600, value=5, step=1)
                        severity = st.selectbox(
                            "Серьезность",
                            options=list(RULE_SEVERITY_OPTIONS.keys()),
                            format_func=lambda key: RULE_SEVERITY_OPTIONS[key],
                        )
                        description = st.text_area("Описание", placeholder="Например: тревога при нахождении человека в закрытой зоне более 20 секунд.")
                        is_active = st.checkbox("Активировать сразу", value=True)
                        submitted = st.form_submit_button("Создать правило")
                    if submitted:
                        if not can_manage:
                            st.error("Недостаточно прав для добавления правила.")
                        else:
                            create_zone_rule_fn(
                                zone_id=selected_zone["id"],
                                rule_type=rule_type,
                                threshold_seconds=int(threshold_seconds),
                                threshold_count=int(threshold_count),
                                cooldown_seconds=int(cooldown_seconds),
                                is_active=is_active,
                                severity=severity,
                                description=description,
                            )
                            st.success("Правило зоны добавлено.")
                        st.rerun()

        with right_col:
            with st.container(border=True):
                st.markdown("### Редактирование правила")
                if not rule_rows:
                    st.info("Для редактирования сначала создайте правило.")
                else:
                    rule_options = {f"{row['Правило']} [{row['ID']}]": row for row in rule_rows}
                    selected_rule_label = st.selectbox("Правило", options=list(rule_options.keys()), key="zone_rule_edit_select")
                    selected_rule_row = rule_options[selected_rule_label]
                    selected_rule = next(rule for rule in zone_rules if rule["id"] == selected_rule_row["ID"])
                    zone_options = {f"{zone['name']} [{zone['id']}]": zone for zone in zones}
                    default_zone_label = next(label for label, zone in zone_options.items() if zone["id"] == selected_rule["zone_id"])
                    with st.form("edit_zone_rule_form"):
                        zone_label = st.selectbox(
                            "Зона",
                            options=list(zone_options.keys()),
                            index=list(zone_options.keys()).index(default_zone_label),
                        )
                        rule_type = st.selectbox(
                            "Тип правила",
                            options=list(ZONE_RULE_TYPES.keys()),
                            index=list(ZONE_RULE_TYPES.keys()).index(selected_rule["rule_type"])
                            if selected_rule["rule_type"] in ZONE_RULE_TYPES
                            else 0,
                            format_func=lambda key: ZONE_RULE_TYPES[key]["label"],
                        )
                        threshold_col1, threshold_col2, threshold_col3 = st.columns(3)
                        with threshold_col1:
                            threshold_seconds = st.number_input(
                                "Порог, сек",
                                min_value=1,
                                max_value=3600,
                                value=int(selected_rule["threshold_seconds"]),
                                key=f"rule_edit_sec_{selected_rule['id']}",
                            )
                        with threshold_col2:
                            threshold_count = st.number_input(
                                "Порог, кол-во",
                                min_value=1,
                                max_value=100,
                                value=int(selected_rule["threshold_count"]),
                                key=f"rule_edit_count_{selected_rule['id']}",
                            )
                        with threshold_col3:
                            cooldown_seconds = st.number_input(
                                "Cooldown, сек",
                                min_value=0,
                                max_value=3600,
                                value=int(selected_rule["cooldown_seconds"]),
                                key=f"rule_edit_cooldown_{selected_rule['id']}",
                            )
                        severity = st.selectbox(
                            "Серьезность",
                            options=list(RULE_SEVERITY_OPTIONS.keys()),
                            index=list(RULE_SEVERITY_OPTIONS.keys()).index(selected_rule["severity"])
                            if selected_rule["severity"] in RULE_SEVERITY_OPTIONS
                            else 1,
                            format_func=lambda key: RULE_SEVERITY_OPTIONS[key],
                        )
                        description = st.text_area("Описание", value=selected_rule.get("description") or "")
                        save = st.form_submit_button("Сохранить изменения")
                    if save:
                        if not can_manage:
                            st.error("Недостаточно прав для обновления правила.")
                        else:
                            update_zone_rule_fn(
                                rule_id=selected_rule["id"],
                                zone_id=zone_options[zone_label]["id"],
                                rule_type=rule_type,
                                threshold_seconds=int(threshold_seconds),
                                threshold_count=int(threshold_count),
                                cooldown_seconds=int(cooldown_seconds),
                                severity=severity,
                                description=description,
                            )
                            st.success("Правило обновлено.")
                        st.rerun()
                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        if st.button("Активировать правило", key=f"activate_zone_rule_{selected_rule['id']}"):
                            if not can_manage:
                                st.error("Недостаточно прав для изменения правила.")
                            else:
                                set_zone_rule_active_fn(rule_id=selected_rule["id"], is_active=True)
                                st.success("Правило активировано.")
                            st.rerun()
                    with action_col2:
                        if st.button("Деактивировать правило", key=f"deactivate_zone_rule_{selected_rule['id']}"):
                            if not can_manage:
                                st.error("Недостаточно прав для изменения правила.")
                            else:
                                set_zone_rule_active_fn(rule_id=selected_rule["id"], is_active=False)
                                st.warning("Правило деактивировано.")
                            st.rerun()


def _render_zone_preview_and_editor(st, *, source_labels, statuses_by_source, zones, can_manage: bool, update_zone_fn, set_zone_active_fn):
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
                    if not can_manage:
                        st.error("Недостаточно прав для обновления зоны.")
                    else:
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
                        if not can_manage:
                            st.error("Недостаточно прав для изменения зоны.")
                        else:
                            set_zone_active_fn(zone_id=selected_zone["id"], is_active=True)
                            st.success("Зона активирована.")
                        st.rerun()
                with action_col2:
                    if st.button("Деактивировать зону", key=f"deactivate_zone_{selected_zone['id']}"):
                        if not can_manage:
                            st.error("Недостаточно прав для изменения зоны.")
                        else:
                            set_zone_active_fn(zone_id=selected_zone["id"], is_active=False)
                            st.warning("Зона деактивирована.")
                        st.rerun()


def _fmt_ts(timestamp_value):
    if not timestamp_value:
        return "—"
    return datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
