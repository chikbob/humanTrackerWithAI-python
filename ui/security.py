"""Security access and audit views."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from services.auth import ROLE_OPTIONS


def render_security_audit(st, *, access_context: dict, audit_logs: list[dict]):
    st.subheader("Доступ и аудит")
    top1, top2, top3 = st.columns(3)
    top1.metric("Текущий оператор", access_context.get("actor_name") or "—")
    top2.metric("Роль", access_context.get("role_label") or ROLE_OPTIONS.get(access_context.get("role"), "—"))
    top3.metric("Записей аудита", len(audit_logs))

    with st.container(border=True):
        st.markdown("### Профиль доступа")
        permissions_rows = [{"Разрешение": permission} for permission in access_context.get("permissions", [])]
        st.dataframe(pd.DataFrame(permissions_rows), width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown("### Журнал аудита")
        if not audit_logs:
            st.dataframe(
                pd.DataFrame(columns=["Время", "Оператор", "Роль", "Действие", "Ресурс", "ID", "Детали"]),
                width="stretch",
                hide_index=True,
            )
            st.caption("Пока нет записей аудита.")
            return
        rows = [
            {
                "Время": datetime.fromtimestamp(row["created_at"]).strftime("%Y-%m-%d %H:%M:%S") if row.get("created_at") else "—",
                "Оператор": row.get("actor_name") or "—",
                "Роль": ROLE_OPTIONS.get(row.get("actor_role"), row.get("actor_role") or "—"),
                "Действие": row.get("action") or "—",
                "Ресурс": row.get("resource_type") or "—",
                "ID": row.get("resource_id") or "—",
                "Детали": ", ".join(f"{key}={value}" for key, value in (row.get("details") or {}).items()) or "—",
            }
            for row in audit_logs
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
