"""Initialize the runtime database and optionally seed demo data for empty environments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.repository import get_db_conn, init_db, reset_and_seed_demo_data


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def should_seed_demo_data() -> bool:
    conn = get_db_conn()
    try:
        row = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM employees) AS employees_count,
                (SELECT COUNT(*) FROM video_sources) AS video_sources_count,
                (SELECT COUNT(*) FROM events) AS events_count
            """
        ).fetchone()
        if row is None:
            return True
        return all(int(row[key] or 0) == 0 for key in row.keys())
    finally:
        conn.close()


def main() -> None:
    init_db()
    if not env_flag("BOOTSTRAP_DEMO_DATA", default=False):
        print("bootstrap: demo seed disabled")
        return
    if not should_seed_demo_data():
        print("bootstrap: existing data found, seed skipped")
        return
    employee_count = int(os.getenv("DEMO_SEED_EMPLOYEES", "120"))
    visit_count = int(os.getenv("DEMO_SEED_VISITS", "900"))
    seed_value = int(os.getenv("DEMO_SEED_VALUE", "42"))
    result = reset_and_seed_demo_data(
        employee_count=employee_count,
        visit_count=visit_count,
        seed=seed_value,
    )
    print(f"bootstrap: seeded demo data {result}")


if __name__ == "__main__":
    main()
