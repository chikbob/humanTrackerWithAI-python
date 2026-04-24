"""Export benchmark results into thesis-friendly tables."""

from __future__ import annotations

import csv
from pathlib import Path


EXPORT_COLUMNS = [
    "scenario_name",
    "model_name",
    "tracker_type",
    "frame_limit",
    "warmup_frames",
    "frames_processed",
    "avg_latency_ms",
    "p95_latency_ms",
    "avg_fps",
    "avg_detections_per_frame",
    "tracked_frame_ratio",
    "detection_count_total",
]


def build_markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "| Нет данных |\n| --- |\n"
    header = "| " + " | ".join(EXPORT_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(EXPORT_COLUMNS)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in EXPORT_COLUMNS) + " |")
    return "\n".join([header, separator, *body]) + "\n"


def export_results_csv(rows: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in EXPORT_COLUMNS})
    return path


def export_results_markdown(rows: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown_table(rows), encoding="utf-8")
    return path
