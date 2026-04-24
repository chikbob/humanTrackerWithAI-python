"""Run thesis benchmark scenarios for detector/tracker comparison."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db.repository import init_db
from research.benchmark import run_benchmark_scenario
from research.export import export_results_csv, export_results_markdown


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark scenarios for detector/tracker comparison.")
    parser.add_argument("--scenario", default="detector_comparison", choices=["detector_comparison", "tracker_comparison", "latency_profile"])
    parser.add_argument("--source", required=True, help="Path to benchmark video file.")
    parser.add_argument("--model", default="yolov8s.pt", help="Model override for tracker/latency scenarios.")
    parser.add_argument("--frame-limit", type=int, default=120)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--notes", default="", help="Optional notes for experiment run.")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "docs", "research", "benchmarks"))
    return parser.parse_args()


def main():
    args = parse_args()
    init_db()
    result = run_benchmark_scenario(
        scenario_name=args.scenario,
        source_path=args.source,
        model_name=args.model,
        frame_limit=args.frame_limit,
        warmup_frames=args.warmup_frames,
        notes=args.notes,
    )
    out_dir = Path(args.out_dir)
    csv_path = export_results_csv(result["results"], out_dir / f"{result['run_key']}.csv")
    md_path = export_results_markdown(result["results"], out_dir / f"{result['run_key']}.md")
    print(
        {
            "run_id": result["run_id"],
            "run_key": result["run_key"],
            "csv": str(csv_path),
            "markdown": str(md_path),
            "result_count": len(result["results"]),
        }
    )


if __name__ == "__main__":
    main()
