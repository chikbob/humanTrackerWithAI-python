"""Export stored benchmark results into thesis-friendly tables."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db.repository import load_benchmark_results, load_experiment_runs
from research.export import export_results_csv, export_results_markdown


def parse_args():
    parser = argparse.ArgumentParser(description="Export stored benchmark results.")
    parser.add_argument("--run-id", type=int, default=None, help="Specific experiment run ID. Defaults to latest run.")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "docs", "research", "benchmarks"))
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id
    if run_id is None:
        runs = load_experiment_runs(limit=1)
        if not runs:
            raise SystemExit("No experiment runs found.")
        run_id = runs[0]["id"]
    rows = load_benchmark_results(run_id=run_id)
    if not rows:
        raise SystemExit(f"No benchmark results found for run_id={run_id}.")
    run_key = rows[0]["run_key"]
    out_dir = Path(args.out_dir)
    csv_path = export_results_csv(rows, out_dir / f"{run_key}.csv")
    md_path = export_results_markdown(rows, out_dir / f"{run_key}.md")
    print({"run_id": run_id, "csv": str(csv_path), "markdown": str(md_path), "rows": len(rows)})


if __name__ == "__main__":
    main()
