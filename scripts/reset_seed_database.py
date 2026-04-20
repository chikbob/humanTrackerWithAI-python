"""Utility to fully reset and seed the monitoring database with enterprise demo data."""

from __future__ import annotations

import argparse

from db.repository import init_db, reset_and_seed_demo_data


def parse_args():
    parser = argparse.ArgumentParser(description="Reset and seed the employee access monitoring database.")
    parser.add_argument("--employees", type=int, default=120, help="Number of employee records to generate.")
    parser.add_argument("--visits", type=int, default=900, help="Number of synthetic visit chains to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic data.")
    return parser.parse_args()


def main():
    args = parse_args()
    init_db()
    result = reset_and_seed_demo_data(employee_count=args.employees, visit_count=args.visits, seed=args.seed)
    print(result)


if __name__ == "__main__":
    main()
