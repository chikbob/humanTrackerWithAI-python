"""Entrypoint for the background monitoring worker."""

from __future__ import annotations

import argparse

from video.worker import main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Background worker for 24/7 enterprise entry-zone monitoring.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute one polling cycle and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(run_once=args.once)
