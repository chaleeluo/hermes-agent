#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from parse_results import extract_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot training progress from lm-eval checkpoint results")
    parser.add_argument("files", nargs="+", type=Path, help="Paths to step-*.json result files (sorted by step)")
    parser.add_argument("--task", default="mmlu", help="Task to plot (default: mmlu)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of markdown table")
    args = parser.parse_args(argv)

    points = []
    for f in sorted(args.files):
        try:
            step = int(f.stem.split("-")[1])
        except (IndexError, ValueError):
            print(f"Warning: cannot extract step from {f.name}, skipping", file=sys.stderr)
            continue
        with open(f) as fp:
            data = json.load(fp)
        score = extract_score(data, args.task)
        if score is not None:
            points.append({"step": step, "score": score})

    if not points:
        print(f"No data points found for task '{args.task}'", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(points, indent=2))
    else:
        print(f"| Step | {args.task.upper()} |")
        print("|------|--------|")
        for p in points:
            print(f"| {p['step']} | {p['score']:.3f} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
