#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parse_results import extract_all_scores, extract_config, load_results


def build_table(results_files: list[Path]) -> str:
    results = [load_results(f) for f in results_files]
    all_tasks: list[str] = []
    for r in results:
        all_tasks.extend(extract_all_scores(r).keys())
    all_tasks = sorted(dict.fromkeys(all_tasks))

    model_names = []
    for f, r in zip(results_files, results):
        cfg = extract_config(r)
        model_names.append(cfg["model_args"].split(",")[0].replace("pretrained=", ""))

    lines: list[str] = []
    header = "| Model | " + " | ".join(t.upper() for t in all_tasks) + " |"
    sep = "|" + "|".join("---" for _ in range(len(all_tasks) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for name, r in zip(model_names, results):
        scores = extract_all_scores(r)
        vals = " | ".join(f"{scores.get(t, 0.0):.3f}" for t in all_tasks)
        lines.append(f"| {name} | {vals} |")

    return "\n".join(lines)


def build_json_output(results_files: list[Path]) -> str:
    results = [load_results(f) for f in results_files]
    entries = []
    for f, r in zip(results_files, results):
        cfg = extract_config(r)
        scores = extract_all_scores(r)
        entries.append({
            "file": str(f),
            "config": cfg,
            "scores": scores,
        })
    return json.dumps(entries, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare lm-eval results across models")
    parser.add_argument("files", nargs="+", type=Path, help="Paths to lm_eval result JSON files")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of markdown")
    args = parser.parse_args(argv)

    for f in args.files:
        if not f.exists():
            print(f"Error: file not found: {f}", file=sys.stderr)
            return 1

    if args.json:
        print(build_json_output(args.files))
    else:
        print(build_table(args.files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
