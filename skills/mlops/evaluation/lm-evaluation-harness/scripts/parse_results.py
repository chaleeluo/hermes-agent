from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TASK_METRIC_MAP: dict[str, str] = {
    "mmlu": "acc",
    "gsm8k": "exact_match",
    "hellaswag": "acc_norm",
    "truthfulqa": "acc",
    "truthfulqa_mc2": "acc",
    "arc_challenge": "acc",
    "arc_easy": "acc",
    "humaneval": "pass@1",
    "mbpp": "pass@1",
    "winogrande": "acc",
    "piqa": "acc_norm",
    "bbh": "acc",
    "ifeval": "inst_level_strict_acc",
    "lambada_openai": "perplexity",
}


def load_results(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def extract_score(results: dict[str, Any], task: str) -> float | None:
    task_results = results.get("results", {}).get(task)
    if task_results is None:
        return None
    metric = TASK_METRIC_MAP.get(task, "acc")
    val = task_results.get(metric)
    if val is not None:
        return float(val)
    for v in task_results.values():
        try:
            return float(v)
        except (ValueError, TypeError):
            continue
    return None


def extract_all_scores(results: dict[str, Any]) -> dict[str, float]:
    task_results = results.get("results", {})
    out: dict[str, float] = {}
    for task in task_results:
        score = extract_score(results, task)
        if score is not None:
            out[task] = score
    return out


def extract_config(results: dict[str, Any]) -> dict[str, Any]:
    config = results.get("config", {})
    return {
        "model": config.get("model", "unknown"),
        "model_args": config.get("model_args", ""),
        "num_fewshot": config.get("num_fewshot", 0),
        "batch_size": config.get("batch_size", "auto"),
        "device": config.get("device", ""),
    }


def format_metric_name(task: str) -> str:
    return TASK_METRIC_MAP.get(task, "acc")


def make_comparison_table(results_list: list[dict[str, Any]], model_names: list[str]) -> list[dict[str, Any]]:
    all_tasks: list[str] = []
    for r in results_list:
        all_tasks.extend(extract_all_scores(r).keys())
    all_tasks = sorted(dict.fromkeys(all_tasks))

    rows: list[dict[str, Any]] = []
    for model_name, result in zip(model_names, results_list):
        scores = extract_all_scores(result)
        row: dict[str, Any] = {"model": model_name}
        for task in all_tasks:
            metric = format_metric_name(task)
            score = scores.get(task)
            row[f"{task}_{metric}"] = score
        rows.append(row)
    return rows
