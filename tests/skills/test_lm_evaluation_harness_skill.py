from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "mlops" / "evaluation" / "lm-evaluation-harness"


def _load_script(name: str):
    path = SKILL_DIR / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


SAMPLE_RESULTS = {
    "results": {
        "mmlu": {"acc": 0.459, "acc_stderr": 0.004},
        "gsm8k": {"exact_match": 0.142, "exact_match_stderr": 0.006},
        "hellaswag": {"acc_norm": 0.765, "acc_norm_stderr": 0.004},
        "truthfulqa_mc2": {"acc": 0.391},
        "humaneval": {"pass@1": 0.288},
    },
    "config": {
        "model": "hf",
        "model_args": "pretrained=meta-llama/Llama-2-7b-hf",
        "num_fewshot": 5,
        "batch_size": "auto",
        "device": "cuda:0",
    },
}


# ── SKILL.md frontmatter ──────────────────────────────────────────────


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir()


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "evaluating-llms-harness"


def test_platforms_appropriate(frontmatter) -> None:
    platforms = set(frontmatter["platforms"])
    assert platforms.issubset({"linux", "macos"}), f"unexpected platforms: {platforms}"


def test_author_present(frontmatter) -> None:
    assert frontmatter["author"] == "Orchestra Research"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_dependencies_listed(frontmatter) -> None:
    deps = frontmatter.get("dependencies", [])
    assert "lm-eval" in deps
    assert "transformers" in deps


def test_metadata_tags_present(frontmatter) -> None:
    tags = frontmatter.get("metadata", {}).get("hermes", {}).get("tags", [])
    assert "Evaluation" in tags
    assert "MMLU" in tags


# ── SKILL.md structure ────────────────────────────────────────────────


def test_skill_md_under_250_lines() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    line_count = len(src.splitlines())
    assert line_count <= 250, f"SKILL.md is {line_count} lines, target ≤200"


def test_skill_md_has_pitfalls_section() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "## Common issues" in src


def test_skill_md_has_quick_start() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "## Quick start" in src


def test_skill_md_has_hardware_requirements() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "## Hardware requirements" in src


def test_skill_md_has_scripts_reference() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "## Shipped scripts" in src


# ── Shipped scripts syntax ────────────────────────────────────────────


@pytest.mark.parametrize(
    "script_name",
    [
        "scripts/parse_results.py",
        "scripts/compare_results.py",
        "scripts/plot_learning_curve.py",
    ],
)
def test_shipped_scripts_parse(script_name: str) -> None:
    src = (SKILL_DIR / script_name).read_text()
    ast.parse(src)


# ── parse_results.py core logic ───────────────────────────────────────


@pytest.fixture(scope="module")
def parse_results_mod():
    return _load_script("parse_results.py")


def test_load_results(tmp_path, parse_results_mod) -> None:
    p = tmp_path / "result.json"
    p.write_text(json.dumps(SAMPLE_RESULTS))
    loaded = parse_results_mod.load_results(p)
    assert loaded["results"]["mmlu"]["acc"] == 0.459


def test_extract_score_known_task(parse_results_mod) -> None:
    assert parse_results_mod.extract_score(SAMPLE_RESULTS, "mmlu") == 0.459
    assert parse_results_mod.extract_score(SAMPLE_RESULTS, "gsm8k") == 0.142
    assert parse_results_mod.extract_score(SAMPLE_RESULTS, "hellaswag") == 0.765
    assert parse_results_mod.extract_score(SAMPLE_RESULTS, "humaneval") == 0.288


def test_extract_score_unknown_task(parse_results_mod) -> None:
    assert parse_results_mod.extract_score(SAMPLE_RESULTS, "nonexistent") is None


def test_extract_all_scores(parse_results_mod) -> None:
    scores = parse_results_mod.extract_all_scores(SAMPLE_RESULTS)
    assert scores["mmlu"] == 0.459
    assert scores["gsm8k"] == 0.142
    assert scores["hellaswag"] == 0.765
    assert scores["humaneval"] == 0.288
    assert len(scores) == 5


def test_extract_config(parse_results_mod) -> None:
    cfg = parse_results_mod.extract_config(SAMPLE_RESULTS)
    assert cfg["model"] == "hf"
    assert cfg["num_fewshot"] == 5
    assert cfg["batch_size"] == "auto"


def test_format_metric_name(parse_results_mod) -> None:
    assert parse_results_mod.format_metric_name("mmlu") == "acc"
    assert parse_results_mod.format_metric_name("gsm8k") == "exact_match"
    assert parse_results_mod.format_metric_name("hellaswag") == "acc_norm"
    assert parse_results_mod.format_metric_name("nonexistent") == "acc"


def test_make_comparison_table(parse_results_mod) -> None:
    results2 = {
        "results": {
            "mmlu": {"acc": 0.549},
            "gsm8k": {"exact_match": 0.287},
            "hellaswag": {"acc_norm": 0.801},
        },
        "config": {"model": "hf", "model_args": "pretrained=meta-llama/Llama-2-13b-hf"},
    }

    rows = parse_results_mod.make_comparison_table(
        [SAMPLE_RESULTS, results2], ["Llama-2-7b", "Llama-2-13b"]
    )
    assert len(rows) == 2
    assert rows[0]["model"] == "Llama-2-7b"
    assert rows[0]["mmlu_acc"] == 0.459
    assert rows[1]["gsm8k_exact_match"] == 0.287


# ── compare_results.py CLI ────────────────────────────────────────────


def test_compare_results_markdown_output(tmp_path) -> None:
    p1 = tmp_path / "model_a.json"
    p2 = tmp_path / "model_b.json"
    p1.write_text(json.dumps(SAMPLE_RESULTS))
    p2.write_text(json.dumps({
        "results": {"mmlu": {"acc": 0.549}, "gsm8k": {"exact_match": 0.287}},
        "config": {"model": "hf", "model_args": "pretrained=meta-llama/Llama-2-13b-hf"},
    }))

    mod = _load_script("compare_results.py")
    exit_code = mod.main([str(p1), str(p2)])
    assert exit_code == 0


def test_compare_results_json_output(tmp_path, capsys) -> None:
    p = tmp_path / "model.json"
    p.write_text(json.dumps(SAMPLE_RESULTS))

    mod = _load_script("compare_results.py")
    exit_code = mod.main(["--json", str(p)])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 1
    assert out[0]["config"]["model"] == "hf"


def test_compare_results_missing_file() -> None:
    mod = _load_script("compare_results.py")
    exit_code = mod.main(["/nonexistent/results.json"])
    assert exit_code == 1


# ── plot_learning_curve.py CLI ────────────────────────────────────────


def test_plot_learning_curve_markdown(tmp_path, capsys) -> None:
    for step in [1000, 2000, 3000]:
        p = tmp_path / f"step-{step}.json"
        p.write_text(json.dumps({
            "results": {"mmlu": {"acc": step / 10000}},
        }))

    mod = _load_script("plot_learning_curve.py")
    exit_code = mod.main([str(p) for p in sorted(tmp_path.glob("step-*.json"))])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "| 1000 | 0.100" in out
    assert "| 3000 | 0.300" in out


def test_plot_learning_curve_json(tmp_path, capsys) -> None:
    p = tmp_path / "step-5000.json"
    p.write_text(json.dumps({
        "results": {"gsm8k": {"exact_match": 0.500}},
    }))

    mod = _load_script("plot_learning_curve.py")
    exit_code = mod.main(["--json", "--task", "gsm8k", str(p)])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["step"] == 5000
    assert out[0]["score"] == 0.5


def test_plot_learning_curve_no_data(tmp_path) -> None:
    p = tmp_path / "step-1000.json"
    p.write_text(json.dumps({"results": {}}))

    mod = _load_script("plot_learning_curve.py")
    exit_code = mod.main([str(p)])
    assert exit_code == 1


# ── Reference docs exist ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "ref_name",
    [
        "references/benchmark-guide.md",
        "references/api-evaluation.md",
        "references/custom-tasks.md",
        "references/distributed-eval.md",
        "references/result-examples.md",
        "references/evaluation-guide.md",
    ],
)
def test_reference_doc_exists(ref_name: str) -> None:
    assert (SKILL_DIR / ref_name).is_file()
