# Result Examples

Example outputs from lm-evaluation-harness runs.

## Single model result

```json
{
  "results": {
    "mmlu": {
      "acc": 0.459,
      "acc_stderr": 0.004
    },
    "gsm8k": {
      "exact_match": 0.142,
      "exact_match_stderr": 0.006
    },
    "hellaswag": {
      "acc_norm": 0.765,
      "acc_norm_stderr": 0.004
    }
  },
  "config": {
    "model": "hf",
    "model_args": "pretrained=meta-llama/Llama-2-7b-hf",
    "num_fewshot": 5
  }
}
```

## Multi-model comparison table

| Model                  | MMLU  | GSM8K | HELLASWAG | TRUTHFULQA |
|------------------------|-------|-------|-----------|------------|
| meta-llama/Llama-2-7b  | 0.459 | 0.142 | 0.765     | 0.391      |
| meta-llama/Llama-2-13b | 0.549 | 0.287 | 0.801     | 0.430      |
| mistralai/Mistral-7B   | 0.626 | 0.395 | 0.812     | 0.428      |
| microsoft/phi-2        | 0.560 | 0.613 | 0.682     | 0.447      |

## Training progress data

```
| Step | MMLU  |
|------|-------|
| 1000 | 0.312 |
| 2000 | 0.389 |
| 3000 | 0.421 |
| 4000 | 0.445 |
| 5000 | 0.459 |
```

## Available task snippets

Use `scripts/parse_results.py` to programmatically extract scores:

```python
from parse_results import load_results, extract_all_scores

data = load_results("results/llama2-7b-eval.json")
scores = extract_all_scores(data)
print(scores)  # {"mmlu": 0.459, "gsm8k": 0.142, ...}
```
