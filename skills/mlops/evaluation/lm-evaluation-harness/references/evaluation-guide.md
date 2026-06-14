# Evaluation Guide

使用 mock 数据评测 lm-evaluation-harness 优化效果。

## 准备 mock 数据

创建两个模型的假结果文件：

```json
{
  "results": {
    "mmlu": {"acc": 0.459},
    "gsm8k": {"exact_match": 0.142},
    "hellaswag": {"acc_norm": 0.765}
  },
  "config": {
    "model": "hf",
    "model_args": "pretrained=meta-llama/Llama-2-7b-hf",
    "num_fewshot": 5
  }
}
```

```json
{
  "results": {
    "mmlu": {"acc": 0.549},
    "gsm8k": {"exact_match": 0.287},
    "hellaswag": {"acc_norm": 0.801}
  },
  "config": {
    "model": "hf",
    "model_args": "pretrained=meta-llama/Llama-2-13b-hf",
    "num_fewshot": 5
  }
}
```

## 评测 1：多模型对比（Markdown 表格）

```bash
python scripts/compare_results.py results/llama-7b.json results/llama-13b.json
```

预期输出：

```
| Model | MMLU | GSM8K | HELLASWAG |
|-------|------|-------|-----------|
| Llama-2-7b | 0.459 | 0.142 | 0.765 |
| Llama-2-13b | 0.549 | 0.287 | 0.801 |
```

## 评测 2：多模型对比（JSON 格式）

```bash
python scripts/compare_results.py --json results/llama-7b.json results/llama-13b.json
```

预期输出：

```json
[
  {
    "file": "results/llama-7b.json",
    "config": {
      "model": "hf",
      "model_args": "pretrained=meta-llama/Llama-2-7b-hf",
      "num_fewshot": 5
    },
    "scores": {
      "mmlu": 0.459,
      "gsm8k": 0.142,
      "hellaswag": 0.765
    }
  }
]
```

## 评测 3：训练进度追踪

准备多个 checkpoint 结果：

```json
{"results": {"mmlu": {"acc": 0.312}}}
{"results": {"mmlu": {"acc": 0.389}}}
{"results": {"mmlu": {"acc": 0.421}}}
```

```bash
python scripts/plot_learning_curve.py results/step-1000.json results/step-2000.json results/step-3000.json
```

预期输出：

```
| Step | MMLU  |
|------|-------|
| 1000 | 0.312 |
| 2000 | 0.389 |
| 3000 | 0.421 |
```

## 评测 4：Python API 直接调用

```python
import sys
sys.path.insert(0, "scripts")
from parse_results import load_results, extract_all_scores, make_comparison_table

r1 = load_results("results/llama-7b.json")
r2 = load_results("results/llama-13b.json")

scores = extract_all_scores(r1)
print("7B scores:", scores)
# {"mmlu": 0.459, "gsm8k": 0.142, "hellaswag": 0.765}

table = make_comparison_table([r1, r2], ["Llama-2-7B", "Llama-2-13B"])
print("table:", table)
# [{"model": "Llama-2-7B", "mmlu_acc": 0.459, ...}, ...]
```

## 评测 5：单项分数提取

```python
from parse_results import load_results, extract_score, extract_config

data = load_results("results/llama-7b.json")

# 提取单项分数
mmlu = extract_score(data, "mmlu")        # 0.459
gsm8k = extract_score(data, "gsm8k")      # 0.142

# 提取配置
cfg = extract_config(data)
# {"model": "hf", "num_fewshot": 5, "batch_size": "auto"}
```

## 边界值测试

```bash
# 文件不存在 → exit code 1
python scripts/compare_results.py /nonexistent/results.json
echo $?  # 1

# 无匹配数据 → exit code 1
python scripts/plot_learning_curve.py results/empty.json
echo $?  # 1

# 未知 task 名 → 返回 None
python -c "
import sys; sys.path.insert(0, 'scripts')
from parse_results import extract_score
data = {'results': {'mmlu': {'acc': 0.5}}}
print(extract_score(data, 'unknown_task'))  # None
"
```
