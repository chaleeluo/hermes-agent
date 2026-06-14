---
name: evaluating-llms-harness
description: "Benchmark LLMs on MMLU, GSM8K, HumanEval and 60+ tasks."
version: 1.0.1
author: Orchestra Research
license: MIT
dependencies: [lm-eval, transformers, vllm]
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Evaluation, LM Evaluation Harness, Benchmarking, MMLU, HumanEval, GSM8K, EleutherAI, Model Quality, Academic Benchmarks, Industry Standard]

---

# lm-evaluation-harness - LLM Benchmarking

Evaluates LLMs across 60+ academic benchmarks (MMLU, HumanEval, GSM8K, TruthfulQA, HellaSwag) using standardized prompts. Industry standard used by EleutherAI, HuggingFace, and major labs.

## Quick start

```bash
pip install lm-eval

# Evaluate any HuggingFace model on core benchmarks
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0 \
  --batch_size auto

# List all available tasks
lm_eval --tasks list
```

## Common workflows

### Workflow 1: Standard benchmark evaluation

```
- [ ] Choose benchmark suite (core: mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge)
- [ ] Configure model (HF, vLLM, API, quantized, custom checkpoint)
- [ ] Run: `lm_eval --model hf --model_args pretrained=<model> --tasks mmlu --num_fewshot 5 --output_path results/ --log_samples`
- [ ] Analyze: `python scripts/compare_results.py results/*.json`
```

**Benchmark reference** (see `references/benchmark-guide.md` for full details):
- Core reasoning: MMLU (57 subjects), GSM8K (math), HellaSwag (commonsense), TruthfulQA (factuality), ARC (science)
- Code: HumanEval (164 Python problems), MBPP
- Recommended suite: `--tasks mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge`

**Model backends:**
- HuggingFace: `--model hf --model_args pretrained=<name>,dtype=bfloat16`
- Quantized: `--model hf --model_args pretrained=<name>,load_in_4bit=True`
- vLLM (5-10x faster): `--model vllm --model_args pretrained=<name>,tensor_parallel_size=1,gpu_memory_utilization=0.8`
- Custom checkpoint: `--model hf --model_args pretrained=/path/to/checkpoint,tokenizer=/path/to/tokenizer`

### Workflow 2: Track training progress

```
- [ ] Evaluate every N steps on fast benchmarks (HellaSwag ~10min, GSM8K ~5min, PIQA ~2min)
- [ ] Avoid slow benchmarks for frequent eval (MMLU ~2h, HumanEval)
- [ ] Use `scripts/plot_learning_curve.py results/step-*.json --task mmlu` to generate progress table
```

### Workflow 3: Compare multiple models

```
- [ ] Define model list in a text file (one HF name per line)
- [ ] Run: `for m in $(cat models.txt); do lm_eval --model hf --model_args pretrained=$m --tasks mmlu,gsm8k --output_path results/$(echo $m | tr / -).json; done`
- [ ] Compare: `python scripts/compare_results.py results/*.json` (markdown table) or `--json` (structured)
```

### Workflow 4: Evaluate API models

OpenAI, Anthropic, local OpenAI-compatible APIs (vLLM, TGI, Ollama). See `references/api-evaluation.md`.

```bash
# OpenAI chat models (generation tasks only, no logprobs)
lm_eval --model openai-chat-completions --model_args model=gpt-4-turbo --tasks mmlu --num_fewshot 5

# Anthropic Claude 3
lm_eval --model anthropic-chat --model_args model=claude-3-5-sonnet-20241022 --tasks mmlu

# Local vLLM server
lm_eval --model local-completions --model_args model=meta-llama/Llama-2-7b-hf,base_url=http://localhost:8000/v1
```

## Shipped scripts

| Script | Purpose |
|--------|---------|
| `scripts/compare_results.py` | Compare multi-model results (markdown table or JSON) |
| `scripts/plot_learning_curve.py` | Extract and display training progress from step-*.json files |
| `scripts/parse_results.py` | Python library for loading and extracting scores from lm-eval JSON results |

## When to use vs alternatives

**Use lm-evaluation-harness when:** benchmarking for papers, comparing models on standard tasks, tracking training progress, reporting standardized metrics, needing reproducible evaluation.

**Alternatives:** HELM (Stanford, broader evaluation), AlpacaEval (instruction-following with LLM judges), MT-Bench (multi-turn conversation), custom scripts (domain-specific).

## Common issues

**Issue: Evaluation too slow**
- Use vLLM backend: `--model vllm --model_args pretrained=...,tensor_parallel_size=2`
- Reduce fewshot: `--num_fewshot 0` instead of 5
- Evaluate subset: `--tasks mmlu_stem` (only STEM subjects)

**Issue: Out of memory**
- Reduce batch size: `--batch_size 1` or `--batch_size auto`
- Quantization: `--model_args pretrained=...,load_in_8bit=True`
- CPU offloading: `--model_args pretrained=...,device_map=auto,offload_folder=offload`

**Issue: Different results than reported**
- Verify fewshot count: `--num_fewshot 5` (most papers use 5-shot)
- Check exact task name: `mmlu` not `mmlu_direct` or `mmlu_fewshot`
- Verify model and tokenizer match: `--model_args pretrained=<name>,tokenizer=<same-name>`

**Issue: HumanEval not executing code**
```bash
pip install human-eval
lm_eval --model hf --model_args pretrained=<model> --tasks humaneval --allow_code_execution
```

**Issue: API authentication**
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
# Verify: echo $OPENAI_API_KEY
```

## Distributed evaluation

See `references/distributed-eval.md` for full guide. Quick reference:

- **HF data parallel:** `accelerate launch --multi_gpu --num_processes 8 -m lm_eval --model hf ...`
- **HF tensor parallel (large models):** `--model_args pretrained=...,parallelize=True`
- **vLLM tensor parallel:** `--model vllm --model_args pretrained=...,tensor_parallel_size=4`
- **vLLM data + tensor combined:** `--model_args pretrained=...,tensor_parallel_size=4,data_parallel_size=2`

## Hardware requirements

- **GPU:** NVIDIA (CUDA 11.8+), CPU works but very slow
- **VRAM:** 7B=16GB (bf16)/8GB (8-bit), 13B=28GB (bf16)/14GB (8-bit), 70B=multi-GPU or quantization
- **Time** (7B on single A100): HellaSwag 10min, GSM8K 5min, MMLU 2h, HumanEval 20min

## Advanced topics

- **Custom tasks:** See `references/custom-tasks.md` for creating domain-specific YAML+Python tasks
- **API evaluation:** See `references/api-evaluation.md` for OpenAI, Anthropic, local endpoints
- **Multi-GPU:** See `references/distributed-eval.md` for data parallel, tensor parallel, SLURM
- **Benchmark guide:** See `references/benchmark-guide.md` for all 60+ tasks and interpretation
- **Result examples:** See `references/result-examples.md` for JSON output and comparison table examples

## Resources

- GitHub: https://github.com/EleutherAI/lm-evaluation-harness
- Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
