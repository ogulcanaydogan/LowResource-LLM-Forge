# Evaluation Methodology

## Benchmarks

### Turkish MMLU

Multiple-choice academic benchmark translated to Turkish. Tests broad knowledge across STEM, humanities, and social sciences.

- **Library:** lm-evaluation-harness (`turkishmmlu` task)
- **Metric:** Accuracy (0.0 - 1.0)
- **Pass threshold:** ≥0.40 (baseline for 7B models)

### Perplexity

Language modeling quality on held-out Turkish text. Lower is better.

- **Method:** Cross-entropy loss on `data/processed/turkish_eval.jsonl`
- **Metric:** Perplexity (exp of average loss)
- **Pass threshold:** <50.0

### Generation Quality

Heuristic scoring on Turkish text generation from 10 diverse prompts.

- **Criteria:** Response length, repetition avoidance, Turkish character usage
- **Metric:** Average score (0.0 - 5.0)
- **Pass threshold:** ≥3.5

## Running Evaluations

```bash
# Default benchmarks (perplexity + generation)
uv run python scripts/run_eval.py --model <model-path>

# Single benchmark
uv run python scripts/run_eval.py --model <model-path> --benchmark perplexity

# Results saved to artifacts/eval/report.md and artifacts/eval/results.json
```

### Optional: Turkish MMLU

`mmlu_tr` requires `lm-evaluation-harness` (`lm_eval`) installed in the evaluation environment.

## Adding Custom Benchmarks

1. Create a new class in `src/forge/evaluation/benchmarks/`
2. Implement a `run()` method returning `dict[str, Any]`
3. Register in `ForgeEvaluator._run_<name>()` method
4. Add to default benchmarks list in `EvalConfig`
