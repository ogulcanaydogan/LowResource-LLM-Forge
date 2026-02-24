# Project Roadmap

This roadmap starts after the current priority training run on A100 is completed and evaluated.

## Current Run Definition of Done

Before moving to improvement work:

1. Complete the active training run (`target_steps=25845`) or end by a valid early-stop condition.
2. Merge adapter into base model and produce a merged checkpoint.
3. Run full evaluation (`perplexity`, `generation`, optional `mmlu_tr`) and save report artifacts.
4. Publish a versioned release candidate with reproducible config references.

## Post-Completion Improvement Plan

### Phase 1: Stability Hardening (Priority P0)

Goal: prevent silent training failure and auto-recover quickly.

- Add NaN/Inf guard callbacks for `loss`, `grad_norm`, and `eval_loss`.
- Fail fast on unstable metrics and auto-resume from last healthy checkpoint.
- Keep `systemd --user` + watchdog as the default runtime path on remote hosts.
- Persist heartbeat and key metrics to machine-readable status files for monitoring.

Exit criteria:

- No silent NaN progression in new runs.
- Automatic recovery from interruption in under 10 minutes.
- Stable checkpoints produced on schedule.

### Phase 2: Turkish Data Expansion and Quality (Priority P0)

Goal: improve model quality using larger, cleaner, better-balanced Turkish corpora.

- Expand corpus with open Turkish sources (for example mC4, OSCAR, Wiki-derived text, curated Turkish instruction datasets).
- Improve deduplication and language filtering thresholds.
- Add quality scoring filters (length, script ratio, repetition, malformed text checks).
- Build a versioned dataset mixture and track it in a changelog.

Suggested starting mixture:

- 60% high-quality instruction data
- 25% domain text relevant to target use-cases
- 15% synthetic/translated augmentation with strict filtering

Exit criteria:

- At least 2x unique Turkish token coverage vs current baseline.
- Low-quality sample ratio below 5% after filtering.

### Phase 3: Training Recipe Optimization on A100 (Priority P0)

Goal: increase quality while preserving training stability.

- Run controlled sweeps for learning rate, warmup ratio, LoRA rank/alpha, and effective batch size.
- Keep bf16 enabled on A100 and tune gradient accumulation for throughput.
- Tune evaluation cadence (`eval_steps=1000`) and checkpoint cadence (`save_steps=1000`).
- Promote only runs with finite metrics and consistent convergence.

Exit criteria:

- Perplexity improves by at least 10% from baseline.
- Generation quality score improves by at least 0.4.
- No regression in safety/format adherence prompts.

### Phase 4: Inference Throughput and Latency (Priority P1)

Goal: approach high-quality serving UX (fast first token + fluent decode).

- Tune vLLM serving args (`max_num_batched_tokens`, `max_num_seqs`, `gpu_memory_utilization`, tensor parallelism).
- Benchmark p50/p95 latency and tokens/sec under concurrent load.
- Add configuration profiles for low-latency and high-throughput modes.
- Evaluate TensorRT-LLM/NIM path only after vLLM baseline is saturated.

Exit criteria:

- At least 30% tokens/sec gain at target concurrency.
- p95 time-to-first-token under defined SLO.

### Phase 5: Evaluation Depth and Release Governance (Priority P1)

Goal: make releases trustworthy and repeatable.

- Expand held-out Turkish eval set by domain.
- Add lightweight human review rubrics for fluency, factuality, and instruction-following.
- Track every release with dataset version, config hash, and benchmark deltas.
- Gate promotion on quality thresholds and regression checks.

Exit criteria:

- Every release has reproducible lineage.
- Promotion decisions are benchmark-backed and auditable.

## Immediate Next Actions After Current Run

1. Generate baseline report from the active A100 run.
2. Launch Phase 1 stability patch set before the next long training job.
3. Build `turkish-v2` dataset mixture and run a short smoke training cycle.
