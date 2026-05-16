# Roadmap

## Current: pre-release (main branch, 78 tests passing)

End-to-end QLoRA fine-tuning pipeline for low-resource languages. Covers data collection, Whisper transcription, training (QLoRA/PEFT), evaluation benchmarks, vLLM serving, and HuggingFace Hub publishing. Turkish and Azerbaijani are active language configs.

---

## v0.1.0 — First Stable Release (May 2026)

- [ ] Tag v0.1.0 — pipeline is functional, CI passes, mypy strict clean
- [ ] Turkish (primary) and Azerbaijani configs verified end-to-end
- [ ] Published HuggingFace leaderboard entry for Turkish base model
- [ ] `forge` CLI reference in README with copy-paste quick-start

**Status**: ready to cut — no blocking code work remaining.

---

## v0.2.0 — Language Expansion (Q2 2026)

- [ ] Welsh QLoRA config (small corpus, CommonVoice-backed)
- [ ] Basque QLoRA config (Basque Center for Language Technology datasets)
- [ ] Catalan QLoRA config (Common Crawl filtered + Catalan Wikipedia)
- [ ] Test coverage: 47% → 60% (currently 40% threshold, target 60%)
- [ ] Evaluation benchmark: BLEU + chrF per language, published to HF dataset card

---

## v0.3.0 — Multi-Model Benchmark (Q3 2026)

- [ ] Benchmark matrix: three supported base models × three languages
- [ ] vLLM production serving guide with GPU memory profiles
- [ ] Replicate deployment with public demo endpoint
- [ ] Curriculum learning scheduler for highly imbalanced low-resource corpora

---

## Known issues / backlog

See [open issues](https://github.com/ogulcanaydogan/LowResource-LLM-Forge/issues).
