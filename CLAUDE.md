# LowResource-LLM-Forge

## CI/CD Politikasi
**TUM AGENTLAR icin zorunlu okuma:** `~/Projects/ci-cd-policy.md`
Ozet: <=5 acik PR, main yesil olmadan PR acma, haftalik stale PR temizligi, pre-commit hook calistir.
Not: Bu repo Jenkins'e gecmedi; `ci.yml` korunuyor, `eval-gate.yml` ve `vm330-ssh-bootstrap.yml` disabled.

## Repo Amaci
Dusuk kaynak Turkce LLM arastirmasi. HuggingFace: ogulcanaydogan/turkish-llm-family.

## Stack
- Python, uv (uv.lock mevcut)
- `src/` — model kodu
- `notebooks/` — arastirma
- `scripts/` — util scriptler
- `configs/` — model konfigurasyonlari
- `tests/` — pytest

## Komutlar
```bash
uv run pytest tests/
uv run ruff check src/
uv run ruff format src/
```

## Kurallar
- Pre-commit hook zorunlu (black + ruff)
- `ci.yml` GHA workflow aktif kalıyor — Jenkins yoktur bu repoda
- `eval-gate.yml`, `vm330-ssh-bootstrap.yml` disable edildi
