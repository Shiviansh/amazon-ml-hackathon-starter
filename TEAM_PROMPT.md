# Prompt for Teammates and AI Assistants

Copy the prompt below into a new coding assistant task when you want help with this repository.

---

You are helping our team work on the Amazon ML Challenge. The checked-out repository is our current source of truth. Before proposing or changing anything, inspect `README.md`, `QUICKSTART.md`, the relevant CLI's `--help`, and the implementation/tests for the exact component. Do not assume claims in playbooks or comments are proven.

## Goals

- Follow the official challenge rules, target, metric, submission format, and available hardware. Ask for missing information only when an assumption would materially change the implementation; otherwise state a safe assumption.
- Establish a reliable local validation baseline before attempting complex objectives, GPU training, tuning, image models, or ensembling.
- Make reproducible, reviewable changes. Preserve existing user work and raw competition files.

## Leakage and evaluation rules

- Keep the competition test set completely out of model selection and calibration.
- Split according to the challenge's data-generating process and any official rules. Group related rows (for example, same product/entity) when row-wise splitting would leak near-duplicates across folds.
- Fit every learned transform using only the corresponding training fold. Target encoding, feature selection, imputation learned from data, calibration, and ensemble-weight fitting must respect fold boundaries.
- Use out-of-fold predictions for honest training-set evaluation and blending. Do not report a single in-sample score as validation performance.
- Report fold metrics and a pooled OOF metric using the official competition metric; note any split limitations and randomness.

## Repository workflow

The primary orchestrator is `src/pipeline.py`. Its current interface can be inspected with:

```powershell
python src/pipeline.py --help
```

For an initial CPU baseline with images disabled:

```powershell
python src/pipeline.py --dataset amazon --device cpu --skip-images --run-dir runs/first
```

Expected default files are `data/raw/train.csv`, `data/raw/test.csv`, and `data/raw/sample_submission.csv`. The built-in profile currently expects `PRODUCT_ID`, target `PRICE`, text columns `TITLE`, `DESCRIPTION`, `PACK_SIZE`, and categorical columns `CATEGORY`, `BRAND`. Check the real files and official task; change profile/code deliberately if they differ. Do not fabricate missing challenge data or assume these fields are guaranteed by this year's dataset.

For direct CV training, inspect `python src/train.py --help`. Current important options include `--train`, `--test`, `--target-column`, `--id-column`, `--model`, `--metric`, `--device`, `--fold-column`/`--n-splits`, and `--output-dir`. Available model and metric choices are defined by that CLI. Do not copy commands from older prompts without checking them.

## Toolkit components

- `src/audit.py`, `src/splits.py`: schema/data diagnostics and split generation.
- `src/features.py`: deterministic catalog text/spec extraction; validate parser outputs on examples from this dataset.
- `src/train.py`, `src/models.py`: fold-based model training. Custom objectives and GPU paths are implementation-specific; verify their actual behavior and installed backend before relying on them.
- `src/embeddings.py`, `src/download_images.py`, `src/ocr.py`, `src/image_embeddings.py`: optional text/image modalities that may require network access, model downloads, substantial disk/VRAM, and extra runtime.
- `src/ensemble.py`: fits blending weights using supplied OOF predictions; ensure those predictions are genuinely out-of-fold and fold-aligned.
- `src/postprocess.py`: optional calibration/clamping; fit only within training/OOF boundaries and compare against an untouched validation estimate.
- `src/validate_submit.py`: format and integrity checks; run it on the final CSV against the official sample submission.

## How to make a recommendation

1. State what you inspected and distinguish observed code behavior from hypotheses.
2. Identify the highest-impact next experiment and why it is worth its runtime.
3. Preserve a baseline and change one major factor at a time; record exact command, config, seed, folds, runtime, and OOF score.
4. Compare the change on identical leakage-safe folds. Keep it only if the improvement is stable and relevant to the official metric.
5. Before modifying files, explain any important tradeoff. After modifying, summarize changed files and verification performed; never claim tests passed unless they were actually run and passed.

Do not promise a leaderboard rank, accuracy/SMAPE improvement, or runtime without measured evidence. Numerical claims in historical playbooks, including podium rates, speed estimates, or “top X%” expectations, are leads to verify—not guarantees.

---
