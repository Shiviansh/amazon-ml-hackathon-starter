# Hackathon Starter: first-hour quickstart

This repository is an experimental toolkit, not a proven competition-winning solution. Begin by checking the official rules, the real CSV headers, sample-submission schema, metric, and hardware. Keep original files unchanged and out of Git. Treat all validation results as estimates, not leaderboard guarantees.

## 1. Set up

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Inspect your actual data first

Put the official files under `data/raw/` and inspect column names, row counts, missingness, ID uniqueness, target values, and sample-submission columns. Do not assume this year's files match an older edition. The built-in `amazon` pipeline profile uses canonical columns `PRODUCT_ID`, `PRICE`, `TITLE`, `DESCRIPTION`, `PACK_SIZE`, `CATEGORY`, `BRAND`, and (unless images are skipped) `IMAGE_URL`.

### If the files use the 2025-style schema

For input files with `sample_id`, `catalog_content`, `image_link`, and `price`, first make canonical copies with the adapter. The raw files remain untouched:

```powershell
python src/prepare_challenge_data.py `
  --train data/raw/train.csv `
  --test data/raw/test.csv `
  --sample data/raw/sample_submission.csv `
  --output-dir data/processed/amazon_2025
```

`catalog_content` is preserved as a whole in `TITLE`; the adapter does not claim to split it into true title, description, brand, category, or pack-size fields. Empty placeholder fields satisfy the current profile. Inspect examples and turn on domain extraction only after checking the pipeline options. If output files already exist, the adapter refuses to overwrite them unless `--overwrite` is passed.

For other schemas, write or configure an explicit adapter based on the actual rules and headers; do not force data into these assumptions.

## 3. Run a simple baseline

Inspect the available CLI and stages:

```powershell
python src/pipeline.py --help
python src/pipeline.py --dataset amazon --list-stages
```

Then use the untouched raw paths, or the canonical adapter outputs above:

```powershell
python src/pipeline.py --dataset amazon --device cpu --skip-images `
  --extract-domain-features `
  --train data/processed/amazon_2025/train.csv `
  --test data/processed/amazon_2025/test.csv `
  --sample data/processed/amazon_2025/sample_submission.csv `
  --run-dir runs/first
```

Start with CPU and a small, reproducible run. Only add GPU training, embeddings, images, OCR, tuning, calibration, or ensembling after confirming the baseline, backend support, available memory, and runtime. Read `--help` for each component; flags can differ.

## 4. First-hour checklist

1. Read the official task/rules and record the metric, target, allowed data, resource cap, and submission format.
2. Inspect train/test/sample schemas, ID order, duplicates, target distribution, missingness, and catalog examples.
3. Decide the split using the data-generating process; group near-duplicates/entities when needed.
4. Run the adapter only if its input schema exactly matches. Otherwise map the schema deliberately.
5. Establish one CPU baseline and save its config, fold IDs, OOF predictions, metric, and runtime.
6. Change one major component at a time and retain it only if it improves the same leakage-safe validation design.

## 5. Keep evaluation honest

- Never use test labels for model selection. Never treat leaderboard feedback as a validation set.
- Fit learned preprocessing, target encoders, feature selection, calibration, and blend weights within the appropriate training/OOF boundaries.
- Compare models on identical folds and report pooled OOF plus fold metrics using the official metric convention.
- Check saved predictions, ID alignment, and submission schema with `src/validate_submit.py` before upload.
- Custom objectives in `src/models.py` are SMAPE/MAPE-inspired surrogates; their Hessian-like output is a positive curvature heuristic, not an exact second derivative. Evaluate them rather than assuming they help.

The historical notes in [reports/AMAZON_ML_CHALLENGE_PODIUM_PLAYBOOK.md](reports/AMAZON_ML_CHALLENGE_PODIUM_PLAYBOOK.md) distinguish cited team-authored reports from general experiment ideas. They are not an official leaderboard record or a prediction of this year's task.
