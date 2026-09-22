# Amazon ML Hackathon Starter

A Python starter repository for building reproducible tabular and product-catalog ML experiments. It includes data checks, split utilities, feature extraction, cross-validated model training, optional image/OCR components, ensembling, calibration, and submission validation.

This is a toolkit, not a proven competition-winning solution. Metrics, GPU support, runtime, and leaderboard gains depend on the challenge data, metric, hardware, package versions, and validation design. Verify every assumption against the official rules and inspect the generated validation results.

## Start here

1. Read [QUICKSTART.md](QUICKSTART.md) for setup, expected inputs, and the supported pipeline entry point.
2. Read [TEAM_PROMPT.md](TEAM_PROMPT.md) before asking an LLM or teammate to modify or run the starter.
3. Treat [the historical notes](reports/AMAZON_ML_CHALLENGE_PODIUM_PLAYBOOK.md) as limited, sourced context—not an official record or prediction.

## Main workflow

```powershell
python src/pipeline.py --dataset amazon --device cpu --skip-images --run-dir runs/first
```

The pipeline uses the built-in Amazon profile defaults and expects raw CSV files under `data/raw/`. You can also provide `--train`, `--test`, and `--sample` paths. Start on CPU and add optional components only after you have a repeatable, leakage-safe validation baseline.

## What the core `src/` files do

| File | Competition role |
|---|---|
| `pipeline.py` | Runs configured stages in dependency order, caches validated outputs, and finishes with submission validation. |
| `prepare_challenge_data.py` | Strictly adapts the 2025-style `sample_id` / `catalog_content` / `image_link` / `price` CSVs to the canonical pipeline schema; it does not infer separate catalog fields. |
| `audit.py` | Profiles schemas, missingness, duplicates, identifiers, target values, and train/test drift; writes diagnostic reports. |
| `splits.py` | Builds deterministic K-fold, stratified, group-aware, stratified-group, or time-based folds. Choose grouping/time keys that match the challenge; no splitter can infer every leakage source automatically. |
| `features.py` | Extracts catalog attributes such as pack counts, quantities, dimensions, and text statistics; supports fold-aware target-encoding features when fold information is supplied. |
| `train.py` | Runs cross-validated training, generates out-of-fold and test predictions, and saves run metadata/artifacts. |
| `models.py` | Provides a shared interface for supported linear, LightGBM, CatBoost, and XGBoost estimators, with backend/device checks and fold-model serialization. Optional smooth objectives are implementation-specific approximations; verify suitability for the official metric. |
| `embeddings.py` | Produces optional dense text representations with configurable transformer backends, resumable caching, and a sparse-feature fallback path. The default model choice is not a claim of state-of-the-art performance. |
| `download_images.py` | Fetches image URLs concurrently with retry/cache handling and a manifest for downstream image processing. |
| `image_embeddings.py` | Produces cached product-image vectors using supported CLIP/DINOv2 backends and aligns outputs by product ID. |
| `ocr.py` | Runs available OCR backends on product images and extracts packaging text/quantity candidates as auxiliary features. OCR outputs require quality checks; they are not guaranteed ground truth. |
| `tune.py` | Runs Optuna hyperparameter trials against the configured fold-based validation objective. Search results are only as reliable as the supplied split and metric. |
| `ensemble.py` | Fits supported non-negative blending weights from aligned OOF predictions and applies them to corresponding test predictions. |
| `postprocess.py` | Optionally blends, calibrates prediction scale, and clamps to configured bounds using OOF/validation predictions. Keep calibration inside the validation design. |
| `validate_submit.py` | Checks submission columns, row counts, IDs/order, missing or non-finite values, and configured value constraints against the sample file. |
| `predict.py` | Loads saved inference artifacts and produces predictions for new rows using the stored preprocessing/model configuration. |
| `error_analysis.py` | Summarizes OOF errors and performance slices to help identify failure patterns and bias. |
| `metrics.py` | Implements common regression/classification metrics (including SMAPE, MAPE, MAE, RMSE, accuracy, and F1); confirm exact conventions and scaling against the official metric. |
| `__init__.py` | Marks `src` as an importable Python package and exposes package metadata/import behavior. |

The repository may also contain standalone data-preparation examples; they are not required by the competition pipeline and may download or derive example datasets. Inspect their sources and dataset terms before using them. `tests/` contains automated checks for the components.

## Important competition hygiene

- Keep supplied data immutable and out of Git. `.gitignore` excludes data and generated artifacts.
- Use only training data for fitting and model selection. Never fit encoders, feature selectors, calibrators, or ensemble weights using held-out validation labels.
- Match the official metric and split strategy. Use OOF predictions for model comparison and stacking; keep the competition test set untouched.
- Store each experiment's config, seed, split identifiers, code revision, and validation metrics.
- Do not infer GPU compatibility from CUDA-enabled PyTorch alone; probe the actual model backend.

## Team collaboration

Use branches for experiments and small, reviewable pull requests. Do not commit credentials, private datasets, cache files, model weights, or submissions. This repository is public: anyone can view, clone, and fork it. Add GitHub collaborators only when teammates need direct write access.
