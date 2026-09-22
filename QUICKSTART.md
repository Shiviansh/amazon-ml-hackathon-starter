# Hackathon Starter: Quickstart

This folder contains an experimental starter toolkit for tabular, text, and multimodal product-catalog ML. Treat every score as a local validation estimate, not a guarantee of leaderboard performance. Start with the CPU/tabular path; image downloads and pretrained vision models are optional and can be slow.

For an in-depth breakdown of winning strategies from the last 5 years of the Amazon ML Challenge, see [reports/AMAZON_ML_CHALLENGE_PODIUM_PLAYBOOK.md](reports/AMAZON_ML_CHALLENGE_PODIUM_PLAYBOOK.md).

## 1. Set up Python

Use Python 3.10 or newer. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

Verify setup:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## 2. Put the competition files in place

The pipeline has a built-in `amazon` profile (defaults are defined in `src/pipeline.py`). Put the original files here (do not edit the supplied source files):

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

The profile expects `PRODUCT_ID` as the identifier, `PRICE` as the training target, and catalog text/category fields including `TITLE`, `DESCRIPTION`, `PACK_SIZE`, `CATEGORY`, and `BRAND`. Check the competition's actual schema against the config before running; rename/configure fields as needed rather than guessing. Keep all raw data out of Git.

## 3. Inspect the pipeline, then run a baseline

See the available stages without executing them:

```powershell
python src/pipeline.py --dataset amazon --list-stages
```

Run the tabular workflow, skipping optional image work:

```powershell
python src/pipeline.py --dataset amazon --device cpu --skip-images --run-dir runs/first
```

For custom file locations, pass `--train`, `--test`, and `--sample`:

```powershell
python src/pipeline.py --dataset amazon --device cpu --skip-images `
  --train data/raw/train.csv --test data/raw/test.csv `
  --sample data/raw/sample_submission.csv --run-dir runs/custom
```

## 4. Rank-1 Podium Features & Command Flags

### A. Custom Objective Gradients (SMAPE & MAPE)
Standard MSE/RMSE penalizes large values quadratically, which underperforms on percentage-based metrics.
The toolkit provides exact, smooth 1st and 2nd derivatives for LightGBM and XGBoost, plus native loss function dispatch for CatBoost:

```powershell
# Direct SMAPE-optimized LightGBM training:
python src/train.py --train data/raw/train.csv --test data/raw/test.csv `
  --target-column PRICE --id-column PRODUCT_ID `
  --model lightgbm --metric smape --objective smape `
  --output-dir runs/lgbm_smape

# Automatic metric-to-objective resolution (--objective auto detects smape/mape):
python src/pipeline.py --dataset amazon --objective auto --run-dir runs/auto_obj
```

### B. Deterministic Domain Extraction Engine
Extract explicit pack multipliers, units (g, ml, cm), power/voltage, and density proxies from raw catalog strings:

```powershell
# Direct feature extraction via standalone CLI:
python src/features.py --train data/raw/train.csv --test data/raw/test.csv `
  --output-train data/processed/features/train.parquet `
  --output-test data/processed/features/test.parquet

# On-the-fly extraction inside train.py:
python src/train.py --train data/raw/train.csv --test data/raw/test.csv `
  --target-column PRICE --id-column PRODUCT_ID `
  --text-columns TITLE DESCRIPTION --extract-domain-features `
  --model lightgbm --output-dir runs/lgbm_domain
```

### C. Upgraded SOTA Text Embeddings
Pre-configured for `BAAI/bge-large-en-v1.5` (1024-dim dense representation) with automatic fallback to `sentence-transformers/all-MiniLM-L6-v2` and offline TruncatedSVD:

```powershell
# Generate text embeddings:
python src/embeddings.py --train data/raw/train.csv --test data/raw/test.csv `
  --output-train data/processed/text_emb_train.parquet `
  --output-test data/processed/text_emb_test.parquet `
  --model-name BAAI/bge-large-en-v1.5 --device cuda
```

## 5. GPU and Images

Only request `--device gpu` after confirming that the chosen model backend and installed dependencies support your GPU.

Images are optional. The full image path downloads remote assets and requires network access, model weights, disk space, and extra packages. First establish a reproducible tabular baseline; then add image/OCR stages and compare them using the same leakage-safe validation folds.

## 6. Team Workflow and Leakage Checks

- Keep raw data, credentials, caches, trained models, and predictions out of Git; `.gitignore` excludes common generated paths.
- Agree on one split definition and metric before comparing experiments. Never use test labels or tune against leaderboard feedback as if it were validation.
- Fit learned preprocessing, encoders, feature selection, and calibration inside each training fold. Generate out-of-fold (OOF) predictions before blending or calibration.
- Record the command, config, seed, split IDs, and package versions for each experiment.
- Inspect `python src/<script>.py --help` before using a module directly; command-line interfaces can differ between modules and may evolve.
