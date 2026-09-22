# Amazon ML Hackathon Starter

A Python starter repository for building reproducible tabular and product-catalog ML experiments. It includes data checks, split utilities, feature extraction, cross-validated model training, optional image/OCR components, ensembling, calibration, and submission validation.

This is a toolkit, not a proven competition-winning solution. Metrics, GPU support, runtime, and leaderboard gains depend on the challenge data, metric, hardware, package versions, and validation design. Verify every assumption against the official rules and inspect the generated validation results.

## Start here

1. Read [QUICKSTART.md](QUICKSTART.md) for setup, expected inputs, and the supported pipeline entry point.
2. Read [TEAM_PROMPT.md](TEAM_PROMPT.md) before asking an LLM or teammate to modify or run the starter.
3. Use [reports/AMAZON_ML_CHALLENGE_PODIUM_PLAYBOOK.md](reports/AMAZON_ML_CHALLENGE_PODIUM_PLAYBOOK.md) as background hypotheses only; independently verify its claims and sources before relying on them.

## Main workflow

```powershell
python src/pipeline.py --dataset amazon --device cpu --skip-images --run-dir runs/first
```

The pipeline uses the built-in Amazon profile defaults and expects raw CSV files under `data/raw/`. You can also provide `--train`, `--test`, and `--sample` paths. Start on CPU and add optional components only after you have a repeatable, leakage-safe validation baseline.

## Repository map

- `src/audit.py`, `src/splits.py`: data checks and split helpers.
- `src/features.py`, `src/embeddings.py`: structured/text features.
- `src/train.py`, `src/models.py`: cross-validation training and model interfaces.
- `src/download_images.py`, `src/ocr.py`, `src/image_embeddings.py`: optional image workflow.
- `src/ensemble.py`, `src/postprocess.py`: OOF blending and calibration.
- `src/validate_submit.py`: submission format checks.
- `tests/`: automated tests; run them locally before relying on changes.

## Important competition hygiene

- Keep supplied data immutable and out of Git. `.gitignore` excludes data and generated artifacts.
- Use only training data for fitting and model selection. Never fit encoders, feature selectors, calibrators, or ensemble weights using held-out validation labels.
- Match the official metric and split strategy. Use OOF predictions for model comparison and stacking; keep the competition test set untouched.
- Store each experiment's config, seed, split identifiers, code revision, and validation metrics.
- Do not infer GPU compatibility from CUDA-enabled PyTorch alone; probe the actual model backend.

## Team collaboration

Use branches for experiments and small, reviewable pull requests. Do not commit credentials, private datasets, cache files, model weights, or submissions. For teammates, add GitHub collaborators explicitly rather than making this repository public.
