# Amazon ML Challenge: evidence-bounded historical notes

This document is a short orientation aid, not an official competition record, verified leaderboard, or recipe for winning. Editions and rules change; use each year's official problem statement and data as the source of truth. Do not cite unsourced team ranks, scores, dataset sizes, model details, or inferred “podium strategies.”

## 2025 Smart Product Pricing Challenge

The available public 2025 solution repositories describe a product-price regression task using catalog text and image links, evaluated with SMAPE. One public team solution documents input columns `sample_id`, `catalog_content`, `image_link`, and `price`; `catalog_content` combines product title, description, and item-pack quantity (IPQ). This is evidence about that solution's data format, not a guarantee that a future edition will reuse it.

The repository by SPAM_LLMs describes pretrained text and image encoders feeding a downstream model, and reports its own public/private leaderboard placements. Treat team-authored descriptions and reported standings as self-reported, not independently audited. Other public repositories document different approaches; there is no single verified “winning architecture” established here.

Sources:

- [SPAM_LLMs 2025 solution repository](https://github.com/RudrakshSJoshi/amlc-multimodal-mlp) — team-authored data/approach and result description.
- [LORAverse 2025 solution repository](https://github.com/theSohamTUmbare/Amazon-ML-Hackathon-2025) — another team-authored implementation; not a leaderboard authority.
- [IIT (ISM) Dhanbad newsletter PDF](https://www.iitism.ac.in/storage/newsletter-documents/newsletter1762776120.pdf) — published ranking table; consult the primary event organizer for official standing.

## What to carry forward

These are general experiment ideas, not historical claims about any particular placing:

- Parse explicit product quantities and units from the catalog text, then check parser precision on labeled examples.
- Compare sparse word/character features, pretrained representations, and tree models using identical leakage-safe folds.
- Treat images as an optional experiment: measure coverage, download cost, runtime, and validation gain before adding the path.
- Optimize and report the official metric on out-of-fold or held-out predictions. Never tune to the competition test labels or leaderboard feedback as if it were validation.
- Keep a simple reproducible baseline and preserve predictions/configuration for every experiment.

## For the next edition

Do not assume the task, schema, metric, split, model cap, or permitted external data. On release, read the official rules, inspect the CSV headers and sample submission, audit the data, and adapt the input pipeline before training. This repository's `src/prepare_challenge_data.py` is specifically a schema adapter for the 2025-style four-column format; it is not a universal parser for future datasets.
