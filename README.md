# feature-analytics-core

Contract-first, model-agnostic offline feature analytics pipeline for recommendation models.

## Why this repo exists

This repository is intentionally separate from `Two_tower` so you can:
- release/deploy analytics independently
- move this project without moving training/inference code
- keep strict boundaries through versioned contracts

## Core principles

1. Analytics core does not import model training/inference internals.
2. Model-specific logic is implemented through adapters.
3. Inputs are validated via contracts before running analytics.

## Expected inputs

- `model_manifest.json`
- `feature_catalog.parquet` (or yaml)
- `validation_predictions.parquet`

## Current structure

- `contracts/` schema and compatibility contracts
- `configs/` runtime configs
- `analytics_core/` model-agnostic analytics modules
- `adapters/` model-specific adapters (optional)
- `pipelines/` orchestration entrypoints
- `tests/` contract and pipeline tests

## Quick start

```bash
python -m pip install -e .
python -m pipelines.build_prediction_dump --config configs/analytics.yaml
python -m pipelines.validate_contracts --config configs/analytics.yaml
python -m pipelines.run_mvp --config configs/analytics.yaml
```

## True Two Tower re-scoring mode

For real permutation importance (no surrogate approximation), configure:

- `pipeline.permutation_mode: two_tower_true_rescore`
- `pipeline.two_tower.src_path` (path to `Two_tower/src`)
- `pipeline.two_tower.artifacts_base` (same artifacts base used by training)
- `pipeline.two_tower.client_id_col` (column in prediction dump)

This mode loads exported Two Tower artifacts and re-scores rows with current inference behavior:
user tower embedding + client embedding lookup + dot-product scaling.

When `two_tower_true_rescore` is used, pipeline writes:
- `outputs/feature_importance/true_rescore_guard_report.json`

This guard report makes usage explicit:
- features used for true re-scoring
- requested-but-ignored features
- scorable features missing in prediction dump
- missing client embedding coverage stats

## Notes

- Keep outputs in `outputs/` for local runs.
- Prefer Parquet for large datasets.
- Add adapter compatibility matrix as versions evolve.

