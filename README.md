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
python -m pipelines.run_mvp --config configs/analytics.yaml
```

## Notes

- Keep outputs in `outputs/` for local runs.
- Prefer Parquet for large datasets.
- Add adapter compatibility matrix as versions evolve.

