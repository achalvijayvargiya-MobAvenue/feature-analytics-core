from __future__ import annotations

import argparse
from pathlib import Path

from analytics_core.io.contracts import load_feature_catalog, load_model_manifest, load_prediction_dump, load_yaml


REQUIRED_MANIFEST_KEYS = {
    "model_id",
    "model_version",
    "trained_at",
    "label_column",
    "prediction_schema_version",
    "feature_catalog_version",
}


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    contracts = cfg.get("contracts", {})
    manifest_path = contracts.get("model_manifest_path")
    feature_catalog_path = contracts.get("feature_catalog_path")
    prediction_path = contracts.get("prediction_dump_path")

    if not manifest_path or not prediction_path or not feature_catalog_path:
        raise ValueError("Missing contracts paths in config")

    m = load_model_manifest(manifest_path)
    missing = sorted(REQUIRED_MANIFEST_KEYS - set(m.keys()))
    if missing:
        raise KeyError(f"model_manifest missing keys: {missing}")

    df = load_prediction_dump(prediction_path)
    for col in ("row_id", "label", "prediction"):
        if col not in df.columns:
            raise KeyError(f"prediction dump missing required column: {col}")

    fc = load_feature_catalog(feature_catalog_path)
    if "feature_name" not in fc.columns:
        raise KeyError("feature catalog missing required column: feature_name")

    print(f"[validate] manifest ok: {Path(manifest_path)}")
    print(f"[validate] feature catalog ok: {Path(feature_catalog_path)}")
    print(f"[validate] prediction dump ok: {Path(prediction_path)}")
    print(f"[validate] rows={len(df)} cols={len(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate contract inputs")
    parser.add_argument("--config", required=True, help="Path to analytics yaml config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

