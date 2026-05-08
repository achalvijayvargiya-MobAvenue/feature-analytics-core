from __future__ import annotations

import argparse
from pathlib import Path

from analytics_core.io.contracts import load_model_manifest, load_prediction_dump, load_yaml


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    contracts = cfg.get("contracts", {})
    out_dir = Path(cfg.get("run", {}).get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = contracts.get("model_manifest_path")
    prediction_path = contracts.get("prediction_dump_path")
    if not manifest_path or not prediction_path:
        raise ValueError("Missing contracts.model_manifest_path or contracts.prediction_dump_path in config")

    manifest = load_model_manifest(manifest_path)
    pred_df = load_prediction_dump(prediction_path)

    print("[mvp] run_name=", cfg.get("run", {}).get("name", "local"))
    print("[mvp] model_id=", manifest.get("model_id"))
    print("[mvp] model_version=", manifest.get("model_version"))
    print("[mvp] rows=", len(pred_df))
    print("[mvp] columns=", len(pred_df.columns))
    print(f"[mvp] output_dir={out_dir}")
    print("[mvp] scaffolding ready. Add permutation/residual modules next.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analytics MVP pipeline")
    parser.add_argument("--config", required=True, help="Path to analytics yaml config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

