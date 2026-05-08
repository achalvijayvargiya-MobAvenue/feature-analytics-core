from __future__ import annotations

import argparse
import json
from pathlib import Path

from analytics_core.importance.permutation_importance import run_permutation_importance
from analytics_core.io.contracts import (
    load_feature_catalog,
    load_model_manifest,
    load_prediction_dump,
    load_yaml,
)
from analytics_core.residuals.residual_dataset import build_residual_dataset


DEFAULT_NON_FEATURE_COLS = {
    "row_id",
    "label",
    "prediction",
    "logits",
    "user_embedding_norm",
    "client_embedding_norm",
}


def _resolve_feature_columns(pred_df, feature_catalog_df) -> list[str]:
    if feature_catalog_df is not None and "feature_name" in feature_catalog_df.columns:
        cols = [str(c) for c in feature_catalog_df["feature_name"].dropna().tolist()]
        cols = [c for c in cols if c in pred_df.columns]
        if cols:
            return sorted(dict.fromkeys(cols))
    inferred = [c for c in pred_df.columns if c not in DEFAULT_NON_FEATURE_COLS]
    return sorted(inferred)


def _build_two_tower_row_scorer(cfg: dict):
    from adapters.two_tower_adapter.scoring import TwoTowerRowScorer, TwoTowerScoringConfig

    tw = cfg.get("two_tower", {})
    src_path = tw.get("src_path")
    artifacts_base = tw.get("artifacts_base")
    if not src_path or not artifacts_base:
        raise ValueError("For two_tower_true_rescore mode, set pipeline.two_tower.src_path and artifacts_base")

    scorer_cfg = TwoTowerScoringConfig(
        src_path=str(src_path),
        artifacts_base=str(artifacts_base),
        client_id_col=str(tw.get("client_id_col", "client_id")),
        device=str(tw.get("device", "cpu")),
        batch_size=int(tw.get("batch_size", 4096)),
    )
    return TwoTowerRowScorer(scorer_cfg)


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    contracts = cfg.get("contracts", {})
    out_dir = Path(cfg.get("run", {}).get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("run", {}).get("seed", 42))

    manifest_path = contracts.get("model_manifest_path")
    feature_catalog_path = contracts.get("feature_catalog_path")
    prediction_path = contracts.get("prediction_dump_path")
    if not manifest_path or not prediction_path:
        raise ValueError("Missing contracts.model_manifest_path or contracts.prediction_dump_path in config")

    manifest = load_model_manifest(manifest_path)
    pred_df = load_prediction_dump(prediction_path)
    label_col = str(manifest.get("label_column", "label"))
    prediction_col = "prediction"
    pipeline_cfg = cfg.get("pipeline", {})

    feature_catalog_df = None
    if feature_catalog_path and Path(feature_catalog_path).exists():
        feature_catalog_df = load_feature_catalog(feature_catalog_path)
    feature_cols = _resolve_feature_columns(pred_df, feature_catalog_df)

    if not feature_cols:
        raise ValueError("No feature columns found in prediction dump")

    feature_importance_dir = out_dir / "feature_importance"
    residual_dir = out_dir / "residual_analysis"
    feature_importance_dir.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)

    print("[mvp] run_name=", cfg.get("run", {}).get("name", "local"))
    print("[mvp] model_id=", manifest.get("model_id"))
    print("[mvp] model_version=", manifest.get("model_version"))
    print("[mvp] rows=", len(pred_df))
    print("[mvp] columns=", len(pred_df.columns))
    print("[mvp] feature_columns=", len(feature_cols))
    print(f"[mvp] output_dir={out_dir}")

    if bool(pipeline_cfg.get("run_permutation_importance", True)):
        permutation_mode = str(pipeline_cfg.get("permutation_mode", "surrogate")).lower().strip()
        row_scorer = None
        if permutation_mode == "two_tower_true_rescore":
            row_scorer = _build_two_tower_row_scorer(pipeline_cfg)
            guard_report = row_scorer.build_guard_report(pred_df, requested_feature_cols=feature_cols)
            guard_report_path = feature_importance_dir / "true_rescore_guard_report.json"
            guard_report_path.write_text(json.dumps(guard_report, indent=2), encoding="utf-8")
            feature_cols = [c for c in row_scorer.scorable_feature_cols if c in pred_df.columns]
            if not feature_cols:
                raise ValueError("No Two Tower user feature columns found in prediction dump for true re-scoring")
            print("[mvp] permutation_mode=two_tower_true_rescore")
            print("[mvp] guard_report=", str(guard_report_path))
            print(
                "[mvp] guard_used_features=",
                guard_report.get("used_feature_cols_count"),
                "ignored_requested=",
                guard_report.get("ignored_requested_feature_cols_count"),
            )
            print(
                "[mvp] guard_missing_client_embeddings=",
                guard_report.get("missing_client_embedding_count"),
                f"({guard_report.get('missing_client_embedding_ratio', 0.0):.4%})",
            )
        elif permutation_mode == "surrogate":
            print("[mvp] permutation_mode=surrogate")
        else:
            raise ValueError(f"Unsupported pipeline.permutation_mode: {permutation_mode!r}")

        pi_out = str(feature_importance_dir / "permutation_importance.csv")
        pi_result = run_permutation_importance(
            validation_df=pred_df,
            feature_cols=feature_cols,
            label_col=label_col,
            output_path=pi_out,
            random_seed=seed,
            row_scorer=row_scorer,
        )
        print("[mvp] permutation_importance=", pi_result.output_path)
        print("[mvp] baseline_auc=", f"{pi_result.baseline_auc:.6f}")
        print("[mvp] baseline_logloss=", f"{pi_result.baseline_logloss:.6f}")
        print("[mvp] baseline_ndcg=", f"{pi_result.baseline_ndcg:.6f}")

    if bool(pipeline_cfg.get("run_residual_dataset", True)):
        residual_dataset_path = str(residual_dir / "residual_dataset.parquet")
        residual_slices_path = str(residual_dir / "residual_slices.csv")
        residual_result = build_residual_dataset(
            prediction_df=pred_df,
            feature_cols=feature_cols,
            label_col=label_col,
            prediction_col=prediction_col,
            dataset_path=residual_dataset_path,
            slices_path=residual_slices_path,
        )
        print("[mvp] residual_dataset=", residual_result.dataset_path)
        print("[mvp] residual_slices=", residual_result.slices_path)
        print("[mvp] residual_rows=", residual_result.rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analytics MVP pipeline")
    parser.add_argument("--config", required=True, help="Path to analytics yaml config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

