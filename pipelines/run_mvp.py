from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analytics_core.importance.permutation_importance import run_permutation_importance
from analytics_core.io.contracts import (
    load_feature_catalog,
    load_model_manifest,
    load_prediction_dump,
    load_yaml,
)
from analytics_core.residuals.residual_model import train_residual_model
from analytics_core.encoding import encode_features_tabular
from analytics_core.interactions.interaction_mining import mine_interactions
from analytics_core.residuals.residual_dataset import build_residual_dataset
from analytics_core.candidates.cross_candidates import generate_cross_candidates
from analytics_core.ops.run_metadata import generate_run_id, write_run_metadata
from analytics_core.ops.data_quality import compute_data_quality


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
    run_cfg = cfg.get("run", {})
    base_out_dir = Path(run_cfg.get("output_dir", "outputs"))
    run_id = str(run_cfg.get("run_id") or generate_run_id(run_cfg.get("name")))
    out_dir = base_out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("run", {}).get("seed", 42))
    t0 = time.time()

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
    print(f"[mvp] run_id={run_id}")

    # Phase-5: basic data quality snapshot
    dq_dir = out_dir / "data_quality"
    dq = compute_data_quality(
        df=pred_df,
        label_col=label_col,
        prediction_col=prediction_col,
        feature_cols=feature_cols,
        output_dir=str(dq_dir),
    )
    print("[mvp] data_quality_summary=", dq.summary_path)

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
        min_slice_count = int(pipeline_cfg.get("residual_min_slice_count", 50))
        residual_result = build_residual_dataset(
            prediction_df=pred_df,
            feature_cols=feature_cols,
            label_col=label_col,
            prediction_col=prediction_col,
            dataset_path=residual_dataset_path,
            slices_path=residual_slices_path,
            min_slice_count=min_slice_count,
        )
        print("[mvp] residual_dataset=", residual_result.dataset_path)
        print("[mvp] residual_slices=", residual_result.slices_path)
        print("[mvp] residual_rows=", residual_result.rows)

    if bool(pipeline_cfg.get("run_residual_model", False)):
        target_col = str(pipeline_cfg.get("residual_target_col", "residual_bce"))
        model_path = str(residual_dir / "residual_model.pkl")
        fi_path = str(residual_dir / "tree_feature_importance.csv")
        metrics_path = str(residual_dir / "residual_model_metrics.json")
        y_true = pred_df[label_col].astype(float).to_numpy()
        y_prob = pred_df[prediction_col].astype(float).clip(1e-7, 1 - 1e-7).to_numpy()
        residual_abs = np.abs(y_true - y_prob)
        residual_bce = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
        rm = train_residual_model(
            df=pred_df.assign(residual_abs=residual_abs, residual_bce=residual_bce),
            feature_cols=feature_cols,
            target_col=target_col,
            model_path=model_path,
            feature_importance_path=fi_path,
            metrics_path=metrics_path,
            seed=seed,
        )
        print("[mvp] residual_model=", rm.model_path)
        print("[mvp] residual_model_metrics=", rm.metrics_path)
        print("[mvp] tree_feature_importance=", rm.feature_importance_path)

        if bool(pipeline_cfg.get("run_interaction_mining", False)):
            import pickle

            with open(model_path, "rb") as f:
                model = pickle.load(f)
            x = encode_features_tabular(pred_df, [c for c in feature_cols if c in pred_df.columns])
            max_ix_rows = int(pipeline_cfg.get("interaction_max_rows", 0))
            if max_ix_rows > 0 and len(x) > max_ix_rows:
                x = x.sample(n=max_ix_rows, random_state=seed).reset_index(drop=True)
                print(f"[mvp] interaction_mining subsample rows={len(x)} (interaction_max_rows={max_ix_rows})")
            interactions_dir = out_dir / "interactions"
            interactions_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(interactions_dir / "interaction_scores.csv")
            ix_method = str(pipeline_cfg.get("interaction_method", "auto"))
            im = mine_interactions(
                model=model,
                x=x,
                output_path=out_path,
                max_pairs=int(pipeline_cfg.get("max_interaction_pairs", 500)),
                method=ix_method,
            )
            print("[mvp] interactions=", im.output_path, "method=", im.method, "pairs=", im.pairs)

            if bool(pipeline_cfg.get("run_cross_candidates", False)):
                cross_dir = out_dir / "cross_candidates"
                cross_dir.mkdir(parents=True, exist_ok=True)
                yaml_path = str(cross_dir / "candidate_crosses.yaml")
                meta_path = str(cross_dir / "candidate_cross_metadata.csv")
                import pandas as pd

                interactions_df = pd.read_csv(out_path)
                cc = generate_cross_candidates(
                    interactions_df=interactions_df,
                    prediction_df=pred_df,
                    output_yaml_path=yaml_path,
                    output_metadata_path=meta_path,
                    top_n=int(pipeline_cfg.get("cross_top_n", 10)),
                    max_cardinality_estimate=int(pipeline_cfg.get("cross_max_cardinality_estimate", 2_000_000)),
                    max_feature_nunique=int(pipeline_cfg.get("cross_max_feature_nunique", 50_000)),
                )
                print("[mvp] cross_candidates_yaml=", cc.yaml_path)
                print("[mvp] cross_candidates_metadata=", cc.metadata_path)
                print("[mvp] cross_candidates_feasible=", cc.n_candidates)

    # Phase-5: run metadata for reproducibility
    elapsed_s = time.time() - t0
    write_run_metadata(
        output_dir=out_dir,
        run_id=run_id,
        config_path=config_path,
        config_dict=cfg,
        inputs={
            "model_manifest_path": manifest_path,
            "feature_catalog_path": feature_catalog_path,
            "prediction_dump_path": prediction_path,
        },
        outputs={
            "elapsed_s": float(elapsed_s),
        },
    )
    print(f"[mvp] done elapsed_s={elapsed_s:.1f} metadata={out_dir / 'run_metadata.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analytics MVP pipeline")
    parser.add_argument("--config", required=True, help="Path to analytics yaml config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

