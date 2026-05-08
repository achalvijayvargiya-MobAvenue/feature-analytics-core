from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from adapters.two_tower_adapter.scoring import TwoTowerRowScorer, TwoTowerScoringConfig
from analytics_core.io.contracts import load_yaml


def _resolve_two_tower_train_config_path(root_cfg: dict) -> str:
    builder = root_cfg.get("prediction_dump_builder", {})
    p = builder.get("two_tower_train_config_path")
    if not p:
        raise ValueError("prediction_dump_builder.two_tower_train_config_path is required")
    return str(p)


def _resolve_columns(train_cfg: dict) -> tuple[str, str, list[str], list[str]]:
    f = train_cfg.get("features", {})
    label_col = str(f.get("label_col", "label"))
    client_id_col = str(f.get("client_id_col", "client_id"))
    user_cols = list(f.get("user_feature_cols") or [])
    client_cols = list(f.get("client_feature_cols") or [])
    return label_col, client_id_col, user_cols, client_cols


def _build_writer(output_path: str, schema: pa.Schema | None):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(str(p), schema=schema, compression="zstd")


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    builder = cfg.get("prediction_dump_builder", {})
    pipeline_cfg = cfg.get("pipeline", {})
    two_tower_cfg = pipeline_cfg.get("two_tower", {})

    train_cfg_path = _resolve_two_tower_train_config_path(cfg)
    train_cfg = load_yaml(train_cfg_path)
    label_col, client_id_col, user_cols, client_cols = _resolve_columns(train_cfg)
    val_path = str(train_cfg.get("paths", {}).get("val"))
    if not val_path:
        raise ValueError(f"Missing paths.val in {train_cfg_path}")

    output_path = str(builder.get("output_path", "outputs/predictions/validation_predictions.parquet"))
    max_rows = int(builder.get("max_rows", 0) or 0)
    batch_rows = int(builder.get("batch_rows", 50000))

    src_path = str(two_tower_cfg.get("src_path", "../Two_tower/src"))
    artifacts_base = str(two_tower_cfg.get("artifacts_base", ""))
    if not artifacts_base:
        raise ValueError("pipeline.two_tower.artifacts_base is required for build_prediction_dump")
    scorer = TwoTowerRowScorer(
        TwoTowerScoringConfig(
            src_path=src_path,
            artifacts_base=artifacts_base,
            client_id_col=client_id_col,
            device=str(two_tower_cfg.get("device", "cpu")),
            batch_size=int(two_tower_cfg.get("batch_size", 2048)),
        )
    )

    feature_cols = list(dict.fromkeys(user_cols + client_cols))
    requested_cols = [label_col, client_id_col] + feature_cols

    dset = pads.dataset(val_path, format="parquet")
    available = set(dset.schema.names)
    missing_required = [c for c in (label_col, client_id_col) if c not in available]
    if missing_required:
        raise KeyError(f"Validation dataset missing required columns: {missing_required}")
    projected = [c for c in requested_cols if c in available]

    row_id = 0
    total_rows = 0
    writer: pq.ParquetWriter | None = None
    output_cols = ["row_id", label_col, "prediction"] + feature_cols + [client_id_col]
    output_cols = list(dict.fromkeys(output_cols))

    print(f"[build_dump] val_path={val_path}")
    print(f"[build_dump] output_path={output_path}")
    print(f"[build_dump] projected_columns={len(projected)} requested={len(requested_cols)}")

    try:
        scanner = pads.Scanner.from_dataset(dset, columns=projected, batch_size=batch_rows)
        for rb in scanner.to_batches():
            pdf = rb.to_pandas(types_mapper=None)

            for c in output_cols:
                if c not in pdf.columns and c != "prediction":
                    if c in feature_cols:
                        pdf[c] = np.nan

            if max_rows > 0:
                remaining = max_rows - total_rows
                if remaining <= 0:
                    break
                if len(pdf) > remaining:
                    pdf = pdf.iloc[:remaining].copy()

            preds = scorer.predict_scores(pdf)
            out = pdf.copy()
            out["prediction"] = preds
            out["row_id"] = np.arange(row_id, row_id + len(out), dtype=np.int64)
            out = out[output_cols]

            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = _build_writer(output_path, table.schema)
            writer.write_table(table)

            total_rows += len(out)
            row_id += len(out)
            print(f"[build_dump] wrote_rows={total_rows}")

            if max_rows > 0 and total_rows >= max_rows:
                break
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise RuntimeError("No rows were written. Check val path and selected columns.")

    print(f"[build_dump] done rows={total_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prediction dump from validation source + true Two Tower scoring")
    parser.add_argument("--config", required=True, help="Path to analytics yaml config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

