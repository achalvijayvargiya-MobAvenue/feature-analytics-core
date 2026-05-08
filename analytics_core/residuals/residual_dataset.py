from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResidualResult:
    dataset_path: str
    slices_path: str
    rows: int


def _binary_cross_entropy_per_row(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    p = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def build_residual_dataset(
    prediction_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    prediction_col: str,
    dataset_path: str,
    slices_path: str,
    max_slice_cardinality: int = 50,
    top_k_per_feature: int = 10,
) -> ResidualResult:
    if label_col not in prediction_df.columns:
        raise KeyError(f"Label column {label_col!r} not found")
    if prediction_col not in prediction_df.columns:
        raise KeyError(f"Prediction column {prediction_col!r} not found")

    df = prediction_df.copy()
    y_true = df[label_col].astype(float).to_numpy()
    y_prob = np.clip(df[prediction_col].astype(float).to_numpy(), 0.0, 1.0)

    df["residual_abs"] = np.abs(y_true - y_prob)
    df["residual_bce"] = _binary_cross_entropy_per_row(y_true, y_prob)

    cols_to_keep = [label_col, prediction_col, "residual_abs", "residual_bce"]
    cols_to_keep.extend([c for c in feature_cols if c in df.columns])
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    out_df = df[cols_to_keep]
    out_df.to_parquet(dataset_path, index=False)

    slice_rows: list[dict] = []
    candidate_slice_cols = [c for c in feature_cols if c in df.columns]
    for c in candidate_slice_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s.dtype):
            continue
        nunique = int(s.nunique(dropna=True))
        if nunique == 0 or nunique > max_slice_cardinality:
            continue
        g = (
            df.assign(_slice_value=s.astype("string").fillna("__MISSING__"))
            .groupby("_slice_value", as_index=False)
            .agg(
                count=("residual_abs", "size"),
                avg_residual_abs=("residual_abs", "mean"),
                avg_residual_bce=("residual_bce", "mean"),
            )
            .sort_values("avg_residual_abs", ascending=False)
            .head(top_k_per_feature)
        )
        for _, row in g.iterrows():
            slice_rows.append(
                {
                    "feature": c,
                    "slice_value": row["_slice_value"],
                    "count": int(row["count"]),
                    "avg_residual_abs": float(row["avg_residual_abs"]),
                    "avg_residual_bce": float(row["avg_residual_bce"]),
                }
            )

    slice_df = pd.DataFrame(slice_rows)
    if slice_df.empty:
        slice_df = pd.DataFrame(
            columns=["feature", "slice_value", "count", "avg_residual_abs", "avg_residual_bce"]
        )
    slice_df.to_csv(slices_path, index=False)

    return ResidualResult(dataset_path=dataset_path, slices_path=slices_path, rows=len(out_df))

