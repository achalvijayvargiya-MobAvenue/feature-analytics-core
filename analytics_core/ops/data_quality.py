from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityResult:
    summary_path: str
    per_feature_path: str


def compute_data_quality(
    *,
    df: pd.DataFrame,
    label_col: str,
    prediction_col: str,
    feature_cols: list[str],
    output_dir: str,
) -> DataQualityResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "label_col": label_col,
        "prediction_col": prediction_col,
    }

    if label_col in df.columns:
        y = df[label_col].astype(float)
        summary["label_pos_rate"] = float(y.mean()) if len(y) else None
        summary["label_null_rate"] = float(y.isna().mean()) if len(y) else None
    if prediction_col in df.columns:
        p = df[prediction_col].astype(float)
        summary["prediction_mean"] = float(p.mean()) if len(p) else None
        summary["prediction_null_rate"] = float(p.isna().mean()) if len(p) else None

    rows = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        s = df[c]
        null_rate = float(s.isna().mean()) if len(s) else 0.0
        nunique = int(s.nunique(dropna=True))
        dt = str(s.dtype)
        rows.append({"feature": c, "dtype": dt, "null_rate": null_rate, "nunique": nunique})

    per_feature = pd.DataFrame(rows).sort_values("null_rate", ascending=False)
    per_feature_path = out / "per_feature.csv"
    per_feature.to_csv(per_feature_path, index=False)

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return DataQualityResult(summary_path=str(summary_path), per_feature_path=str(per_feature_path))

