from __future__ import annotations

import numpy as np
import pandas as pd


def encode_features_tabular(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Encode mixed-type columns into numeric matrix for tree models.

    - numeric -> float (median-impute)
    - bool -> int8
    - datetime -> int64 view
    - object/string -> factorize codes
    """

    enc = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s.dtype):
            enc[c] = s.astype("int8")
        elif pd.api.types.is_datetime64_any_dtype(s.dtype):
            enc[c] = s.view("int64").astype("float64")
            enc[c] = enc[c].replace({np.iinfo("int64").min: np.nan}).fillna(0.0)
        elif pd.api.types.is_numeric_dtype(s.dtype):
            vals = s.astype("float64")
            enc[c] = vals.fillna(vals.median() if not vals.dropna().empty else 0.0)
        else:
            vals = s.astype("string").fillna("__MISSING__")
            codes, _ = pd.factorize(vals, sort=True)
            enc[c] = codes.astype("int32")
    return enc

