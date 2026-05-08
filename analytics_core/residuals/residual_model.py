from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from analytics_core.encoding import encode_features_tabular


@dataclass(frozen=True)
class ResidualModelResult:
    model_path: str
    feature_importance_path: str
    metrics_path: str
    n_rows: int
    n_features: int


def train_residual_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    model_path: str,
    feature_importance_path: str,
    metrics_path: str,
    seed: int = 42,
) -> ResidualModelResult:
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found for residual model")

    usable = [c for c in feature_cols if c in df.columns]
    if not usable:
        raise ValueError("No usable feature columns for residual model")

    x = encode_features_tabular(df, usable)
    y = df[target_col].astype(float).to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)

    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_val)
    metrics = {
        "target_col": target_col,
        "rows_train": int(x_train.shape[0]),
        "rows_val": int(x_val.shape[0]),
        "rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
        "mae": float(mean_absolute_error(y_val, pred)),
        "r2": float(r2_score(y_val, pred)),
    }

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    imp = pd.DataFrame(
        {
            "feature": list(x.columns),
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(feature_importance_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, indent=2))

    return ResidualModelResult(
        model_path=model_path,
        feature_importance_path=feature_importance_path,
        metrics_path=metrics_path,
        n_rows=int(df.shape[0]),
        n_features=int(x.shape[1]),
    )

