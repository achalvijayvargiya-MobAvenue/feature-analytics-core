from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, ndcg_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PermutationResult:
    output_path: str
    baseline_auc: float
    baseline_logloss: float
    baseline_ndcg: float
    n_features: int


class RowScorerProtocol:
    """Minimal protocol for true model re-scoring."""

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - runtime protocol
        raise NotImplementedError


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return float("nan")


def _safe_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        p = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return float(log_loss(y_true, p))
    except Exception:
        return float("nan")


def _safe_ndcg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(ndcg_score(y_true[np.newaxis, :], y_pred[np.newaxis, :]))
    except Exception:
        return float("nan")


def _encode_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    enc = pd.DataFrame(index=df.index)
    for c in feature_cols:
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


def _train_surrogate_model(x_train: pd.DataFrame, y_train: np.ndarray):
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    model.fit(x_train, y_train)
    return model


def run_permutation_importance(
    validation_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    output_path: str,
    random_seed: int = 42,
    row_scorer: RowScorerProtocol | None = None,
) -> PermutationResult:
    if label_col not in validation_df.columns:
        raise KeyError(f"Label column {label_col!r} not found in prediction dump")
    if not feature_cols:
        raise ValueError("No feature columns provided for permutation importance")

    usable = [c for c in feature_cols if c in validation_df.columns]
    if not usable:
        raise ValueError("None of the configured features are present in prediction dump")

    y_all = validation_df[label_col].astype(float).to_numpy()
    if row_scorer is None:
        x_all = _encode_features(validation_df, usable)
        stratify = y_all if len(np.unique(y_all)) > 1 else None
        x_train, x_val, y_train, y_val = train_test_split(
            x_all, y_all, test_size=0.3, random_state=random_seed, stratify=stratify
        )
        model = _train_surrogate_model(x_train, y_train)
        y_pred = model.predict_proba(x_val)[:, 1]
        baseline_auc = _safe_auc(y_val, y_pred)
        baseline_logloss = _safe_logloss(y_val, y_pred)
        baseline_ndcg = _safe_ndcg(y_val, y_pred)
        base_df = validation_df.loc[x_val.index].copy()
    else:
        base_df = validation_df.copy()
        y_val = y_all
        y_pred = row_scorer.predict_scores(base_df)
        baseline_auc = _safe_auc(y_val, y_pred)
        baseline_logloss = _safe_logloss(y_val, y_pred)
        baseline_ndcg = _safe_ndcg(y_val, y_pred)

    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []
    for c in usable:
        perm_df = base_df.copy()
        perm_df[c] = rng.permutation(perm_df[c].to_numpy())
        if row_scorer is None:
            x_perm = _encode_features(perm_df, usable)
            y_perm = model.predict_proba(x_perm)[:, 1]
        else:
            y_perm = row_scorer.predict_scores(perm_df)

        auc_drop = baseline_auc - _safe_auc(y_val, y_perm)
        ndcg_drop = baseline_ndcg - _safe_ndcg(y_val, y_perm)
        logloss_increase = _safe_logloss(y_val, y_perm) - baseline_logloss
        rows.append(
            {
                "feature": c,
                "auc_drop": auc_drop,
                "ndcg_drop": ndcg_drop,
                "logloss_increase": logloss_increase,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
    out_df["importance_rank"] = np.arange(1, len(out_df) + 1)
    out_df.to_csv(output_path, index=False)

    return PermutationResult(
        output_path=output_path,
        baseline_auc=baseline_auc,
        baseline_logloss=baseline_logloss,
        baseline_ndcg=baseline_ndcg,
        n_features=len(usable),
    )

