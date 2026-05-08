from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yaml


@dataclass(frozen=True)
class CrossCandidateResult:
    yaml_path: str
    metadata_path: str
    n_candidates: int


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype)


def _safe_nunique(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except Exception:
        return 0


def generate_cross_candidates(
    *,
    interactions_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    output_yaml_path: str,
    output_metadata_path: str,
    top_n: int = 10,
    max_cardinality_estimate: int = 2_000_000,
    max_feature_nunique: int = 50_000,
) -> CrossCandidateResult:
    """
    Phase-3: propose cross features from interaction pairs.

    Notes:
    - For now, we generate *recommendations* only (YAML + metadata).
    - We skip numeric×numeric or very high-cardinality columns.
    """

    required_cols = {"feature_a", "feature_b", "interaction_score"}
    if not required_cols.issubset(interactions_df.columns):
        raise KeyError(f"interactions_df missing required columns: {sorted(required_cols - set(interactions_df.columns))}")

    rows = []
    seen = set()
    for _, r in interactions_df.sort_values("interaction_score", ascending=False).iterrows():
        a = str(r["feature_a"])
        b = str(r["feature_b"])
        if a == b:
            continue
        if a not in prediction_df.columns or b not in prediction_df.columns:
            continue

        # normalize pair order
        k = tuple(sorted([a, b]))
        if k in seen:
            continue
        seen.add(k)

        sa = prediction_df[a]
        sb = prediction_df[b]

        # skip numeric×numeric crosses (usually too large/noisy for hashed crosses without binning)
        if _is_numeric(sa) and _is_numeric(sb):
            continue

        na = _safe_nunique(sa)
        nb = _safe_nunique(sb)
        if na <= 1 or nb <= 1:
            continue
        if na > max_feature_nunique or nb > max_feature_nunique:
            continue

        card_est = int(na) * int(nb)
        feasible = bool(card_est <= max_cardinality_estimate)

        name = f"{k[0]}_x_{k[1]}"
        rows.append(
            {
                "cross_feature": name,
                "feature_a": k[0],
                "feature_b": k[1],
                "interaction_score": float(r["interaction_score"]),
                "nunique_a": int(na),
                "nunique_b": int(nb),
                "cardinality_estimate": int(card_est),
                "feasible": bool(feasible),
            }
        )
        if len(rows) >= int(top_n):
            break

    meta = pd.DataFrame(rows).sort_values("interaction_score", ascending=False).reset_index(drop=True)
    meta.to_csv(output_metadata_path, index=False)

    candidates = [str(x) for x in meta.loc[meta["feasible"] == True, "cross_feature"].tolist()]  # noqa: E712
    payload = {
        "candidate_cross_features": candidates,
        "notes": {
            "generated_from": "interaction_scores.csv",
            "top_n_requested": int(top_n),
            "max_cardinality_estimate": int(max_cardinality_estimate),
            "max_feature_nunique": int(max_feature_nunique),
        },
    }
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return CrossCandidateResult(
        yaml_path=output_yaml_path, metadata_path=output_metadata_path, n_candidates=int(len(candidates))
    )

