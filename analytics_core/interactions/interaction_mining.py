from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InteractionMiningResult:
    output_path: str
    method: str
    pairs: int


def _interaction_pairs_from_shap(model, x: pd.DataFrame, *, max_pairs: int = 500) -> pd.DataFrame:
    import shap

    explainer = shap.TreeExplainer(model)
    # For regression this returns (N, F, F)
    vals = explainer.shap_interaction_values(x)
    arr = np.asarray(vals)
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP interaction shape: {arr.shape}")
    mean_abs = np.mean(np.abs(arr), axis=0)  # (F,F)
    f = mean_abs.shape[0]
    rows = []
    cols = list(x.columns)
    for i in range(f):
        for j in range(i + 1, f):
            rows.append((cols[i], cols[j], float(mean_abs[i, j])))
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "interaction_score"])
    out = out.sort_values("interaction_score", ascending=False).head(max_pairs).reset_index(drop=True)
    return out


def _interaction_pairs_from_tree_splits(model, feature_names: list[str], *, max_pairs: int = 500) -> pd.DataFrame:
    """
    Fallback interaction heuristic: count split co-occurrence within each tree.
    Not as strong as SHAP interactions, but works without shap.
    """

    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError("Model has no booster_ to extract trees from")
    dump = booster.dump_model()
    trees = dump.get("tree_info", []) or []

    pair_counts: dict[tuple[str, str], int] = {}

    def walk(node, path_features: set[str]):
        if "split_feature" in node:
            idx = int(node["split_feature"])
            if 0 <= idx < len(feature_names):
                f = feature_names[idx]
            else:
                f = f"f{idx}"
            new_path = set(path_features)
            new_path.add(f)
            # recurse
            walk(node.get("left_child", {}), new_path)
            walk(node.get("right_child", {}), new_path)
        else:
            # leaf: update all pairs in path
            feats = sorted(path_features)
            for i in range(len(feats)):
                for j in range(i + 1, len(feats)):
                    k = (feats[i], feats[j])
                    pair_counts[k] = pair_counts.get(k, 0) + 1

    for t in trees:
        tree = t.get("tree_structure") or {}
        walk(tree, set())

    rows = [(a, b, float(c)) for (a, b), c in pair_counts.items()]
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "interaction_score"])
    out = out.sort_values("interaction_score", ascending=False).head(max_pairs).reset_index(drop=True)
    return out


def mine_interactions(
    *,
    model,
    x: pd.DataFrame,
    output_path: str,
    max_pairs: int = 500,
    method: str = "auto",
) -> InteractionMiningResult:
    """
    method:
      - auto: SHAP interactions if possible, else tree split co-occurrence
      - shap: SHAP only (raises if it fails)
      - tree_split: fast heuristic from booster trees (no SHAP)
    """

    key = (method or "auto").strip().lower()
    if key == "tree_split":
        used = "tree_split_cooccurrence"
        out = _interaction_pairs_from_tree_splits(model, list(x.columns), max_pairs=max_pairs)
    elif key == "shap":
        used = "shap"
        out = _interaction_pairs_from_shap(model, x, max_pairs=max_pairs)
    else:
        used = "shap"
        try:
            out = _interaction_pairs_from_shap(model, x, max_pairs=max_pairs)
        except Exception:
            used = "tree_split_cooccurrence"
            out = _interaction_pairs_from_tree_splits(model, list(x.columns), max_pairs=max_pairs)
    out.to_csv(output_path, index=False)
    return InteractionMiningResult(output_path=output_path, method=used, pairs=int(out.shape[0]))

