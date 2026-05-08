from __future__ import annotations

import io
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class TwoTowerScoringConfig:
    src_path: str
    artifacts_base: str
    client_id_col: str = "client_id"
    device: str = "cpu"
    batch_size: int = 4096


class TwoTowerRowScorer:
    """True scorer using exported Two Tower artifacts."""

    def __init__(self, cfg: TwoTowerScoringConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._load_runtime()

    def _load_runtime(self) -> None:
        src_path = str(Path(self.cfg.src_path).expanduser().resolve())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from two_tower.features.encode import encode_cats, encode_multi_matrix, encode_nums
        from two_tower.features.vocab import vocab_from_dict
        from two_tower.inference.artifact_paths import training_artifact_uris
        from two_tower.io.uris import read_uri_bytes
        from two_tower.model.two_tower import UserMLPTower

        arts = training_artifact_uris(self.cfg.artifacts_base)

        user_ckpt = torch.load(io.BytesIO(read_uri_bytes(arts["user_tower"])), map_location="cpu")
        vocab_artifact = pickle.loads(read_uri_bytes(arts["vocab"]))

        client_raw = read_uri_bytes(arts["client_embeddings"])
        client_df = pd.read_parquet(io.BytesIO(client_raw))
        if "client_id" not in client_df.columns or "embedding" not in client_df.columns:
            raise KeyError("client_embeddings artifact must contain columns: client_id, embedding")

        self.client_emb_by_id: dict[object, np.ndarray] = {}
        for _, row in client_df.iterrows():
            cid = row["client_id"]
            vec = np.asarray(row["embedding"], dtype=np.float32)
            self.client_emb_by_id[cid] = vec
            self.client_emb_by_id[str(cid)] = vec

        self.user_cat_cols = list(vocab_artifact["user_cat_cols"])
        self.user_num_cols = list(vocab_artifact["user_num_cols"])
        self.user_multi_cols = list(vocab_artifact.get("user_multi_cols", []))
        self.multi_cat_max_tokens = int(vocab_artifact["multi_cat_max_tokens"])
        self.user_vocabs = {k: vocab_from_dict(v) for k, v in vocab_artifact["user_vocabs"].items()}
        self.user_multi_vocabs = {k: vocab_from_dict(v) for k, v in vocab_artifact["user_multi_vocabs"].items()}
        self.log_scale = float(user_ckpt.get("log_scale", np.log(20.0)))

        self.user_model = UserMLPTower(
            user_vocab_sizes=list(user_ckpt["user_vocab_sizes"]),
            user_num_dim=int(user_ckpt["user_num_dim"]),
            emb_dim=int(user_ckpt["emb_dim"]),
            hidden=list(user_ckpt.get("user_hidden", [256, 256])),
            user_multi_vocab_sizes=list(user_ckpt.get("user_multi_vocab_sizes", [])) or None,
            user_multi_emb_dims=list(user_ckpt.get("user_multi_emb_dims", [])) or None,
            multi_pool=str(user_ckpt.get("multi_cat_pool", "mean")),
            use_pretrained_cat=bool(user_ckpt.get("use_pretrained_cat", False)),
            pretrained_emb_dim=int(user_ckpt.get("pretrained_emb_dim", 128)),
            target_cat_emb_dim=int(user_ckpt.get("target_cat_emb_dim", 64)),
            freeze_base=bool(user_ckpt.get("freeze_base", True)),
        )
        self.user_model.load_state_dict(user_ckpt["state_dict"])
        self.user_model.to(self.device)
        self.user_model.eval()

        self.encode_cats = encode_cats
        self.encode_nums = encode_nums
        self.encode_multi_matrix = encode_multi_matrix
        self.scaling = float(np.exp(self.log_scale))

    @property
    def scorable_feature_cols(self) -> list[str]:
        return list(self.user_cat_cols) + list(self.user_num_cols) + list(self.user_multi_cols)

    def _lookup_client_matrix(self, client_ids: list[object]) -> np.ndarray:
        dim = next(iter(self.client_emb_by_id.values())).shape[0]
        out = np.zeros((len(client_ids), dim), dtype=np.float32)
        for i, cid in enumerate(client_ids):
            emb = self.client_emb_by_id.get(cid)
            if emb is None:
                emb = self.client_emb_by_id.get(str(cid))
            if emb is not None:
                out[i] = emb
        return out

    def build_guard_report(self, df: pd.DataFrame, requested_feature_cols: list[str] | None = None) -> dict[str, Any]:
        cid_col = self.cfg.client_id_col
        if cid_col not in df.columns:
            raise KeyError(f"Required client id column missing for true scoring: {cid_col!r}")

        scorable = list(self.scorable_feature_cols)
        requested = list(requested_feature_cols or [])
        used = [c for c in scorable if c in df.columns]
        ignored_requested = [c for c in requested if c not in used]
        missing_from_dump = [c for c in scorable if c not in df.columns]

        client_ids = df[cid_col].tolist()
        missing_embedding_count = 0
        missing_examples: list[str] = []
        for cid in client_ids:
            emb = self.client_emb_by_id.get(cid)
            if emb is None:
                emb = self.client_emb_by_id.get(str(cid))
            if emb is None:
                missing_embedding_count += 1
                if len(missing_examples) < 20:
                    missing_examples.append(str(cid))

        total_rows = len(df)
        missing_ratio = (missing_embedding_count / total_rows) if total_rows else 0.0
        return {
            "mode": "two_tower_true_rescore",
            "client_id_col": cid_col,
            "total_rows": int(total_rows),
            "scorable_feature_cols_count": len(scorable),
            "used_feature_cols_count": len(used),
            "requested_feature_cols_count": len(requested),
            "ignored_requested_feature_cols_count": len(ignored_requested),
            "missing_scorable_feature_cols_count": len(missing_from_dump),
            "missing_client_embedding_count": int(missing_embedding_count),
            "missing_client_embedding_ratio": float(missing_ratio),
            "used_feature_cols": used,
            "ignored_requested_feature_cols": ignored_requested,
            "missing_scorable_feature_cols": missing_from_dump,
            "missing_client_embedding_examples": missing_examples,
        }

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        cid_col = self.cfg.client_id_col
        if cid_col not in df.columns:
            raise KeyError(f"Required client id column missing for true scoring: {cid_col!r}")

        batch_size = max(1, int(self.cfg.batch_size))
        preds: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(df), batch_size):
                chunk = df.iloc[start : start + batch_size]
                user_cat = self.encode_cats(chunk, self.user_cat_cols, self.user_vocabs).to(self.device)
                user_num = self.encode_nums(chunk, self.user_num_cols).to(self.device)
                user_multi = self.encode_multi_matrix(
                    chunk, self.user_multi_cols, self.user_multi_vocabs, self.multi_cat_max_tokens
                ).to(self.device)
                user_emb = self.user_model(user_cat, user_num, user_multi).detach().cpu().numpy().astype(np.float32)

                client_ids = chunk[cid_col].tolist()
                client_emb = self._lookup_client_matrix(client_ids)
                logits = (user_emb * client_emb).sum(axis=1) * self.scaling
                prob = 1.0 / (1.0 + np.exp(-logits))
                preds.append(prob.astype(np.float64))

        return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float64)

