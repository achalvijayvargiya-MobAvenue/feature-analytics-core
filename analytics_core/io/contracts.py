from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def _read_text(path: str | Path) -> str:
    s = str(path)
    if s.startswith("s3://"):
        import s3fs

        with s3fs.S3FileSystem().open(s, "r", encoding="utf-8") as f:
            return f.read()
    return Path(path).read_text(encoding="utf-8")


def load_yaml(path: str | Path) -> dict:
    data = yaml.safe_load(_read_text(path)) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data)}")
    return data


def load_model_manifest(path: str | Path) -> dict:
    data = json.loads(_read_text(path))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data)}")
    return data


def load_prediction_dump(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(str(path))


def load_feature_catalog(path: str | Path) -> pd.DataFrame:
    s = str(path)
    suffix = Path(s).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(s)
    if suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(_read_text(path)) or {}
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            if "features" in raw and isinstance(raw["features"], list):
                return pd.DataFrame(raw["features"])
            if "feature_catalog" in raw and isinstance(raw["feature_catalog"], list):
                return pd.DataFrame(raw["feature_catalog"])
            # Best effort: key-value mapping by feature name.
            rows = []
            for k, v in raw.items():
                if isinstance(v, dict):
                    row = {"feature_name": str(k)}
                    row.update(v)
                    rows.append(row)
            if rows:
                return pd.DataFrame(rows)
        raise TypeError(f"Unsupported yaml feature catalog structure in {path}")
    raise ValueError(f"Unsupported feature catalog format: {suffix}")

