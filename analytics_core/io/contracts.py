from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {p}, got {type(data)}")
    return data


def load_model_manifest(path: str | Path) -> dict:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {p}, got {type(data)}")
    return data


def load_prediction_dump(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p)

