from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    output_dir: str
    metadata_path: str


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def generate_run_id(name: str | None = None) -> str:
    base = (name or "run").strip().replace(" ", "_")
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return f"{base}_{ts}"


def write_run_metadata(
    *,
    output_dir: Path,
    run_id: str,
    config_path: str,
    config_dict: dict,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> RunMetadata:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "created_at_utc": _now_utc_iso(),
        "host": {"hostname": socket.gethostname(), "user": os.getenv("USER") or os.getenv("USERNAME")},
        "config_path": config_path,
        "config": config_dict,
        "inputs": inputs,
        "outputs": outputs,
    }
    meta_path = output_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return RunMetadata(run_id=run_id, output_dir=str(output_dir), metadata_path=str(meta_path))

