# src/paths.py

from __future__ import annotations

from typing import Dict
from pathlib import Path


def get_project_paths() -> Dict[str, Path]:
    """
    Infer project-level paths assuming this file lives in src/ under the
    project root.

    Returns
    -------
    dict
        Dictionary with keys:
        - PROJECT_ROOT
        - DATA_DIR
        - RAW_DIR
        - PROC_DIR
    """
    # This file is src/paths.py  -> parents[1] is project root
    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    proc_dir = data_dir / "processed"

    return {
        "PROJECT_ROOT": project_root,
        "DATA_DIR": data_dir,
        "RAW_DIR": raw_dir,
        "PROC_DIR": proc_dir,
    }
