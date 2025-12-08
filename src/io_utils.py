# src/io_utils.py

from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import json


def pick_single_file(pattern: str, root: Path) -> Path:
    """
    Pick a single file under `root` matching `pattern`, preferring the latest
    (lexicographically sorted) if multiple matches exist.

    Parameters
    ----------
    pattern : str
        Glob pattern (e.g. "train_text_emb_*.csv").
    root : Path
        Directory in which to search.

    Returns
    -------
    Path
        Selected file path.

    Raises
    ------
    FileNotFoundError
        If no files match the pattern.
    """
    root = Path(root)
    candidates = sorted(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files match pattern '{pattern}' under {root}")

    if len(candidates) > 1:
        print(f"[pick_single_file] {len(candidates)} matches found, using latest:\n"
              f"  {candidates[-1].name}")

    return candidates[-1]


def load_json(path: Path) -> Any:
    """
    Load a JSON file and return its content.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    Any
        Parsed JSON content.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prep_summary(path: Path) -> Dict[str, Any]:
    """
    Convenience wrapper to load the multimodal prep summary JSON.

    Parameters
    ----------
    path : Path
        Path to 'multimodal_prep_summary.json'.

    Returns
    -------
    dict
        Summary dictionary.
    """
    return load_json(path)
