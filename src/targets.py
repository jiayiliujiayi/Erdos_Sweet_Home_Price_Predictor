# src/targets.py

from __future__ import annotations

from typing import Literal
import numpy as np


TransformType = Literal["log", "log1p"]


def detect_log_transform(
    y_log,
    y_raw,
    tol: float = 1e-3,
) -> TransformType:
    """
    Heuristically detect whether y_log was created via log or log1p.

    Parameters
    ----------
    y_log : array-like
        Log-transformed targets used in modeling.
    y_raw : array-like
        Raw targets in original units (e.g., dollars).
    tol : float, default=1e-3
        Tolerance margin when comparing mean absolute differences.

    Returns
    -------
    {"log", "log1p"}
        Detected transform type.

    Notes
    -----
    This assumes either:
      - y_log = log(y_raw)
      - y_log = log1p(y_raw)
    and picks the one with smaller mean absolute difference.
    """
    y_log = np.asarray(y_log).reshape(-1)
    y_raw = np.asarray(y_raw).reshape(-1)

    pred_from_log = np.exp(y_log)
    pred_from_log1p = np.expm1(y_log)

    diff_log = np.mean(np.abs(pred_from_log - y_raw))
    diff_log1p = np.mean(np.abs(pred_from_log1p - y_raw))

    if diff_log + tol < diff_log1p:
        return "log"
    else:
        # default to log1p when close or better
        return "log1p"


def backtransform(
    y_pred_log,
    transform_type: TransformType,
):
    """
    Back-transform model predictions from log space to original scale.

    Parameters
    ----------
    y_pred_log : array-like
        Predicted targets in log space.
    transform_type : {"log", "log1p"}
        Transformation used to create y_log.

    Returns
    -------
    np.ndarray
        Predictions in original scale.
    """
    y_pred_log = np.asarray(y_pred_log)

    if transform_type == "log":
        return np.exp(y_pred_log)
    elif transform_type == "log1p":
        return np.expm1(y_pred_log)
    else:
        raise ValueError(f"Unsupported transform_type: {transform_type}")
