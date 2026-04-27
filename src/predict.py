"""
predict.py
----------
Reusable inference function for the X-Health Default Prediction model.

Usage example:
    from src.predict import predict, predict_proba

    result = predict({
        "ioi_3months": 12.5,
        "valor_vencido": 0.0,
        "valor_total_pedido": 45000.0,
    })
    # Returns: {"default": 0} or {"default": 1}

    # Using a custom threshold (more conservative):
    result = predict(input_dict, threshold=0.4)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.preprocessing import (
    replace_missing_strings,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

# ── Default paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PIPELINE_PATH = _PROJECT_ROOT / "models" / "pipeline_xgb.joblib"

# ── Module-level cache (loaded once per process) ──────────────────────────────
_pipeline = None


def _load_pipeline(pipeline_path: Path = None):
    """Load the pipeline into memory (cached after first call)."""
    global _pipeline
    if _pipeline is None:
        path = pipeline_path or _DEFAULT_PIPELINE_PATH
        _pipeline = joblib.load(path)
    return _pipeline


def _build_input_df(input_dict: dict) -> pd.DataFrame:
    """Convert input dictionary to a single-row DataFrame ready for the pipeline."""
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    row = {col: input_dict.get(col, np.nan) for col in all_features}
    df_input = pd.DataFrame([row])
    return replace_missing_strings(df_input)


# ── Public API ────────────────────────────────────────────────────────────────

def predict(input_dict: dict, threshold: float = 0.5) -> dict:
    """
    Receive input data for a single order and return the default prediction.

    Parameters
    ----------
    input_dict : dict
        Dictionary with feature names as keys. Missing features are
        imputed automatically by the pipeline using training statistics.
    threshold : float, optional
        Classification threshold (default 0.5). Lower values make the model
        more conservative — catching more defaults at the cost of more false alarms.
        Should be defined by the finance team based on acceptable risk.

    Returns
    -------
    dict
        {"default": 0} — order predicted to be paid on time.
        {"default": 1} — order predicted to default.

    Example
    -------
    >>> predict({"ioi_3months": 10.0, "valor_vencido": 0.0, "valor_total_pedido": 30000.0})
    {"default": 0}

    >>> predict({"ioi_3months": 10.0, "valor_vencido": 0.0, "valor_total_pedido": 30000.0}, threshold=0.4)
    {"default": 1}
    """
    pipeline = _load_pipeline()
    df_input = _build_input_df(input_dict)

    probability = float(pipeline.predict_proba(df_input)[0][1])
    prediction = 1 if probability >= threshold else 0
    return {"default": prediction}


def predict_proba(input_dict: dict, threshold: float = 0.5) -> dict:
    """
    Like predict(), but also returns the raw probability of default.
    Use this when you want to apply a custom threshold or inspect the model confidence.

    Parameters
    ----------
    threshold : float, optional
        Classification threshold (default 0.5).

    Returns
    -------
    dict
        {"default": 0 or 1, "probability": float}
    """
    pipeline = _load_pipeline()
    df_input = _build_input_df(input_dict)

    probability = float(pipeline.predict_proba(df_input)[0][1])
    prediction = 1 if probability >= threshold else 0
    return {"default": prediction, "probability": round(probability, 4)}
