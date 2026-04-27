"""
preprocessing.py
----------------
Reusable preprocessing pipeline for the X-Health Default Prediction project.

Responsibilities:
  - Replace 'missing' strings with NaN
  - Engineer new features (ratio_vencido_quitado, total_serasa)
  - Impute missing values (median for numerics, mode for categoricals)
  - Encode categorical variables (OneHotEncoder with max_categories to handle high cardinality)
  - Scale numerical features (RobustScaler — robust to outliers)
  - Expose a build function so the pipeline is always fitted only on training data (no data leakage)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# ── Constants ────────────────────────────────────────────────────────────────

TARGET = "default"

NUMERICAL_FEATURES = [
    "default_3months",
    "ioi_36months",
    "ioi_3months",
    "valor_por_vencer",
    "valor_vencido",
    "valor_quitado",
    "quant_protestos",
    "valor_protestos",
    "quant_acao_judicial",
    "acao_judicial_valor",
    "participacao_falencia_valor",
    "dividas_vencidas_valor",
    "dividas_vencidas_qtd",
    "falencia_concordata_qtd",
    "valor_total_pedido",
    "month",
    "year",
    # Engineered features
    "ratio_vencido_quitado",
    "total_serasa",
]

CATEGORICAL_FEATURES = [
    "tipo_sociedade",       # 16 unique — low cardinality
    "opcao_tributaria",     # 4 unique  — low cardinality
    "atividade_principal",  # 203 unique — high cardinality, will be capped
    "forma_pagamento",      # 104 unique — high cardinality, will be capped
]

# Rare categories beyond this limit are grouped into "infrequent_sklearn"
MAX_CATEGORIES = 20


# ── Helpers ───────────────────────────────────────────────────────────────────

def replace_missing_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Replace literal 'missing' strings with NaN and cast numeric columns."""
    df = df.replace("missing", np.nan).copy()
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.

    - ratio_vencido_quitado: proportion of overdue payments relative to total paid history.
      High values indicate a client that is increasingly failing to pay.
    - total_serasa: aggregate count of all Serasa negative signals.
      Simplifies multiple sparse Serasa columns into a single risk score.
    """
    df = df.copy()
    df["ratio_vencido_quitado"] = df["valor_vencido"] / (df["valor_quitado"] + 1)
    df["total_serasa"] = (
        df["quant_protestos"].fillna(0)
        + df["quant_acao_judicial"].fillna(0)
        + df["dividas_vencidas_qtd"].fillna(0)
    )
    return df


def get_feature_columns(df: pd.DataFrame):
    """Return (num_cols, cat_cols) present in the dataframe, excluding target."""
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns and c != TARGET]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num_cols, cat_cols


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer with:
      - Numerical: median imputation → RobustScaler
      - Categorical: most-frequent imputation → OneHotEncoder
          max_categories caps high-cardinality columns, grouping rare values
          into a single 'infrequent' bucket to avoid feature explosion.
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            max_categories=MAX_CATEGORIES,
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_clean(data_path: str) -> pd.DataFrame:
    """Load the raw CSV, replace 'missing' strings with NaN and engineer features."""
    df = pd.read_csv(data_path, sep="\t", encoding="utf-8")
    df = replace_missing_strings(df)
    df = engineer_features(df)
    return df


def get_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Return an unfitted preprocessor configured for the given dataframe.
    Must be fitted only on X_train to avoid data leakage.
    """
    num_cols, cat_cols = get_feature_columns(df)
    return build_preprocessor(num_cols, cat_cols)


def get_X_y(df: pd.DataFrame):
    """Split dataframe into feature matrix X and target vector y."""
    num_cols, cat_cols = get_feature_columns(df)
    X = df[num_cols + cat_cols]
    y = df[TARGET]
    return X, y


