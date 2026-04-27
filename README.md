# X-Health Default Prediction — ML Pipeline

A production-level Machine Learning pipeline to predict the **probability of default** for B2B clients of X-Health, a company that sells health devices on credit.

---

## Business Understanding

X-Health operates in the B2B health electronics market. Clients place orders and pay in the future — either upfront or in installments. The finance team has observed a growing number of **non-payments (defaults)**, which directly impacts revenue and cash flow.

**Goal:** Build a model capable of predicting, at the moment of a new order, whether a given client is likely to default — enabling the finance team to take preventive action (e.g., request a guarantee, reduce credit limit, or escalate for manual review).

**Key trade-off:**
- **False Negative** (missed default) → financial loss — higher cost
- **False Positive** (blocking a good client) → lost sale — lower cost

The model must be calibrated to **minimize False Negatives** while keeping False Positives at an acceptable level. The optimal threshold (0.65) was determined by maximizing the F1-score on the test set.

---

## Project Structure

```
/data                   ← Dataset
/notebooks
    01_eda.ipynb        ← Exploratory Data Analysis
    02_modeling.ipynb   ← Training, evaluation, tuning and model saving
    03_inference.ipynb  ← Prediction function demo
/src
    preprocessing.py    ← Reusable preprocessing pipeline + feature engineering
    predict.py          ← Prediction functions (predict, predict_proba)
/models
    pipeline_xgb.joblib        ← Original XGBoost pipeline
    pipeline_xgb_tuned.joblib  ← Tuned pipeline (used for inference)
/artifacts              ← Additional artifacts
/reports                ← Exported plots
requirements.txt
README.md
```

---

## Dataset Description

- **Path:** `data/dataset_2021-5-26-10-14.csv`
- **Format:** TSV (`sep='\t'`, `encoding='utf-8'`)
- **Rows:** ~117,000 purchase events
- **Missing values:** represented as the string `"missing"`

| Column | Description |
|--------|-------------|
| `default_3months` | Number of defaults in the last 3 months (X-Health internal) |
| `ioi_36months` / `ioi_3months` | Average interval between orders (days) |
| `valor_por_vencer` | Total upcoming payments (R$) |
| `valor_vencido` | Total overdue payments (R$) |
| `valor_quitado` | Total paid historically (R$) |
| `quant_protestos` | Number of Serasa title protests |
| `valor_protestos` | Total value of protests (R$) |
| `quant_acao_judicial` | Number of lawsuits (Serasa) |
| `acao_judicial_valor` | Total value of lawsuits (R$) |
| `participacao_falencia_valor` | Bankruptcy exposure value (R$) |
| `dividas_vencidas_valor` | Total overdue debts — Serasa (R$) |
| `dividas_vencidas_qtd` | Number of overdue debts — Serasa |
| `falencia_concordata_qtd` | Number of bankruptcies |
| `tipo_sociedade` | Company type |
| `opcao_tributaria` | Tax regime |
| `atividade_principal` | Main business activity |
| `forma_pagamento` | Payment terms agreed for the order |
| `valor_total_pedido` | Total order value (R$) |
| `month` / `year` | Order date |
| `default` | **Target:** 0 = paid on time, 1 = defaulted |

---

## EDA Insights

| # | Insight | Business Implication |
|---|---------|----------------------|
| 1 | **Class imbalance** — ~16.7% default rate | Standard accuracy is misleading; ROC-AUC and Recall are better metrics |
| 2 | **Financial variables are heavily right-skewed** | RobustScaler chosen over StandardScaler |
| 3 | **Serasa variables are zero for 85%+ of clients** | Any non-zero value is already a risk signal |
| 4 | **Temporal pattern** | Default rate varies over time — `month`/`year` add predictive value |
| 5 | **`atividade_principal` has 203 unique values** | `max_categories=20` applied to prevent feature explosion |

---

## Key Business Insight — Internal History Beats Serasa

> **The strongest predictor of default is not Serasa data — it is the client's own history within X-Health.**

Analysis of the model's highest-confidence predictions revealed that clients flagged with **99%+ probability of default** often had zero Serasa alerts. What drove the prediction was:

- `default_3months` — already defaulted multiple times in the last 3 months
- `ioi_3months` — placing orders at a very high frequency (every few days)
- `valor_por_vencer` — large amounts still due while continuing to buy

**Practical implication:** Before consulting Serasa, the finance team should check internal history. A client with `default_3months > 0` is already a strong red flag, regardless of their external credit score.

---

## Modeling Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Missing value imputation | Median (numeric), Mode (categorical) | Robust to outliers common in financial data |
| Scaling | RobustScaler | Uses IQR — not affected by extreme values |
| Encoding | OneHotEncoder with `max_categories=20` | Prevents feature explosion from high-cardinality columns |
| Imbalance handling | Class weighting (`scale_pos_weight=5`) | No synthetic data — every sample is real and transparent |
| Data leakage prevention | Preprocessor fitted only on `X_train` | Full sklearn Pipeline ensures test set is never seen during fitting |
| Feature engineering | `ratio_vencido_quitado`, `total_serasa` | Capture meaningful business signals from existing variables |
| Feature selection | `SelectFromModel` (importance > 0.01) | Removes low-signal features to reduce noise and improve generalization |
| Baseline model | Logistic Regression | Linear, interpretable, fast — used as reference |
| **Final model** | **XGBoost + Hyperparameter Tuning** | Best ROC-AUC; handles non-linearity and imbalance natively |
| Tuning strategy | RandomizedSearchCV + StratifiedKFold (5 folds) | Efficient search while preserving class balance across folds |
| **Threshold** | **0.65** | Maximizes F1-score on test set |

---

## Evaluation Results

| Metric | Logistic Regression | XGBoost Original | XGBoost Tuned |
|--------|--------------------|--------------------|----------------|
| ROC-AUC | 0.7095 | 0.8843 | **0.9160** |
| Precision (Default) | 0.29 | 0.47 | **0.61** |
| Recall (Default) | 0.60 | 0.78 | **0.78** |
| F1 (Default) | 0.39 | 0.59 | **0.69** |
| Accuracy | 0.69 | 0.82 | **0.88** |

The tuned model maintained the same recall (78% of real defaults captured) while significantly improving precision (0.47 → 0.61) — fewer false alarms with no loss in default detection.

Key plots saved in `/reports`:
- `roc_pr_curves.png` — ROC and Precision-Recall curves (LR vs XGBoost)
- `confusion_matrices.png` — Confusion matrices
- `feature_importance.png` — Top 20 XGBoost features
- `roc_tuned_vs_original.png` — Original vs Tuned ROC curve
- `threshold_tuned_vs_original.png` — Threshold analysis comparison

---

## Model Explainability — SHAP Analysis

To ensure the model's decisions can be understood and justified, SHAP (SHapley Additive exPlanations) analysis was applied to the final tuned XGBoost pipeline.

### Global Feature Importance (Bar Plot)

Mean absolute SHAP values across 1000 test samples. Top drivers:

| Rank | Feature | Interpretation |
|------|---------|----------------|
| 1 | `ioi_3months` | High order frequency in last 3 months → high risk |
| 2 | `ioi_36months` | Long-term order pattern — consistent signal |
| 3 | `month` | Seasonal effect captured by the model |
| 4 | `valor_quitado` | Higher total paid history → lower risk |
| 5 | `default_3months` | Recent default history → direct risk flag |
| 7 | `ratio_vencido_quitado` | Engineered feature — validates feature engineering decision |

### Direction of Impact (Beeswarm Plot)

- Clients with **high `default_3months`** (red dots) cluster strongly to the right → always increases default risk
- Clients with **high `valor_quitado`** (red dots) cluster to the left → good payment history reduces risk
- **Serasa features** (`valor_protestos`, `quant_protestos`) have scattered, low-magnitude impact — confirming internal history is more predictive

### Individual Explanation (Waterfall Plot)

For the highest-confidence defaulter in the test set (`probability ≈ 100%`):
- The model started from a base prediction of `E[f(x)] = 0.155`
- `valor_protestos = 105.013` pushed the score by **+4.23**
- `ioi_3months = 0.259` added **+2.22**
- `default_3months = 1` confirmed with **+1.41**
- Final log-odds output: `f(x) = 14.35` → near-certain default

> SHAP confirms that model decisions are **traceable and explainable** at the individual client level — a key requirement for responsible credit decisioning.

---

## How to Run

### 1. Create and activate virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate       # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run EDA

```
notebooks/01_eda.ipynb
```

### 4. Train, tune and save the model

```
notebooks/02_modeling.ipynb
```

> Generates `models/pipeline_xgb.joblib` and `models/pipeline_xgb_tuned.joblib`

### 5. Run inference demo

```
notebooks/03_inference.ipynb
```

---

## Prediction Function Usage

```python
from src.predict import predict, predict_proba

# Basic prediction (default threshold = 0.65)
result = predict({
    "default_3months": 0,
    "ioi_3months": 28.0,
    "valor_vencido": 0.0,
    "valor_total_pedido": 25000.0,
})
# Output: {"default": 0}

# With probability — useful for inspecting model confidence
result = predict_proba({
    "default_3months": 4,
    "ioi_3months": 6.21,
    "valor_por_vencer": 77610.74,
    "valor_total_pedido": 16194.23,
})
# Output: {"default": 1, "probability": 0.9981}
```

**Adjusting the threshold:**
```python
predict(order, threshold=0.4)   # more conservative — catches more defaults
predict(order, threshold=0.8)   # more permissive — fewer false alarms
```

---

## Reproducibility

- Random seed fixed at `SEED = 42` across all steps
- Full tuned pipeline saved in `models/pipeline_xgb_tuned.joblib`
- All dependencies listed in `requirements.txt`
