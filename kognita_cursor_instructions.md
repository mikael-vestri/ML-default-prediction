# 🧠 Data Science Case - Kognita Lab (X-Health Default Prediction)

## 🎯 Objective
Build a complete, production-level Machine Learning pipeline to predict the probability of default for B2B clients based on historical and external credit data.

---

## 📁 Project Structure

The project MUST follow this structure:

```
/data
/notebooks
/src
/models
/artifacts
/reports
README.md
requirements.txt
```

---

## ⚙️ General Guidelines

- Code must be clean, modular, and reusable.
- ALL code comments MUST be written in English.
- Follow best practices for real-world ML projects.
- Ensure full reproducibility (fixed seeds, deterministic behavior where possible).
- Avoid hardcoding values unless strictly necessary.

---

## 🔄 Execution Workflow (VERY IMPORTANT)

For EACH step:

1. First, propose a clear action plan
2. Explain what will be done and why
3. ASK for user approval:
   - "Can I proceed?"
   - "Would you like to change anything?"
4. Only proceed after approval

---

## 📊 Step 1 — Exploratory Data Analysis (EDA)

Create a notebook:

/notebooks/01_eda.ipynb

Tasks:
- Load dataset (handle sep='\t' and encoding)
- Replace "missing" values with proper NaNs
- Analyze:
  - distributions
  - correlations
  - missing values
  - target imbalance
- Generate insights (business-oriented)

---

## 🧼 Step 2 — Data Preprocessing

Create reusable code in /src/preprocessing.py

Tasks:
- Missing value treatment
- Encoding categorical variables
- Feature scaling (if needed)
- Feature selection (optional but encouraged)

---

## 🤖 Step 3 — Model Training

Create notebook:

/notebooks/02_modeling.ipynb

Tasks:
- Train/test split
- Handle class imbalance (if needed)
- Train baseline model (e.g., Logistic Regression)
- Train advanced model (e.g., Random Forest, XGBoost)
- Evaluate using:
  - ROC-AUC
  - Precision / Recall
  - Confusion Matrix

---

## 📦 Step 4 — Model Saving

- Save trained model to /models
- Save preprocessing artifacts to /artifacts
- Ensure inference can reuse them

---

## 🔮 Step 5 — Prediction Function

Create notebook:

/notebooks/03_inference.ipynb

Also implement reusable function in /src/predict.py

Function must:

def predict(input_dict: dict) -> dict:
    """
    Receives input data and returns default prediction.
    """

Return format:
{"default": 0} or {"default": 1}

---

## 📘 README.md (MANDATORY & EVOLVING)

You MUST create and continuously update the README.md.

It should include:

- Business understanding
- Data description
- EDA insights
- Modeling decisions
- Evaluation results
- How to run the project
- Example of prediction usage

---

## 🧠 Business Perspective

At every stage, explain:

- Why this matters for the finance team
- How the model could be used in real decision-making
- Trade-offs (false positives vs false negatives)

---

## 🧪 Best Practices to Follow

- Use functions instead of repeated code
- Keep notebooks clean and readable
- Separate experimentation from reusable logic
- Log important decisions
- Document assumptions clearly

---

## 🚨 Important Notes

- Focus is NOT only on model performance
- Focus is on:
  - reasoning
  - code quality
  - structure
  - clarity

---

## ✅ Final Deliverables

- 3 notebooks (EDA, Modeling, Inference)
- Clean project structure
- requirements.txt
- Saved model + artifacts
- Complete README.md
