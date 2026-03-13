<div align="center">

```
██╗  ██╗███████╗ █████╗ ██████╗ ████████╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
██║  ██║██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
███████║█████╗  ███████║██████╔╝   ██║   ██║  ███╗██║   ██║███████║██████╔╝██║  ██║
██╔══██║██╔══╝  ██╔══██║██╔══██╗   ██║   ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
██║  ██║███████╗██║  ██║██║  ██║   ██║   ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
```

### Heart Failure Risk Predictor — Explainable ML · Clinical Decision Support

[![Python](https://img.shields.io/badge/Python-3.10+-00e5cc?style=flat-square&logo=python&logoColor=000)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-4d9fff?style=flat-square&logo=streamlit&logoColor=000)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-9b7ff5?style=flat-square&logo=scikitlearn&logoColor=000)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-00e5cc?style=flat-square)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-4d9fff?style=flat-square)](LICENSE)

*Centrale Casablanca · Coding Week March 2026 · Team 1 — k. Zerhouni*

</div>

---

## Overview

HeartGuard is a full-stack machine learning application for heart failure mortality risk prediction. Given 12 clinical patient parameters, it outputs a **calibrated probability score**, a **SHAP feature-attribution chart**, and **clinical flag warnings** — all inside a polished dark-themed Streamlit interface.

The system was built to demonstrate end-to-end ML engineering: from raw data ingestion and memory optimisation through model training, evaluation, and deployment, with explainability baked in from day one.

---

## Interface

The app is divided into three zones:

**Sidebar — Patient Input Panel**
All 12 clinical inputs are grouped into four labelled sections (Demographics, Cardiac Markers, Biochemistry, Comorbidities, Follow-Up). A single "Run Prediction" button triggers inference.

**Stats Row**
Four metric cards always visible at the top: training sample count, survivor count, deceased count, and model ROC-AUC — pulled live from the balanced dataset.

**Results Panel** *(appears after prediction)*
- Colour-coded risk banner (HIGH / MODERATE / LOW) with animated progress bar
- Half-donut gauge with segmented arcs and glow-fill up to the needle
- Side-by-side mortality vs survival probability cards
- Clinical flags: auto-detected biomarker anomalies with severity chips
- SHAP waterfall bar chart: top-8 features by absolute contribution, with glow bars and value labels
- Patient data summary table

---

## Project Structure

```
heartguard/
│
├── app/
│   └── app.py                    ← Streamlit interface (main entry point)
│
├── src/
│   ├── data_processing.py        ← load, optimise, impute, scale
│   ├── train_model.py            ← train RandomForest, save to models/
│   └── evaluate_model.py         ← metrics, cross-val, model comparison
│
├── data/
│   ├── heart_failure_clinical_records_dataset.csv   ← raw UCI dataset
│   └── nouvelle_dataset_equilibree.csv              ← SMOTE-balanced training set
│
├── models/
│   └── random_forest.pkl         ← serialised trained model (joblib)
│
├── tests/
│   └── test_data_processing.py   ← pytest suite (9 tests)
│
├── notebooks/
│   └── eda.ipynb                 ← EDA, SHAP summary plots, memory demo
│
├── docs/
│   └── prompt_engineering.md     ← 3-version prompt iteration log
│
├── .github/
│   └── workflows/
│       └── ci.yml                ← GitHub Actions: lint + test on push
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/your-org/heartguard.git
cd heartguard
pip install -r requirements.txt
```

### 2 — Train the model

```bash
python src/train_model.py
# → saves models/random_forest.pkl
# → prints Accuracy, ROC-AUC, F1
```

### 3 — Launch the app

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` — fill in patient parameters, click **Run Prediction**.

---

## Dataset

**Source**: [UCI Heart Failure Clinical Records](https://archive.uci.edu/dataset/519/heart+failure+clinical+records)  
**Patients**: 299 · **Features**: 12 clinical variables + `DEATH_EVENT` target  
**Class distribution**: 68% survived (0) · 32% deceased (1) — imbalanced

| Feature | Type | Description |
|---|---|---|
| `age` | float | Patient age in years |
| `anaemia` | binary | Decrease in red blood cells or haemoglobin |
| `creatinine_phosphokinase` | int | CPK enzyme level (mcg/L) |
| `diabetes` | binary | Diabetes diagnosis |
| `ejection_fraction` | int | % of blood leaving per heartbeat |
| `high_blood_pressure` | binary | Hypertension diagnosis |
| `platelets` | float | Platelet count (kiloplatelets/mL) |
| `serum_creatinine` | float | Serum creatinine level (mg/dL) |
| `serum_sodium` | int | Serum sodium level (mEq/L) |
| `sex` | binary | 1 = Male · 0 = Female |
| `smoking` | binary | Current smoker |
| `time` | int | Follow-up period (days) |

---

## Critical Questions

### 1 · Is the dataset balanced?

**No.** The raw dataset is imbalanced: ~68% survived (0), ~32% deceased (1).

**Solution applied**: SMOTE (Synthetic Minority Over-sampling Technique) on the **training split only**, never on the test set (to avoid data leakage). The resulting balanced CSV is saved as `nouvelle_dataset_equilibree.csv`.

**Impact of balancing**:

| Metric | Without SMOTE | With SMOTE | Δ |
|---|---|---|---|
| Accuracy | 80% | 87% | +7pp |
| F1-Score | 0.72 | 0.85 | +0.13 |
| Recall (deceased) | 0.61 | 0.84 | +0.23 |
| ROC-AUC | 0.86 | 0.91 | +0.05 |

---

### 2 · Which model performs best?

**XGBoost** achieves the highest performance across all five metrics:

| Model | ROC-AUC | Accuracy | F1 | Recall | Precision |
|---|---|---|---|---|---|
| **XGBoost ✅** | **0.91** | **87%** | **0.85** | **0.84** | **0.86** |
| LightGBM | 0.90 | 86% | 0.84 | 0.83 | 0.85 |
| Random Forest | 0.89 | 85% | 0.82 | 0.81 | 0.83 |
| Logistic Regression | 0.82 | 79% | 0.76 | 0.72 | 0.80 |

For a clinical decision-support tool, **recall** (sensitivity) is the most critical metric — missing a true death event is more dangerous than a false alarm. XGBoost achieves the highest recall (0.84) while maintaining strong precision.

The app ships with `random_forest.pkl` as the default serialised model (simpler to reproduce without XGBoost dependency), but the comparison table is shown in the model info expander.

---

### 3 · What are the top SHAP features?

Global feature importance computed with `shap.TreeExplainer` across the test set:

| Rank | Feature | Mean |SHAP| | Clinical Interpretation |
|---|---|---|---|
| 1 | `time` (follow-up) | 0.142 | Longer follow-up → patient survived longer → lower risk |
| 2 | `ejection_fraction` | 0.118 | Higher EF = stronger heart pump = lower risk |
| 3 | `serum_creatinine` | 0.097 | High creatinine signals renal failure, strongly predicts death |
| 4 | `age` | 0.063 | Older patients have higher baseline mortality |
| 5 | `serum_sodium` | 0.041 | Low sodium (hyponatremia) correlates with worse outcomes |

Features 6–12 (`diabetes`, `anaemia`, `sex`, etc.) contribute <0.02 each — far less predictive in this dataset.

---

### 4 · Prompt Engineering — How was Claude used?

The `optimize_memory` function was developed through **three prompt iterations**, documented in `docs/prompt_engineering.md`:

**Version 1 — Vague prompt**
> *"Write a function to reduce memory usage of a pandas DataFrame."*

Result: only handled `float64 → float32`. Missed integer optimisation and categorical conversion entirely. No print statement, no size validation.

**Version 2 — Structured with role**
> *"You are a data engineer. Write a production-quality Python function called `optimize_memory(df)` that reduces DataFrame memory by down-casting float64 to float32 and int64 to int32. Print before/after sizes."*

Result: correctly handled floats and integers, added informative print. Still missed low-cardinality object → category conversion.

**Version 3 — Few-shot with constraint**
> *"Here is the function signature and one example: [example for float32]. Now extend it to also handle int64→int32 with range-checking, and object columns with nunique/len < 0.5 → category. Return the optimised DataFrame."*

Result: production-ready code exactly matching the final `data_processing.py` implementation. The few-shot example locked in the pattern and the explicit constraint about cardinality prevented over-categorisation.

**Lesson**: specificity + role + constraint + example = reliable generation for utility functions.

---

## Data Pipeline

```
Raw CSV (299 rows · imbalanced)
       │
       ▼
  load_data()        ← pd.read_csv
       │
       ▼
  optimize_memory()  ← float64→32 · int64→32 · object→category
       │
       ▼
  handle_missing()   ← median (numeric) · mode (categorical)
       │
       ▼
  handle_outliers()  ← IQR clip on [CPK, platelets, creatinine…]
       │
       ▼
  train/test split   ← 80/20 stratified
       │
  SMOTE (train only)
       │
       ▼
  scale_features()   ← StandardScaler fit on train · transform both
       │
       ▼
  RandomForestClassifier(n_estimators=100, max_depth=10)
       │
       ▼
  joblib.dump → models/random_forest.pkl
```

---

## Tests

```bash
pytest tests/ -v
```

9 tests covering the full `data_processing.py` module:

| Test | What it verifies |
|---|---|
| `test_optimize_memory_float` | float64 → float32 conversion |
| `test_optimize_memory_int` | int64 → int32 conversion |
| `test_optimize_memory_category` | object → category for low-cardinality columns |
| `test_optimize_memory_reduction` | final size < original size |
| `test_handle_missing_numeric` | median imputation fills NaN in numeric columns |
| `test_handle_missing_categorical` | mode imputation fills NaN in object columns |
| `test_handle_outliers_clip` | IQR clipping keeps values within bounds |
| `test_handle_outliers_remove` | IQR removal drops outlier rows |
| `test_get_feature_target` | correct X / y split, no target in X |

---

## CI/CD

GitHub Actions runs on every push to `main` and on all pull requests:

```yaml
# .github/workflows/ci.yml
- flake8 src/ tests/        ← style lint
- pytest tests/ --tb=short  ← full test suite
```

---

## Requirements

```
streamlit>=1.32
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.0
shap>=0.45
joblib>=1.3
matplotlib>=3.8
imbalanced-learn>=0.12
pytest>=8.0
```

---

## Clinical Flags — Threshold Reference

| Biomarker | Warning | Critical | Source |
|---|---|---|---|
| Ejection Fraction | < 40% | < 30% | ESC Heart Failure Guidelines |
| Serum Creatinine | > 1.3 mg/dL | > 2.0 mg/dL | KDIGO CKD staging |
| Serum Sodium | < 135 mEq/L | < 130 mEq/L | SIADH / hyponatremia criteria |
| CPK | > 1200 mcg/L | — | Myocardial stress marker |
| Age | > 75 years | — | Elevated baseline risk |

---

<div align="center">

Built with 🫀 by **Team 1 — k. Zerhouni**  
Centrale Casablanca · Coding Week · March 2026

</div>
