# Post Renewal Churn Prediction

A machine learning system that predicts **customer churn after renewal** using behavioral signals from billings, customer care calls, and renewal negotiation calls. Built with **XGBoost**, the model identifies at-risk customers within a 4-week post-renewal window, enabling proactive retention strategies.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Data Sources](#data-sources)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Notebook Pipeline](#notebook-pipeline)
- [Design Decisions](#design-decisions)
- [Tech Stack](#tech-stack)

---

## Project Overview

This project focuses on **post-renewal churn prediction** — identifying customers who are likely to churn shortly after their renewal date. The model targets a specific cohort: customers whose deal closed within **4 weeks (28 days) after** their `Prospect_Renewal_Date`.

### Problem Statement

With a baseline churn rate of ~9–13% (varying by year), simply predicting "no churn" for everyone yields 90%+ accuracy but provides zero business value. This project builds a model that **prioritizes recall and precision on the minority class** (churners), using PR-AUC as the primary evaluation metric.

### Approach

1. **Exploratory Data Analysis** across 4 data sources (billings, emails, customer care calls, renewal calls)
2. **Hypothesis Testing** — 9 formal statistical hypotheses validated
3. **Feature Engineering** — 40+ features across 5 categories, with careful leakage prevention
4. **XGBoost Modeling** — Baseline → Tuning → Final evaluation with SHAP explainability
5. **Risk Tiering** — Customers classified into Critical / High / Medium / Low risk tiers

---

## Key Findings

### Hypothesis Test Results

| # | Hypothesis | Feature | Result | Churn Lift |
|---|-----------|---------|--------|-----------|
| H1 | Zero anchorings → higher churn | `anchoring_zero` | ✅ Significant | 2.50× |
| H2 | Auto-renewal OFF → higher churn | `auto_renewal_off` | ✅ Significant | 5.21× |
| H3 | Price increase → higher churn | `price_increase_10pct` | ✅ Significant | 0.75× (inverted!) |
| H4 | Leave signal → higher churn | `any_leave_signal` | ✅ Significant | 3.17× |
| H5 | Competitor mention → higher churn | `any_competitor_signal` | ✅ Significant | 1.27× |
| H6 | Lower tenure → higher churn | `Tenure_Years` | ✅ Significant | — |
| H7 | Unknown payment → higher churn | `payment_unknown` | ❌ Not significant | — |
| H8 | Negative email sentiment → higher churn | `email_pct_negative` | ✅ Significant | — |
| H9 | More negative flags → higher churn | `total_negative_flags` | ✅ Significant | — |

### Top Churn Drivers (by SHAP importance)

- **`auto_renewal_off`** — Customers without auto-renewal are 5.2× more likely to churn
- **`any_leave_signal`** — Explicit leave/cancel mentions in calls are a 3.2× risk multiplier
- **`anchoring_zero`** — Zero product anchorings indicate 2.5× higher churn risk
- **`cc_ever_suggest_leave`** — Customer care calls where leaving was discussed
- **`rc_competitor_mentioned`** — Competitor mentions during renewal calls

---

## Data Sources

Four raw CSV datasets in `data/raw/`:

| Source | File | Rows | Columns | Description |
|--------|------|------|---------|-------------|
| Billings | `billings.csv` | 122,082 | 59 | Backbone table — one row per customer per renewal year |
| Customer Care Calls | `cc_calls.csv` | 32,882 | 33 | Support/care call records with sentiment and issue flags |
| Emails | `emails.csv` | 123,389 | 27 | CRM email interactions (⚠️ excluded from model) |
| Renewal Calls | `renewal_calls.csv` | 186,534 | 41 | Renewal negotiation call records |

> **⚠️ Email Data Exclusion:** Email data has a 1-year offset (`emails.year = Renewal_Year + 1`), meaning emails arrive *after* the renewal decision. Including them would introduce **temporal leakage**, so they are intentionally excluded from the feature set.

> **⚠️ Leakage Columns:** Several columns in `renewal_calls.csv` are explicitly excluded: `Membership_Renewal_Decision`, `Churn_Category`, `Desire_To_Cancel`, `Customer_Renewal_Response_Category` — these encode the outcome directly.

---

## Feature Engineering

40+ features organized into **5 categories**, built via `src/features/builder.py`:

### 1. Billing Structural
`band_enc`, tenure buckets (`tenure_0_1`, `tenure_2_3`, `tenure_4_7`, `tenure_8plus`), `Current_Anchorings`, `Connection_Qty`, `Discount_Amount`, scoring features

### 2. Billing Derived
`is_first_year`, `auto_renewal_off`, `anchoring_zero/1plus/3plus`, `band_downgraded/upgraded`, `low_release_score`

### 3. Customer Care Call Behavioral
`cc_call_count`, `cc_ever_suggest_leave`, `cc_ever_hardship`, `cc_ever_complained`, `cc_ever_platform_issues`, `cc_ever_pricing`, `cc_ever_refund`, `cc_pct_dissatisfied`, `cc_avg_sentiment_delta`

### 4. Renewal Call Signals
`rc_call_count`, `rc_discount_requested`, `rc_price_discussed`, `rc_competitor_mentioned`, `rc_switching_intent`, `rc_rescheduled`, `rc_agent_flagged`, `rc_high_friction`, `rc_agent_chased`

### 5. Cross-File Composite
`any_leave_signal`, `any_competitor_signal`, `any_financial_hardship`, `any_complaint`, `total_negative_flags`, `total_contact_count`, `has_no_behavioural_data`, `high_friction_score`, `critical_risk_billing`

---

## Model Architecture

### XGBoost Classifier

| Parameter | V1 Baseline | V2 Tuned (Final) |
|-----------|-------------|-------------------|
| `n_estimators` | 400 | 600 |
| `max_depth` | 5 | 3 |
| `learning_rate` | 0.04 | 0.03 |
| `scale_pos_weight` | 9.5 | 5.0 |

- **Training data:** `Renewal_Year = 2024`
- **Test data:** `Renewal_Year = 2025`
- **Cross-validation:** Stratified 5-fold (tracking PR-AUC, ROC-AUC, F1)
- **Threshold tuning:** Maximize F1 subject to min_precision ≥ 0.25 and min_recall ≥ 0.50
- **Risk tiering:** Quantile-based — Critical (top 10%), High (70–90th percentile), Medium (40–70th), Low (bottom 40%)

### Saved Models

| File | Description |
|------|-------------|
| `churn_model_final.json` | Final production model (V2 tuned) |
| `churn_model_v1_baseline.json` | V1 baseline for comparison |
| `churn_model_v2_tuned.json` | V2 after hyperparameter tuning |
| `best_params.json` | Tuned hyperparameters |

---

## Project Structure

```
post_renewal_churn_prediction/
├── configs/
│   ├── config.yaml
│   └── features.yaml
├── data/
│   └── raw/
│       ├── billings.csv
│       ├── cc_calls.csv
│       ├── emails.csv
│       └── renewal_calls.csv
├── models_saved/
│   ├── best_params.json
│   ├── churn_model_final.json
│   ├── churn_model_v1_baseline.json
│   ├── churn_model_v1_baseline_params.json
│   ├── churn_model_v2_tuned.json
│   └── churn_model_v2_tuned_params.json
├── notebooks/
│   ├── 01_eda/                      # EDA per data source (4 notebooks)
│   ├── 02_integration/              # Joined data EDA
│   ├── 03_hypotheses/               # 9 hypothesis tests
│   ├── 04_feature_engineering/      # Feature building pipeline
│   └── 05_modeling/                 # Baseline → Tuning → Final eval
├── reports/
│   ├── churn_predictions_2025.csv   # Final predictions output
│   ├── hypothesis_results.csv       # Statistical test results
│   └── figures/                     # 24 evaluation/SHAP plots
├── src/
│   ├── data/
│   │   ├── cleaner.py               # Data cleaning (430 lines)
│   │   └── loader.py                # Data loading utilities (107 lines)
│   ├── features/
│   │   └── builder.py               # Feature engineering pipeline (317 lines)
│   ├── models/
│   │   ├── evaluate.py              # Evaluation metrics & plots (285 lines)
│   │   └── train.py                 # XGBoost training & CV (240 lines)
│   └── viz/
│       └── plots.py
├── tests/
│   └── test_cleaner.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Source Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `src/data/loader.py` | 107 | Data loading with year mapping documentation |
| `src/data/cleaner.py` | 430 | Type fixes, value standardization for all 4 sources |
| `src/features/builder.py` | 317 | Full feature engineering pipeline with cohort definition |
| `src/models/train.py` | 240 | XGBoost training, cross-validation, threshold tuning, risk tiers |
| `src/models/evaluate.py` | 285 | Metrics, PR/ROC curves, confusion matrix, calibration, SHAP |

---

## Setup Instructions

### Prerequisites

- **Python**: 3.14.3
- **pip**: 25.3

### Installation

```bash
# Clone the repository
git clone https://github.com/rahulparaselli/post_renewal_churn_prediction.git
cd post_renewal_churn_prediction

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place the four raw CSV files in `data/raw/`:
- `billings.csv`
- `cc_calls.csv`
- `emails.csv`
- `renewal_calls.csv`

---

## Usage

### Running the Notebook Pipeline

Execute the notebooks in order:

```
notebooks/01_eda/           → Exploratory analysis per data source
notebooks/02_integration/   → Cross-source data integration EDA
notebooks/03_hypotheses/    → Statistical hypothesis testing
notebooks/04_feature_engineering/ → Feature building
notebooks/05_modeling/      → Model training, tuning, and evaluation
```

### Using Source Modules Directly

```python
from src.data.loader import load_all
from src.data.cleaner import clean_all
from src.features.builder import build_cohort_features

# Load and clean data
billings, cc_calls, emails, renewal_calls = load_all()
billings, cc_calls, emails, renewal_calls = clean_all(billings, cc_calls, emails, renewal_calls)

# Build features for a specific renewal year
X, y = build_cohort_features(billings, cc_calls, renewal_calls, renewal_year=2025)
```

### Loading the Trained Model

```python
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model('models_saved/churn_model_final.json')

# Predict churn probabilities
churn_probs = model.predict_proba(X)[:, 1]
```

---

## Notebook Pipeline

### Phase 1: EDA (`01_eda/`)
- `01_billings_eda.ipynb` — Outcome distributions, year patterns, column profiling, null analysis
- `02_emails_eda.ipynb` — Year offset discovery, Time_to_Renewal windows, flag distributions
- `03_cc_calls_eda.ipynb` — Call patterns, sentiment analysis, messy data identification
- `04_renewal_calls_eda.ipynb` — Direction patterns, discount/competitor analysis, leakage detection

### Phase 2: Integration (`02_integration/`)
- `01_data_integration_eda.ipynb` — Multi-source joins, feature correlations, year mapping validation

### Phase 3: Hypothesis Testing (`03_hypotheses/`)
- `01_hypothesis_testing.ipynb` — 9 formal statistical tests with visualizations

### Phase 4: Feature Engineering (`04_feature_engineering/`)
- `01_feature_engineering.ipynb` — Cohort construction and feature pipeline execution

### Phase 5: Modeling (`05_modeling/`)
- `01_baseline_model.ipynb` — V1 baseline XGBoost with cross-validation
- `02_experiments.ipynb` — Hyperparameter tuning (scale_pos_weight experiments)
- `03_final_model_evaluation.ipynb` — Final evaluation, SHAP analysis, calibration, predictions export

---

## Design Decisions

1. **Post-renewal cohort window (28 days):** Only customers whose `Closed_Date` falls within 4 weeks after `Prospect_Renewal_Date` are included. This focuses predictions specifically on the post-renewal period.

2. **Email data excluded from features:** Since `emails.year = Renewal_Year + 1`, emails arrive *after* the renewal decision — using them would be temporal leakage.

3. **No accuracy metric:** With ~9.5% churn rate, accuracy is misleading (predicting all "stays" = 90.5% accuracy). The primary metric is **PR-AUC**, with constraints of **Recall ≥ 0.50** and **Precision ≥ 0.25**.

4. **Aggressive leakage prevention:** Multiple columns carefully excluded — `Total_Net_Paid` ($0 for churned), `Payment_Method` (UNKNOWN for churned), `Total_Renewal_Score_New` (−0.66 correlation with outcome), `Status_Scores`, `Tenure_Scores`.

5. **No rows dropped during cleaning:** Cleaning only fixes data types and standardizes values. `"Not Discussed"` in Yes/No columns → `NaN` (absence of mention ≠ confirmed No). Row filtering happens downstream in notebooks.

6. **Temporal train/test split:** Train on `Renewal_Year=2024`, test on `Renewal_Year=2025` — mimics real-world deployment where the model is trained on historical data and predicts on future cohorts.

---

## Tech Stack

- **Python** 3.14.3
- **Machine Learning:** XGBoost, scikit-learn
- **Explainability:** SHAP
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Notebooks:** Jupyter

---

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Churn predictions | `reports/churn_predictions_2025.csv` | Final model predictions for 2025 cohort |
| Hypothesis results | `reports/hypothesis_results.csv` | Statistical test results for 9 hypotheses |
| Evaluation plots | `reports/figures/` | 24 PNG plots (PR/ROC curves, SHAP, calibration, etc.) |
| Final model | `models_saved/churn_model_final.json` | Production-ready XGBoost model |

---

## License

This project is for academic/internal use.