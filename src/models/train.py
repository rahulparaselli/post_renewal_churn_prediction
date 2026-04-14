"""
src/models/train.py
────────────────────
Model training functions for XGBoost churn prediction.

Why XGBoost:
  - Best performance on tabular mixed-type data
  - Handles class imbalance via scale_pos_weight
  - Built-in feature importance
  - Works well with the ~30-50 features we have

Why NOT accuracy as a metric:
  - 9.5% churn rate means predicting "everyone stays" = 90.5% accuracy
  - We care about Precision-Recall AUC and catching churners (Recall)
  - Target: Recall >= 0.65 at a threshold that keeps Precision >= 0.30
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, f1_score,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models_saved"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.04,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 9.5,   # ~90.5% retained / 9.5% churned
    "eval_metric":      "aucpr",
    "use_label_encoder": False,
    "random_state":     42,
    "n_jobs":           -1,
}


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict = None,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Stratified k-fold cross-validation.

    Stratified = each fold has the same churn rate as the full dataset.
    This is important with 9.5% minority class.

    Returns dict with mean and std of key metrics.
    """
    params = params or DEFAULT_PARAMS
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pr_aucs, roc_aucs, f1s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, proba)
        roc_auc = roc_auc_score(y_val, proba)

        # find best threshold by F1
        prec, rec, thresholds = precision_recall_curve(y_val, proba)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
        best_thresh = thresholds[np.argmax(f1_scores[:-1])]
        preds = (proba >= best_thresh).astype(int)
        f1 = f1_score(y_val, preds)

        pr_aucs.append(pr_auc)
        roc_aucs.append(roc_auc)
        f1s.append(f1)

        if verbose:
            print(f"  Fold {fold+1}: PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  F1={f1:.4f}")

    results = {
        "pr_auc_mean":  np.mean(pr_aucs),
        "pr_auc_std":   np.std(pr_aucs),
        "roc_auc_mean": np.mean(roc_aucs),
        "roc_auc_std":  np.std(roc_aucs),
        "f1_mean":      np.mean(f1s),
        "f1_std":       np.std(f1s),
    }

    if verbose:
        print(f"\n  CV Summary ({n_splits}-fold):")
        print(f"  PR-AUC  = {results['pr_auc_mean']:.4f} ± {results['pr_auc_std']:.4f}")
        print(f"  ROC-AUC = {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
        print(f"  F1      = {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN FINAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None,
    model_name: str = "churn_model_v1",
) -> XGBClassifier:
    """
    Train the final model on the full training set.

    If X_val/y_val are provided (test cohort), uses them as early stopping set.
    Saves the model to models_saved/.
    """
    params = params or DEFAULT_PARAMS.copy()

    model = XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    # save model
    save_path = MODELS_DIR / f"{model_name}.json"
    model.save_model(str(save_path))
    print(f"Model saved → {save_path}")

    # save params
    params_path = MODELS_DIR / f"{model_name}_params.json"
    save_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
    with open(params_path, "w") as f:
        json.dump(save_params, f, indent=2)

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  FIND OPTIMAL THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.25,
    min_recall: float = 0.50,
) -> float:
    """
    Find the probability threshold that maximises F1 subject to
    minimum precision and recall constraints.

    Why this matters: the default 0.5 threshold is wrong for imbalanced data.
    With 9.5% churn rate, almost nothing scores above 0.5 — you need a lower
    threshold. This function finds the best threshold for your business tradeoff.

    min_precision = 0.25 means: of all customers you flag as high risk,
                   at least 25% should actually churn (acceptable false alarm rate)
    min_recall    = 0.50 means: you want to catch at least 50% of all churners
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)

    best_f1 = 0
    best_thresh = 0.3  # default fallback

    for p, r, t in zip(prec[:-1], rec[:-1], thresholds):
        if p >= min_precision and r >= min_recall:
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

    return float(best_thresh)


# ─────────────────────────────────────────────────────────────────────────────
#  RISK TIERING
# ─────────────────────────────────────────────────────────────────────────────

def assign_risk_tier(proba: np.ndarray) -> np.ndarray:
    """
    Convert probabilities to business-friendly risk tiers.

    Tiers are based on quantiles of the churn probability distribution
    rather than fixed thresholds — this ensures every tier has customers.

    Tier 1 — Critical Risk  : top 10% probability
    Tier 2 — High Risk      : 70th–90th percentile
    Tier 3 — Medium Risk    : 40th–70th percentile
    Tier 4 — Low Risk       : bottom 40%
    """
    p90 = np.percentile(proba, 90)
    p70 = np.percentile(proba, 70)
    p40 = np.percentile(proba, 40)

    tiers = np.where(
        proba >= p90, "Critical",
        np.where(
            proba >= p70, "High",
            np.where(
                proba >= p40, "Medium",
                "Low"
            )
        )
    )
    return tiers
