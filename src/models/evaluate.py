"""
src/models/evaluate.py
───────────────────────
Model evaluation functions.

Why NOT accuracy:
  With 9.5% churn rate, a model that predicts everyone stays
  gets 90.5% accuracy and catches ZERO churners. Useless.

Metrics we care about:
  - PR-AUC   : area under precision-recall curve — best for imbalanced data
  - ROC-AUC  : discrimination ability overall
  - Recall   : what fraction of actual churners did we catch?
  - Precision : of customers we flagged, what fraction actually churned?
  - F1       : harmonic mean of precision and recall

Business translation:
  Recall=0.65 means: we catch 65 out of every 100 churners.
  Precision=0.30 means: of every 10 customers we flag as high risk, 3 actually churn.
  That is an acceptable outreach rate — it is much better than random.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score,
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FULL EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = None,
    model_name: str = "model",
    save_plots: bool = True,
) -> dict:
    """
    Complete model evaluation with all metrics and plots.

    If threshold is None, uses optimal threshold from PR curve.
    Returns dict of all metrics.
    """
    proba = model.predict_proba(X_test)[:, 1]

    # ── find threshold ────────────────────────────────────────────────────────
    if threshold is None:
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, proba)
        f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
        threshold = thresh_arr[np.argmax(f1_arr[:-1])]

    preds = (proba >= threshold).astype(int)

    # ── core metrics ──────────────────────────────────────────────────────────
    metrics = {
        "threshold":   float(threshold),
        "pr_auc":      float(average_precision_score(y_test, proba)),
        "roc_auc":     float(roc_auc_score(y_test, proba)),
        "precision":   float(precision_score(y_test, preds, zero_division=0)),
        "recall":      float(recall_score(y_test, preds, zero_division=0)),
        "f1":          float(f1_score(y_test, preds, zero_division=0)),
        "n_flagged":   int(preds.sum()),
        "n_actual_churn": int(y_test.sum()),
        "n_test":      len(y_test),
    }
    metrics["caught_churners"] = int((preds * y_test).sum())
    metrics["missed_churners"] = metrics["n_actual_churn"] - metrics["caught_churners"]

    # ── print summary ─────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"Model Evaluation — {model_name}")
    print("=" * 50)
    print(f"  Threshold used:    {metrics['threshold']:.3f}")
    print(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1:                {metrics['f1']:.4f}")
    print()
    print(f"  Total test customers:  {metrics['n_test']:,}")
    print(f"  Actual churners:       {metrics['n_actual_churn']:,}")
    print(f"  Customers flagged:     {metrics['n_flagged']:,}")
    print(f"  Churners caught:       {metrics['caught_churners']:,} ({metrics['recall']*100:.1f}%)")
    print(f"  Churners missed:       {metrics['missed_churners']:,}")
    print()
    print(classification_report(y_test, preds, target_names=["Retained","Churned"]))

    # ── plots ─────────────────────────────────────────────────────────────────
    if save_plots:
        fig = _plot_evaluation(y_test, proba, preds, threshold, model_name)
        save_path = FIGURES_DIR / f"{model_name}_evaluation.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Plot saved → {save_path}")

    return metrics


def _plot_evaluation(y_true, proba, preds, threshold, model_name):
    """Four-panel evaluation plot."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # ── 1. Precision-Recall Curve ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    prec, rec, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    ax1.plot(rec, prec, color="#185FA5", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax1.axhline(y_true.mean(), color="grey", ls="--", label=f"Random = {y_true.mean():.3f}")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # ── 2. ROC Curve ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)
    ax2.plot(fpr, tpr, color="#0F6E56", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax2.plot([0,1], [0,1], color="grey", ls="--", label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    # ── 3. Probability Distribution ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(proba[y_true==0], bins=50, alpha=0.6, color="#0F6E56", label="Retained (0)", density=True)
    ax3.hist(proba[y_true==1], bins=50, alpha=0.6, color="#E63946", label="Churned (1)", density=True)
    ax3.axvline(threshold, color="black", ls="--", lw=2, label=f"Threshold = {threshold:.3f}")
    ax3.set_xlabel("Predicted churn probability")
    ax3.set_ylabel("Density")
    ax3.set_title("Score Distribution by True Label")
    ax3.legend()

    # ── 4. Confusion Matrix ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(y_true, preds)
    im = ax4.imshow(cm, cmap="Blues")
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Pred: Retained", "Pred: Churned"])
    ax4.set_yticklabels(["True: Retained", "True: Churned"])
    ax4.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax4)

    fig.suptitle(f"Model Evaluation — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 25,
    model_name: str = "model",
    save: bool = True,
) -> pd.DataFrame:
    """Plot and return feature importances (gain-based)."""
    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in importance.items()
    ]).sort_values("importance", ascending=False)

    # map internal feature names back if needed
    if feature_names and len(feature_names) == len(model.feature_importances_):
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
    else:
        fi_df = imp_df

    top = fi_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.32)))
    colors = ["#E63946" if i < 5 else "#185FA5" if i < 15 else "#888"
              for i in range(len(top))]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_title(f"Feature Importance (top {top_n}) — {model_name}")
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()

    if save:
        save_path = FIGURES_DIR / f"{model_name}_feature_importance.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

    return fi_df


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
    model_name: str = "model",
):
    """
    Calibration plot: does a predicted probability of 0.3 mean 30% actually churn?
    A well-calibrated model falls on the diagonal.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_actual = [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() > 0:
            bin_means.append(proba[mask].mean())
            bin_actual.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(bin_means, bin_actual, "o-", color="#185FA5", lw=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual churners")
    ax.set_title(f"Calibration Plot — {model_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    save_path = FIGURES_DIR / f"{model_name}_calibration.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  SCORE AT MULTIPLE THRESHOLDS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def threshold_analysis(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """
    Show precision, recall, F1, and number flagged at multiple thresholds.
    Helps the business decide what tradeoff they want.
    """
    rows = []
    for t in np.arange(0.05, 0.55, 0.05):
        preds = (proba >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        n_flagged = preds.sum()
        caught = (preds * y_true).sum()
        rows.append({
            "threshold": round(t, 2),
            "precision": round(p, 3),
            "recall":    round(r, 3),
            "f1":        round(f1, 3),
            "flagged":   int(n_flagged),
            "caught":    int(caught),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
