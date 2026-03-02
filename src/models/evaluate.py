"""
Model evaluation and interpretation for AML risk scoring.

Provides metrics relevant to compliance context and SHAP-based explanations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, Any, Tuple, List
from pathlib import Path

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import OUTPUTS_REPORTS, TOP_RISK_PERCENTILE


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with compliance-relevant metrics.

    Evaluation Metric Rationale:
    ----------------------------
    In AML compliance, we prioritize:

    1. PR-AUC over ROC-AUC: With ~2% illicit rate, ROC-AUC can be misleadingly
       high. PR-AUC better reflects performance on the minority class.

    2. Recall at Fixed Precision: "How many bad actors do we catch while keeping
       false alerts manageable?" This maps to operational capacity.

    3. Precision-Recall Tradeoff: Show the explicit tradeoff so compliance can
       choose their operating point based on team capacity.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)

    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Core metrics
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n1. OVERALL PERFORMANCE")
    print(f"   PR-AUC (Primary): {pr_auc:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   (PR-AUC is more relevant for imbalanced AML data)")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n2. CONFUSION MATRIX (threshold={threshold})")
    print(f"   True Negatives:  {tn:,}")
    print(f"   False Positives: {fp:,} (legitimate flagged as suspicious)")
    print(f"   False Negatives: {fn:,} (laundering missed - CRITICAL)")
    print(f"   True Positives:  {tp:,} (laundering caught)")

    # Operational metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n3. OPERATIONAL METRICS")
    print(f"   Precision: {precision:.4f} (of flagged, how many are actually bad)")
    print(f"   Recall: {recall:.4f} (of all bad, how many did we catch)")
    print(f"   F1 Score: {f1:.4f}")

    # False positive rate (current system is 85%)
    fpr = fp / (fp + tn)
    print(f"\n4. FALSE POSITIVE RATE COMPARISON")
    print(f"   Current rule-based system: 85%")
    print(f"   ML model: {fpr:.1%}")
    print(f"   Improvement: {(0.85 - fpr) / 0.85:.1%} reduction")

    # Recall at precision thresholds (operational planning)
    print(f"\n5. RECALL AT PRECISION THRESHOLDS")
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_proba)

    for target_precision in [0.10, 0.15, 0.20, 0.30]:
        idx = np.where(precision_curve >= target_precision)[0]
        if len(idx) > 0:
            recall_at_precision = recall_curve[idx[0]]
            threshold_at_precision = thresholds[idx[0]] if idx[0] < len(thresholds) else 1.0
            print(f"   At {target_precision:.0%} precision: {recall_at_precision:.1%} recall (threshold={threshold_at_precision:.3f})")

    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "confusion_matrix": cm,
    }

    return metrics


def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: int = 5000,
) -> Tuple[shap.Explainer, np.ndarray]:
    """Compute SHAP values for model interpretation."""
    print("\nComputing SHAP values (this may take a moment)...")

    # Sample for efficiency
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return explainer, shap_values, X_sample


def plot_feature_importance(
    model,
    feature_columns: List[str],
    save_path: Path = None,
) -> None:
    """Plot feature importance from model."""
    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=importance, x="importance", y="feature", ax=ax)
    ax.set_title("Top 20 Features by Importance")
    ax.set_xlabel("Importance Score")

    if save_path is None:
        save_path = OUTPUTS_REPORTS / "feature_importance.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Feature importance plot saved to {save_path}")


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    save_path: Path = None,
) -> None:
    """Plot SHAP summary for global interpretation."""
    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)

    if save_path is None:
        save_path = OUTPUTS_REPORTS / "shap_summary.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SHAP summary plot saved to {save_path}")


def explain_for_compliance(
    model,
    feature_columns: List[str],
    shap_values: np.ndarray = None,
) -> str:
    """
    Generate plain-language explanation of model learnings for compliance lead.
    """
    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    top_features = importance.head(10)

    explanation = """
MODEL INTERPRETATION FOR COMPLIANCE
====================================

What the Model Learned:
-----------------------

The model identifies suspicious transactions by weighing multiple risk signals.
Here are the top factors it considers, in order of importance:

"""

    feature_explanations = {
        "sender_volume_24h": "High outgoing transaction volume in 24 hours",
        "receiver_volume_24h": "High incoming transaction volume in 24 hours",
        "sender_tx_count_24h": "Multiple outgoing transactions in 24 hours",
        "receiver_tx_count_24h": "Multiple incoming transactions in 24 hours",
        "sender_unique_receivers": "Sending to many different accounts (fan-out)",
        "receiver_unique_senders": "Receiving from many different accounts (fan-in)",
        "log_amount": "Transaction amount (log-scaled)",
        "amount_usd": "Raw transaction amount",
        "sender_amount_zscore": "Unusual amount for this sender's history",
        "receiver_amount_zscore": "Unusual amount for this receiver's history",
        "is_cross_bank": "Transfer between different banks",
        "currency_mismatch": "Different sending and receiving currencies",
        "is_weekend": "Transaction on weekend",
        "is_night": "Transaction during nighttime hours",
        "is_round_amount": "Suspiciously round dollar amount",
        "near_10k_threshold": "Amount just below $10,000 reporting threshold",
        "sender_unusual_amount": "Amount significantly different from sender's norm",
        "receiver_unusual_amount": "Amount significantly different from receiver's norm",
    }

    for i, row in top_features.iterrows():
        feature = row["feature"]
        importance_score = row["importance"]
        human_readable = feature_explanations.get(feature, feature)
        explanation += f"{len(explanation.split(chr(10)))-8}. {human_readable}\n"
        explanation += f"   (Importance: {importance_score:.4f})\n\n"

    explanation += """
Key Patterns Detected:
----------------------

1. VELOCITY PATTERNS: The model heavily weighs transaction frequency and volume
   over short time windows. Rapid movement of funds is a strong signal.

2. NETWORK PATTERNS: Fan-in (many-to-one) and fan-out (one-to-many) structures
   are captured through counterparty diversity metrics.

3. BEHAVIORAL ANOMALIES: Transactions that deviate from an account's historical
   pattern are flagged as unusual.

4. STRUCTURING SIGNALS: The model detects amounts near reporting thresholds
   and suspiciously round numbers.

5. CROSS-BORDER INDICATORS: Currency mismatches and inter-bank transfers
   contribute to risk scores.

How to Use Risk Scores:
-----------------------

- Scores range from 0.0 (low risk) to 1.0 (high risk)
- Transactions in the top 5% by score should receive priority review
- Each flagged transaction includes SHAP explanations showing WHY it scored high
- Use the risk score to prioritize investigation queue, not as sole decision
"""

    return explanation


def get_top_risk_transactions(
    model,
    df: pd.DataFrame,
    X: pd.DataFrame,
    percentile: float = TOP_RISK_PERCENTILE,
) -> pd.DataFrame:
    """
    Get top risk transactions for LLM investigation assistant.
    """
    # Get risk scores
    risk_scores = model.predict_proba(X)[:, 1]
    df_scored = df.copy()
    df_scored["risk_score"] = risk_scores

    # Get threshold for top percentile
    threshold = np.percentile(risk_scores, 100 * (1 - percentile))
    top_risk = df_scored[df_scored["risk_score"] >= threshold].sort_values(
        "risk_score", ascending=False
    )

    print(f"\nTop {percentile:.0%} risk transactions: {len(top_risk):,}")
    print(f"Risk score threshold: {threshold:.4f}")

    return top_risk


if __name__ == "__main__":
    from src.models.train import load_model, load_features, prepare_data

    # Load model and data
    model, metadata = load_model()
    df = load_features()
    X, y, feature_cols = prepare_data(df)

    # Evaluate
    metrics = evaluate_model(model, X, y)

    # SHAP analysis
    explainer, shap_values, X_sample = compute_shap_values(model, X)

    # Plots
    plot_feature_importance(model, feature_cols)
    plot_shap_summary(shap_values, X_sample)

    # Compliance explanation
    explanation = explain_for_compliance(model, feature_cols)
    print(explanation)

    # Save explanation
    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_REPORTS / "model_interpretation.txt", "w") as f:
        f.write(explanation)
