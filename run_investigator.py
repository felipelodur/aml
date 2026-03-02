"""
Part 2: AI-Powered Investigation Assistant

Takes the top 5% highest-risk transactions from the ML model and generates
preliminary investigation briefs for each, covering:
- Transaction summary
- Risk signals in plain language
- Entity behavioral context
- Pattern analysis (known laundering typologies)
- Recommended action

Usage:
    python run_investigator.py                  # Mock mode (5 examples)
    python run_investigator.py --use-llm        # Real LLM mode
    python run_investigator.py --n-briefs 10    # More examples
"""

import os
import sys
from pathlib import Path

# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.models.train import load_model, load_features, temporal_split

print("=" * 70)
print("PART 2: AI-POWERED INVESTIGATION ASSISTANT")
print("=" * 70)

# =============================================================================
# STEP 1: Load trained model and test data
# =============================================================================
print("\n[1/4] Loading model and data...")

model, meta = load_model()  # Loads lgbm_best_gfp
feature_cols = meta['all_features']
print(f"  Model: LightGBM + GFP ({len(feature_cols)} features)")

df = load_features()
df_train, _, df_test = temporal_split(df)
print(f"  Test set: {len(df_test):,} transactions")

# =============================================================================
# STEP 2: Add GFP features and get predictions
# =============================================================================
print("\n[2/4] Computing risk scores...")

# Build GFP features from training data
train_s2r = df_train.groupby('from_account')['to_account'].apply(set).to_dict()
train_r2s = df_train.groupby('to_account')['from_account'].apply(set).to_dict()

df_test = df_test.copy()
df_test['gfp_sender_in_degree'] = df_test['from_account'].map(lambda x: len(train_r2s.get(x, set())))
df_test['gfp_receiver_out_degree'] = df_test['to_account'].map(lambda x: len(train_s2r.get(x, set())))
sender_out = df_test['from_account'].map(lambda x: len(train_s2r.get(x, set())))
df_test['gfp_sender_fan_ratio'] = sender_out / (df_test['gfp_sender_in_degree'] + 1)
receiver_in = df_test['to_account'].map(lambda x: len(train_r2s.get(x, set())))
df_test['gfp_receiver_fan_ratio'] = df_test['gfp_receiver_out_degree'] / (receiver_in + 1)
total_out = df_train.groupby('from_account')['amount_usd'].sum()
total_in = df_train.groupby('to_account')['amount_usd'].sum()
df_test['gfp_sender_flow_ratio'] = df_test['from_account'].map(total_out).fillna(0) / (df_test['from_account'].map(total_in).fillna(0) + 1)
df_test['gfp_receiver_flow_ratio'] = df_test['to_account'].map(total_out).fillna(0) / (df_test['to_account'].map(total_in).fillna(0) + 1)
df_test = df_test.fillna(0)

# Get predictions
X_test = df_test[feature_cols]
y_proba = model.predict_proba(X_test)[:, 1]
df_test["risk_score"] = y_proba

# =============================================================================
# STEP 3: Select top 5% highest-risk transactions
# =============================================================================
print("\n[3/4] Selecting top 5% highest-risk transactions...")

threshold_95 = np.percentile(y_proba, 95)
top_5pct = df_test[df_test['risk_score'] >= threshold_95].copy()
top_5pct = top_5pct.sort_values('risk_score', ascending=False)

print(f"  Threshold (95th percentile): {threshold_95:.6f}")
print(f"  Transactions in top 5%: {len(top_5pct):,}")
print(f"  Fraud rate in top 5%: {top_5pct['is_laundering'].mean():.2%}")

# =============================================================================
# STEP 4: Generate investigation briefs
# =============================================================================
print("\n[4/4] Generating investigation briefs...")

from src.llm.investigator import generate_investigation_briefs, _format_brief_with_verification
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n-briefs", type=int, default=5, help="Number of briefs to generate")
parser.add_argument("--use-llm", action="store_true", help="Use local LLM (Phi-3) for brief generation")
parser.add_argument("--llm-provider", type=str, default="local", choices=["local", "anthropic", "openai"],
                    help="LLM provider: local (Phi-3), anthropic, or openai")
parser.add_argument("--no-verify", action="store_true", help="Skip verification")
args, _ = parser.parse_known_args()

if args.use_llm:
    print(f"  Using LLM provider: {args.llm_provider}")
    print("  Typology: ML Classifier (84% accuracy)")
    print("  Brief: Local LLM (Phi-3)" if args.llm_provider == "local" else f"  Brief: {args.llm_provider} API")

results = generate_investigation_briefs(
    top_5pct,
    model,
    feature_cols,
    n_examples=args.n_briefs,
    use_mock=not args.use_llm,
    llm_provider=args.llm_provider,
    model_type="LightGBM",
    verify_briefs=not args.no_verify,
    use_llm_judge=False,  # Use programmatic verification only for speed
)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("INVESTIGATION ASSISTANT COMPLETE")
print("=" * 70)
print(f"""
Pipeline Summary:
  1. Loaded LightGBM + GFP model ({len(feature_cols)} features)
  2. Scored {len(df_test):,} test transactions
  3. Selected top 5% ({len(top_5pct):,} transactions, {top_5pct['is_laundering'].mean():.1%} fraud rate)
  4. Generated {args.n_briefs} investigation briefs

Outputs:
  - outputs/briefs/all_investigation_briefs.txt
  - outputs/briefs/brief_1.txt ... brief_{args.n_briefs}.txt
  - outputs/briefs/verification_report.txt

Each brief includes:
  - Transaction summary
  - Risk signals in plain language (from SHAP)
  - Entity behavioral context
  - Typology analysis (Fan-In, Fan-Out, Rapid Movement, etc.)
  - Recommended action (Escalate/EDD/Review/Clear)
  - Suggested next steps
  - Analyst notes (limitations, areas requiring human judgment)
""")

# Show first example
print("=" * 70)
print("EXAMPLE BRIEF (Brief #1):")
print("=" * 70)
print(_format_brief_with_verification(results[0]))
