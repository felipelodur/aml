"""
AML Risk Scoring System - Main Entry Point

Usage:
    python main.py                    # Run full pipeline
    python main.py --train-only       # Only model training
    python main.py --briefs-only      # Only generate investigation briefs
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_training():
    """Train the best model with GFP features."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    # Import and run training script
    import train_best_model
    print("\nTraining complete!")


def run_investigation_briefs(n_examples=5, use_mock=True):
    """Generate investigation briefs for high-risk transactions."""
    print("\n" + "="*60)
    print("LLM INVESTIGATION ASSISTANT")
    print("="*60)

    from src.models.train import load_model, load_features, temporal_split
    from src.llm.investigator import generate_investigation_briefs

    # Load model
    model, meta = load_model()
    feature_cols = meta['all_features']

    # Load and prepare test data
    df = load_features()
    df_train, _, df_test = temporal_split(df)

    # Add GFP features
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
    df_test['risk_score'] = y_proba

    # Get top risk transactions
    top_risk = df_test.nlargest(100, 'risk_score')

    # Generate briefs
    briefs = generate_investigation_briefs(
        top_risk,
        model,
        feature_cols,
        n_examples=n_examples,
        use_mock=use_mock,
        model_type="LightGBM",
    )

    return briefs


def run_full_pipeline():
    """Run the complete pipeline."""
    print("\n" + "="*60)
    print("AML RISK SCORING SYSTEM")
    print("="*60)
    print("Running full pipeline...")

    # Training
    run_training()

    # Investigation briefs
    run_investigation_briefs(n_examples=5, use_mock=True)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nOutputs generated:")
    print("  - outputs/models/lgbm_best_gfp.joblib")
    print("  - outputs/briefs/all_investigation_briefs.txt")


def main():
    parser = argparse.ArgumentParser(description="AML Risk Scoring System")
    parser.add_argument("--train-only", action="store_true", help="Only run model training")
    parser.add_argument("--briefs-only", action="store_true", help="Only generate investigation briefs")
    parser.add_argument("--use-llm", action="store_true", help="Use real LLM instead of mock")
    parser.add_argument("--n-briefs", type=int, default=5, help="Number of briefs to generate")

    args = parser.parse_args()

    if args.train_only:
        run_training()
    elif args.briefs_only:
        run_investigation_briefs(n_examples=args.n_briefs, use_mock=not args.use_llm)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
