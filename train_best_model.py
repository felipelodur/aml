"""Train the best AML risk scoring model: LightGBM with GFP features."""

import os
import sys
from pathlib import Path

# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score, average_precision_score
from src.models.train import load_features, temporal_split
from src.config import OUTPUTS_MODELS

print('='*70)
print('TRAINING AML RISK SCORING MODEL: LIGHTGBM + GFP FEATURES')
print('='*70)

# Load and split data
df = load_features()
df_train, df_val, df_test = temporal_split(df)

print(f"\nData splits:")
print(f"  Train: {len(df_train):,} ({df_train['is_laundering'].mean():.4%} laundering)")
print(f"  Val:   {len(df_val):,} ({df_val['is_laundering'].mean():.4%} laundering)")
print(f"  Test:  {len(df_test):,} ({df_test['is_laundering'].mean():.4%} laundering)")

# Build graph structures from training data only (no data leakage)
print("\nBuilding graph features from training data...")
train_s2r = df_train.groupby('from_account')['to_account'].apply(set).to_dict()
train_r2s = df_train.groupby('to_account')['from_account'].apply(set).to_dict()

def add_gfp_features(df, s2r, r2s, train_df):
    """Add Graph Feature Preprocessor features."""
    df = df.copy()

    # Degree features
    df['gfp_sender_in_degree'] = df['from_account'].map(lambda x: len(r2s.get(x, set())))
    df['gfp_receiver_out_degree'] = df['to_account'].map(lambda x: len(s2r.get(x, set())))

    # Fan ratios (key for detecting fan-in/fan-out patterns)
    sender_out = df['from_account'].map(lambda x: len(s2r.get(x, set())))
    df['gfp_sender_fan_ratio'] = sender_out / (df['gfp_sender_in_degree'] + 1)

    receiver_in = df['to_account'].map(lambda x: len(r2s.get(x, set())))
    df['gfp_receiver_fan_ratio'] = df['gfp_receiver_out_degree'] / (receiver_in + 1)

    # Flow ratios (net sender vs net receiver)
    total_out = train_df.groupby('from_account')['amount_usd'].sum()
    total_in = train_df.groupby('to_account')['amount_usd'].sum()

    sender_out_amt = df['from_account'].map(total_out).fillna(0)
    sender_in_amt = df['from_account'].map(total_in).fillna(0)
    df['gfp_sender_flow_ratio'] = sender_out_amt / (sender_in_amt + 1)

    receiver_out_amt = df['to_account'].map(total_out).fillna(0)
    receiver_in_amt = df['to_account'].map(total_in).fillna(0)
    df['gfp_receiver_flow_ratio'] = receiver_out_amt / (receiver_in_amt + 1)

    return df.fillna(0)

# Add GFP features to all splits
df_train = add_gfp_features(df_train, train_s2r, train_r2s, df_train)
df_val = add_gfp_features(df_val, train_s2r, train_r2s, df_train)
df_test = add_gfp_features(df_test, train_s2r, train_r2s, df_train)

# Get feature columns
from src.features.engineering import get_feature_columns
baseline_features = get_feature_columns(df_train)
baseline_features = [f for f in baseline_features if not f.startswith('gfp_')]
gfp_features = [c for c in df_train.columns if c.startswith('gfp_')]
all_features = baseline_features + gfp_features

print(f"\nFeatures: {len(baseline_features)} baseline + {len(gfp_features)} GFP = {len(all_features)} total")
print(f"GFP features: {gfp_features}")

# Prepare data
y_train = df_train['is_laundering']
y_val = df_val['is_laundering']
y_test = df_test['is_laundering']

# Hyperparameters (optimized for this dataset)
params = dict(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=7,
    num_leaves=32,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

print(f'\nHyperparameters:')
for k, v in params.items():
    if k not in ['verbose', 'n_jobs']:
        print(f'  {k}: {v}')

# Train model
print('\nTraining...')
model = LGBMClassifier(**params)
model.fit(
    df_train[all_features], y_train,
    eval_set=[(df_val[all_features], y_val)],
    callbacks=[early_stopping(stopping_rounds=100)]
)

print(f'Trees trained: {model.n_estimators_}')

# Evaluate
proba = model.predict_proba(df_test[all_features])[:, 1]
roc_auc = roc_auc_score(y_test, proba)
pr_auc = average_precision_score(y_test, proba)

print(f'\nTest Performance:')
print(f'  ROC-AUC: {roc_auc:.4f}')
print(f'  PR-AUC:  {pr_auc:.4f}')

# Top-K metrics
print(f'\nTop-K Performance:')
print(f'{"Top-K":<10} {"Recall":>10} {"Precision":>10}')
print('-'*30)
total_pos = y_test.sum()
for top_pct in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
    thresh = np.percentile(proba, 100 - top_pct)
    flagged = proba >= thresh
    tp = (flagged & (y_test.values == 1)).sum()
    recall = tp / total_pos
    precision = tp / flagged.sum()
    print(f'Top {top_pct}%{recall:>10.1%}{precision:>10.1%}')

# Feature importance
print(f'\nTop 15 Features:')
importance = pd.DataFrame({'feature': all_features, 'importance': model.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
for _, row in importance.head(15).iterrows():
    tag = ' (GFP)' if row['feature'].startswith('gfp_') else ''
    print(f"  {row['feature']:<40} {row['importance']:>6.0f}{tag}")

# Save the model
OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
joblib.dump(model, OUTPUTS_MODELS / 'lgbm_best_gfp.joblib')
joblib.dump({
    'all_features': all_features,
    'baseline_features': baseline_features,
    'gfp_features': gfp_features,
    'params': params,
    'trees': model.n_estimators_,
    'roc_auc': roc_auc,
    'pr_auc': pr_auc,
}, OUTPUTS_MODELS / 'lgbm_best_gfp_meta.joblib')

print(f'\nModel saved to: {OUTPUTS_MODELS / "lgbm_best_gfp.joblib"}')
print('Training complete!')
