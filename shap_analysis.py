"""
SHAP Analysis for Feature Importance
Generates:
1. Top predictive signals (SHAP summary plot)
2. Grouped SHAP by feature category
"""

import os
import sys
from pathlib import Path

# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from src.models.train import load_features, temporal_split

print('='*70)
print('SHAP ANALYSIS')
print('='*70)

# Load model and data
model = joblib.load('outputs/models/lgbm_best_gfp.joblib')
meta = joblib.load('outputs/models/lgbm_best_gfp_meta.joblib')
feature_cols = meta['all_features']

df = load_features()
df_train, df_val, df_test = temporal_split(df)

# Add GFP features
train_s2r = df_train.groupby('from_account')['to_account'].apply(set).to_dict()
train_r2s = df_train.groupby('to_account')['from_account'].apply(set).to_dict()

def add_gfp_features(df, train_df):
    df = df.copy()
    df['gfp_sender_in_degree'] = df['from_account'].map(lambda x: len(train_r2s.get(x, set())))
    df['gfp_receiver_out_degree'] = df['to_account'].map(lambda x: len(train_s2r.get(x, set())))
    sender_out = df['from_account'].map(lambda x: len(train_s2r.get(x, set())))
    df['gfp_sender_fan_ratio'] = sender_out / (df['gfp_sender_in_degree'] + 1)
    receiver_in = df['to_account'].map(lambda x: len(train_r2s.get(x, set())))
    df['gfp_receiver_fan_ratio'] = df['gfp_receiver_out_degree'] / (receiver_in + 1)
    total_out = train_df.groupby('from_account')['amount_usd'].sum()
    total_in = train_df.groupby('to_account')['amount_usd'].sum()
    df['gfp_sender_flow_ratio'] = df['from_account'].map(total_out).fillna(0) / (df['from_account'].map(total_in).fillna(0) + 1)
    df['gfp_receiver_flow_ratio'] = df['to_account'].map(total_out).fillna(0) / (df['to_account'].map(total_in).fillna(0) + 1)
    return df.fillna(0)

df_test = add_gfp_features(df_test, df_train)
X_test = df_test[feature_cols]

# Sample for SHAP (full dataset too large)
print('\nSampling 10,000 transactions for SHAP analysis...')
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(10000, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx]

# Compute SHAP values
print('Computing SHAP values (this may take a minute)...')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# For binary classification, shap_values is a list [class_0, class_1]
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Use class 1 (fraud)

print(f'SHAP values shape: {shap_values.shape}')

# ============================================================================
# PLOT 1: Top Predictive Signals (SHAP Summary)
# ============================================================================
print('\nGenerating SHAP summary plot...')

fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
plt.title('Top 20 Predictive Signals (Mean |SHAP|)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/reports/shap_top_features.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: outputs/reports/shap_top_features.png')

# Also create beeswarm plot
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
plt.title('SHAP Feature Impact (Top 15)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/reports/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: outputs/reports/shap_beeswarm.png')

# ============================================================================
# PLOT 2: Grouped SHAP by Feature Category
# ============================================================================
print('\nComputing grouped SHAP values by category...')

# Define feature categories (5 groups)
feature_categories = {
    'Amount': ['amount_usd', 'log_amount', 'currency_mismatch', 'is_round_amount',
               'near_10k_threshold', 'near_3k_threshold', 'is_self_transfer'] +
              [f for f in feature_cols if f.startswith('payment_')],  # Include payment format
    'Velocity': ['sender_daily_count', 'sender_daily_volume', 'receiver_daily_count',
                 'receiver_daily_volume', 'sender_cumulative_count', 'receiver_cumulative_count',
                 'sender_seq_in_day', 'receiver_seq_in_day'],
    'Deviation': ['sender_avg_amount', 'sender_std_amount', 'sender_amount_zscore',
                  'sender_unusual_amount', 'receiver_avg_amount', 'receiver_std_amount',
                  'receiver_amount_zscore', 'receiver_unusual_amount'],
    'Timing': ['hour', 'day_of_week', 'is_weekend', 'is_night', 'day_of_month',
               'sender_time_since_last', 'receiver_time_since_last', 'sender_is_burst',
               'receiver_is_burst', 'sender_is_rapid', 'receiver_is_rapid',
               'sender_avg_gap', 'receiver_avg_gap', 'sender_gap_ratio', 'receiver_gap_ratio',
               'sender_faster_than_usual', 'receiver_faster_than_usual',
               'sender_tx_number', 'receiver_tx_number', 'sender_is_new', 'receiver_is_new',
               'sender_hour_deviation', 'sender_unusual_hour',
               'sender_last5_timespan', 'sender_recent_velocity', 'sender_velocity_change',
               'sender_accelerating', 'receiver_last5_timespan', 'receiver_recent_velocity',
               'receiver_velocity_change', 'receiver_accelerating'],
    'Network': ['sender_unique_receivers', 'receiver_unique_senders', 'sender_total_tx',
                'receiver_total_tx', 'sender_concentration', 'receiver_concentration',
                'gfp_sender_in_degree', 'gfp_receiver_out_degree', 'gfp_sender_fan_ratio',
                'gfp_receiver_fan_ratio', 'gfp_sender_flow_ratio', 'gfp_receiver_flow_ratio',
                'sender_is_corp', 'receiver_is_corp', 'is_cross_bank'],  # Include entity features
}

# Calculate mean |SHAP| per category
category_importance = {}
shap_df = pd.DataFrame(shap_values, columns=feature_cols)

for category, features in feature_categories.items():
    # Find features that exist in our feature set
    valid_features = [f for f in features if f in feature_cols]
    if valid_features:
        # Sum absolute SHAP values for features in this category
        category_shap = shap_df[valid_features].abs().sum(axis=1).mean()
        category_importance[category] = category_shap
        print(f'  {category}: {len(valid_features)} features, mean |SHAP| = {category_shap:.4f}')

# Sort by importance
category_importance = dict(sorted(category_importance.items(), key=lambda x: x[1], reverse=True))

# Plot grouped SHAP
fig, ax = plt.subplots(figsize=(10, 6))
categories = list(category_importance.keys())
values = list(category_importance.values())
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

bars = ax.barh(categories[::-1], values[::-1], color=colors[:len(categories)][::-1])
ax.set_xlabel('Mean |SHAP| Value (Feature Group Contribution)', fontsize=12)
ax.set_title('Feature Group Importance', fontsize=14, fontweight='bold')

# Add value labels
for bar, val in zip(bars, values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/reports/shap_grouped_features.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: outputs/reports/shap_grouped_features.png')

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print('\n' + '='*70)
print('FEATURE GROUP IMPORTANCE SUMMARY')
print('='*70)
print(f"\n{'Category':<15} {'Features':>10} {'Mean |SHAP|':>15} {'% of Total':>12}")
print('-'*55)

total_shap = sum(category_importance.values())
for cat, val in category_importance.items():
    n_features = len([f for f in feature_categories[cat] if f in feature_cols])
    pct = val / total_shap * 100
    print(f'{cat:<15} {n_features:>10} {val:>15.4f} {pct:>11.1f}%')

print('-'*55)
print(f"{'TOTAL':<15} {len(feature_cols):>10} {total_shap:>15.4f} {100:>11.1f}%")

print('\nDone!')
