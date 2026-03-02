"""
Model training for AML transaction risk scoring.

Uses LightGBM with proper temporal train/validation/test split
to avoid data leakage and simulate production deployment.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any

import lightgbm as lgb

from src.config import (
    RANDOM_STATE,
    DATA_PROCESSED,
    OUTPUTS_MODELS,
)
from src.features.engineering import engineer_features, get_feature_columns


def load_features() -> pd.DataFrame:
    """Load processed features from parquet file."""
    feature_path = DATA_PROCESSED / "features.parquet"

    if feature_path.exists():
        print(f"Loading processed features from {feature_path}")
        return pd.read_parquet(feature_path)
    else:
        print("Processed features not found. Running feature engineering...")
        return engineer_features()


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally: train on past, validate/test on future.

    This avoids data leakage and simulates production deployment where
    the model only has access to historical data when scoring new transactions.
    """
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    return df_train, df_val, df_test


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Prepare features and target for training."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["is_laundering"]
    return X, y, feature_cols


def train_model(
    df: pd.DataFrame = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    num_leaves: int = 31,
    early_stopping_rounds: int = 20,
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Train LightGBM classifier with proper temporal split.

    Approach:
    ---------
    1. Temporal split: Train on past, validate on near-future, test on far-future
    2. No SMOTE: Let model learn natural class probabilities
    3. No class weighting: Produces well-calibrated probability scores
    4. Early stopping: Prevent overfitting using validation set
    """
    if df is None:
        df = load_features()

    # Temporal split
    print("\n" + "="*60)
    print("TEMPORAL DATA SPLIT")
    print("="*60)

    df_train, df_val, df_test = temporal_split(df)

    print(f"\n{'Split':<8} {'Rows':>12} {'Laundering':>12} {'Rate':>10} {'Period'}")
    print("-"*75)
    print(f"{'Train':<8} {len(df_train):>12,} {df_train['is_laundering'].sum():>12,} {df_train['is_laundering'].mean():>10.4%} {df_train['timestamp'].min().date()} to {df_train['timestamp'].max().date()}")
    print(f"{'Val':<8} {len(df_val):>12,} {df_val['is_laundering'].sum():>12,} {df_val['is_laundering'].mean():>10.4%} {df_val['timestamp'].min().date()} to {df_val['timestamp'].max().date()}")
    print(f"{'Test':<8} {len(df_test):>12,} {df_test['is_laundering'].sum():>12,} {df_test['is_laundering'].mean():>10.4%} {df_test['timestamp'].min().date()} to {df_test['timestamp'].max().date()}")

    # Prepare features
    feature_cols = get_feature_columns(df_train)
    X_train, y_train, _ = prepare_data(df_train)
    X_val, y_val, _ = prepare_data(df_val)
    X_test, y_test, _ = prepare_data(df_test)

    print(f"\nFeatures: {len(feature_cols)}")

    # Model hyperparameters
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    lgb_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_child_samples": 50,
        "lambda_l1": 0.0,
        "lambda_l2": 0.1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,  # No class weighting for calibrated probabilities
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    print(f"\nHyperparameters:")
    print(f"  n_estimators: {n_estimators} (max, with early stopping)")
    print(f"  learning_rate: {learning_rate}")
    print(f"  max_depth: {max_depth}")
    print(f"  num_leaves: {num_leaves}")
    print(f"  early_stopping_rounds: {early_stopping_rounds}")

    model = lgb.LGBMClassifier(**lgb_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=20),
        ],
    )

    print(f"\nTrees used: {model.n_estimators_}")
    print(f"Best iteration: {model.best_iteration_}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\nTop 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")

    # Store metadata
    metadata = {
        "feature_columns": feature_cols,
        "split_type": "temporal",
        "train_period": f"{df_train['timestamp'].min()} to {df_train['timestamp'].max()}",
        "val_period": f"{df_val['timestamp'].min()} to {df_val['timestamp'].max()}",
        "test_period": f"{df_test['timestamp'].min()} to {df_test['timestamp'].max()}",
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "lgb_params": lgb_params,
        "n_trees": model.n_estimators_,
        "best_iteration": model.best_iteration_,
    }

    return model, metadata, df_test, feature_cols


def save_model(
    model: lgb.LGBMClassifier,
    metadata: Dict[str, Any],
    model_name: str = "lgbm_best_gfp",
) -> Path:
    """Save trained model and metadata."""
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUTS_MODELS / f"{model_name}.joblib"
    metadata_path = OUTPUTS_MODELS / f"{model_name}_metadata.joblib"

    joblib.dump(model, model_path)
    joblib.dump(metadata, metadata_path)

    print(f"\nModel saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")

    return model_path


def load_model(model_name: str = "lgbm_best_gfp") -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """Load trained model and metadata."""
    model_path = OUTPUTS_MODELS / f"{model_name}.joblib"
    metadata_path = OUTPUTS_MODELS / f"{model_name}_meta.joblib"

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    return model, metadata


if __name__ == "__main__":
    # Train model
    model, metadata, df_test, feature_cols = train_model()

    # Save model
    save_model(model, metadata)

    print("\nTraining complete!")
