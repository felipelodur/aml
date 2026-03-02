"""
Feature engineering for AML transaction risk scoring.

Derives behavioral, temporal, network, and typology features from raw transaction data.
Memory-optimized for large datasets (7M+ transactions).
"""

import pandas as pd
import numpy as np
from typing import Tuple

from src.config import (
    TRANSACTIONS_FILE,
    ACCOUNTS_FILE,
    TRANSACTION_COLUMNS,
    DATA_PROCESSED,
)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw transaction and account data."""
    print("Loading raw transaction data...")

    # Load transactions with explicit column names (CSV has duplicate 'Account' headers)
    df_tx = pd.read_csv(
        TRANSACTIONS_FILE,
        names=TRANSACTION_COLUMNS,
        header=0,
        parse_dates=["timestamp"],
    )

    # Load accounts
    df_accounts = pd.read_csv(ACCOUNTS_FILE)
    df_accounts.columns = ["bank_name", "bank_id", "account_number", "entity_id", "entity_name"]

    print(f"Loaded {len(df_tx):,} transactions and {len(df_accounts):,} accounts")
    return df_tx, df_accounts


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp."""
    print("Adding temporal features...")

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(np.int8)
    df["day_of_month"] = df["timestamp"].dt.day

    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive amount-based features."""
    print("Adding amount features...")

    # Amount in USD equivalent (simplified - using amount_paid as proxy)
    df["amount_usd"] = df["amount_paid"].astype(np.float32)

    # Log transform for skewed distribution
    df["log_amount"] = np.log1p(df["amount_usd"]).astype(np.float32)

    # Currency mismatch (cross-border indicator)
    df["currency_mismatch"] = (df["receiving_currency"] != df["payment_currency"]).astype(np.int8)

    # Round amount detection (structuring signal)
    df["is_round_amount"] = ((df["amount_usd"] % 1000 == 0) & (df["amount_usd"] > 0)).astype(np.int8)

    # Near-threshold amounts (common AML thresholds: $10k, $3k)
    df["near_10k_threshold"] = ((df["amount_usd"] >= 9000) & (df["amount_usd"] < 10000)).astype(np.int8)
    df["near_3k_threshold"] = ((df["amount_usd"] >= 2500) & (df["amount_usd"] < 3000)).astype(np.int8)

    # Self-transfer detection
    df["is_self_transfer"] = (
        (df["from_bank"] == df["to_bank"]) &
        (df["from_account"] == df["to_account"])
    ).astype(np.int8)

    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add velocity features using memory-efficient approach.

    Instead of rolling windows (memory-intensive), we use:
    1. Daily aggregates joined back to transactions
    2. Cumulative counts within account groups
    """
    print("Computing velocity features...")

    # Create account identifiers
    df["from_account_id"] = df["from_bank"].astype(str) + "_" + df["from_account"].astype(str)
    df["to_account_id"] = df["to_bank"].astype(str) + "_" + df["to_account"].astype(str)

    # Extract date for daily aggregation
    df["tx_date"] = df["timestamp"].dt.date

    # Sort by timestamp for cumulative features
    df = df.sort_values("timestamp").reset_index(drop=True)

    # === Daily velocity (proxy for 24h window) ===
    print("  Computing daily aggregates...")

    # Sender daily stats
    sender_daily = df.groupby(["from_account_id", "tx_date"]).agg(
        sender_daily_count=("amount_usd", "count"),
        sender_daily_volume=("amount_usd", "sum")
    ).reset_index()

    df = df.merge(sender_daily, on=["from_account_id", "tx_date"], how="left")

    # Receiver daily stats
    receiver_daily = df.groupby(["to_account_id", "tx_date"]).agg(
        receiver_daily_count=("amount_usd", "count"),
        receiver_daily_volume=("amount_usd", "sum")
    ).reset_index()

    df = df.merge(receiver_daily, on=["to_account_id", "tx_date"], how="left")

    # === Cumulative transaction count (running total per account) ===
    print("  Computing cumulative counts...")

    df["sender_cumulative_count"] = df.groupby("from_account_id").cumcount() + 1
    df["receiver_cumulative_count"] = df.groupby("to_account_id").cumcount() + 1

    # === Transaction sequence within day ===
    df["sender_seq_in_day"] = df.groupby(["from_account_id", "tx_date"]).cumcount() + 1
    df["receiver_seq_in_day"] = df.groupby(["to_account_id", "tx_date"]).cumcount() + 1

    # Clean up temp column
    df = df.drop(columns=["tx_date"])

    # Convert to efficient types
    for col in ["sender_daily_count", "receiver_daily_count", "sender_cumulative_count",
                "receiver_cumulative_count", "sender_seq_in_day", "receiver_seq_in_day"]:
        df[col] = df[col].astype(np.int32)

    for col in ["sender_daily_volume", "receiver_daily_volume"]:
        df[col] = df[col].astype(np.float32)

    return df


def add_behavioral_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deviation from historical behavior.
    Memory-efficient using transform instead of merge.
    """
    print("Computing behavioral deviation features...")

    # Sender historical stats using transform (memory efficient)
    df["sender_avg_amount"] = df.groupby("from_account_id")["amount_usd"].transform("mean").astype(np.float32)
    df["sender_std_amount"] = df.groupby("from_account_id")["amount_usd"].transform("std").fillna(0).astype(np.float32)

    # Receiver historical stats
    df["receiver_avg_amount"] = df.groupby("to_account_id")["amount_usd"].transform("mean").astype(np.float32)
    df["receiver_std_amount"] = df.groupby("to_account_id")["amount_usd"].transform("std").fillna(0).astype(np.float32)

    # Z-score deviation
    df["sender_amount_zscore"] = np.where(
        df["sender_std_amount"] > 0,
        (df["amount_usd"] - df["sender_avg_amount"]) / df["sender_std_amount"],
        0
    ).astype(np.float32)

    df["receiver_amount_zscore"] = np.where(
        df["receiver_std_amount"] > 0,
        (df["amount_usd"] - df["receiver_avg_amount"]) / df["receiver_std_amount"],
        0
    ).astype(np.float32)

    # Binary flag for unusual amounts (> 2 std deviations)
    df["sender_unusual_amount"] = (np.abs(df["sender_amount_zscore"]) > 2).astype(np.int8)
    df["receiver_unusual_amount"] = (np.abs(df["receiver_amount_zscore"]) > 2).astype(np.int8)

    return df


def add_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute network/graph features.
    Captures fan-in, fan-out, and counterparty diversity patterns.
    """
    print("Computing network features...")

    # Unique counterparties per sender (fan-out potential)
    sender_counterparties = df.groupby("from_account_id")["to_account_id"].nunique()
    df["sender_unique_receivers"] = df["from_account_id"].map(sender_counterparties).astype(np.int32)

    # Unique counterparties per receiver (fan-in potential)
    receiver_counterparties = df.groupby("to_account_id")["from_account_id"].nunique()
    df["receiver_unique_senders"] = df["to_account_id"].map(receiver_counterparties).astype(np.int32)

    # Total transaction counts per account
    sender_total = df.groupby("from_account_id").size()
    df["sender_total_tx"] = df["from_account_id"].map(sender_total).astype(np.int32)

    receiver_total = df.groupby("to_account_id").size()
    df["receiver_total_tx"] = df["to_account_id"].map(receiver_total).astype(np.int32)

    # Counterparty concentration ratio
    df["sender_concentration"] = (df["sender_unique_receivers"] / df["sender_total_tx"].clip(lower=1)).astype(np.float32)
    df["receiver_concentration"] = (df["receiver_unique_senders"] / df["receiver_total_tx"].clip(lower=1)).astype(np.float32)

    return df


def add_entity_features(df: pd.DataFrame, df_accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich transactions with account/entity information.
    """
    print("Adding entity features...")

    # Create account lookup for entity types
    corp_accounts = set(
        df_accounts[df_accounts["entity_name"].str.startswith("Corporation", na=False)]["account_number"]
    )

    # Map sender/receiver entity type
    df["sender_is_corp"] = df["from_account"].isin(corp_accounts).astype(np.int8)
    df["receiver_is_corp"] = df["to_account"].isin(corp_accounts).astype(np.int8)

    # Cross-bank transfer
    df["is_cross_bank"] = (df["from_bank"] != df["to_bank"]).astype(np.int8)

    return df


def add_payment_format_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode payment format."""
    print("Adding payment format features...")

    # Create dummies for payment format
    payment_dummies = pd.get_dummies(df["payment_format"], prefix="payment", dtype=np.int8)
    df = pd.concat([df, payment_dummies], axis=1)

    return df


def add_temporal_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add enhanced temporal behavior features to capture:
    - Time since last transaction (behavioral change detection)
    - Burst detection (rapid transaction sequences)
    - Deviation from historical timing patterns
    - Velocity acceleration (is activity increasing?)

    These features help detect sudden changes in account behavior,
    which is a strong signal for money laundering activity.
    """
    print("Computing temporal behavior features...")

    # Ensure sorted by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # === Time since last transaction ===
    print("  Computing time-since-last features...")

    # Sender: time since their last outgoing transaction
    df["sender_prev_timestamp"] = df.groupby("from_account_id")["timestamp"].shift(1)
    df["sender_time_since_last"] = (
        (df["timestamp"] - df["sender_prev_timestamp"]).dt.total_seconds() / 3600
    ).astype(np.float32)  # hours
    df["sender_time_since_last"] = df["sender_time_since_last"].fillna(-1)  # -1 for first tx

    # Receiver: time since their last incoming transaction
    df["receiver_prev_timestamp"] = df.groupby("to_account_id")["timestamp"].shift(1)
    df["receiver_time_since_last"] = (
        (df["timestamp"] - df["receiver_prev_timestamp"]).dt.total_seconds() / 3600
    ).astype(np.float32)
    df["receiver_time_since_last"] = df["receiver_time_since_last"].fillna(-1)

    # === Burst detection ===
    print("  Computing burst detection features...")

    # Is this a burst? (transaction within 1 hour of previous)
    df["sender_is_burst"] = (
        (df["sender_time_since_last"] >= 0) &
        (df["sender_time_since_last"] < 1)
    ).astype(np.int8)

    df["receiver_is_burst"] = (
        (df["receiver_time_since_last"] >= 0) &
        (df["receiver_time_since_last"] < 1)
    ).astype(np.int8)

    # Rapid transactions (within 10 minutes)
    df["sender_is_rapid"] = (
        (df["sender_time_since_last"] >= 0) &
        (df["sender_time_since_last"] < 0.167)  # 10 minutes
    ).astype(np.int8)

    df["receiver_is_rapid"] = (
        (df["receiver_time_since_last"] >= 0) &
        (df["receiver_time_since_last"] < 0.167)
    ).astype(np.int8)

    # === Historical timing patterns ===
    print("  Computing historical timing patterns...")

    # Average time gap for this account (excluding first tx)
    sender_avg_gap = df[df["sender_time_since_last"] >= 0].groupby("from_account_id")["sender_time_since_last"].mean()
    df["sender_avg_gap"] = df["from_account_id"].map(sender_avg_gap).fillna(0).astype(np.float32)

    receiver_avg_gap = df[df["receiver_time_since_last"] >= 0].groupby("to_account_id")["receiver_time_since_last"].mean()
    df["receiver_avg_gap"] = df["to_account_id"].map(receiver_avg_gap).fillna(0).astype(np.float32)

    # Gap ratio: current gap vs historical average (detects unusual timing)
    df["sender_gap_ratio"] = np.where(
        (df["sender_avg_gap"] > 0) & (df["sender_time_since_last"] >= 0),
        df["sender_time_since_last"] / df["sender_avg_gap"],
        1.0
    ).astype(np.float32)

    df["receiver_gap_ratio"] = np.where(
        (df["receiver_avg_gap"] > 0) & (df["receiver_time_since_last"] >= 0),
        df["receiver_time_since_last"] / df["receiver_avg_gap"],
        1.0
    ).astype(np.float32)

    # Binary: faster than usual (< 0.5x average gap)
    df["sender_faster_than_usual"] = (
        (df["sender_time_since_last"] >= 0) &
        (df["sender_gap_ratio"] < 0.5)
    ).astype(np.int8)

    df["receiver_faster_than_usual"] = (
        (df["receiver_time_since_last"] >= 0) &
        (df["receiver_gap_ratio"] < 0.5)
    ).astype(np.int8)

    # === Transaction sequence position ===
    print("  Computing sequence position features...")

    # What transaction number is this for the account?
    df["sender_tx_number"] = df.groupby("from_account_id").cumcount() + 1
    df["receiver_tx_number"] = df.groupby("to_account_id").cumcount() + 1

    # Is this a new account? (first 3 transactions)
    df["sender_is_new"] = (df["sender_tx_number"] <= 3).astype(np.int8)
    df["receiver_is_new"] = (df["receiver_tx_number"] <= 3).astype(np.int8)

    # === Hour-based behavioral deviation ===
    print("  Computing hour deviation features...")

    # Typical hour for this sender
    sender_typical_hour = df.groupby("from_account_id")["hour"].transform("median")
    df["sender_hour_deviation"] = np.abs(df["hour"] - sender_typical_hour).astype(np.float32)

    # Binary: unusual hour (> 6 hours from typical)
    df["sender_unusual_hour"] = (df["sender_hour_deviation"] > 6).astype(np.int8)

    # === Recent velocity (last 5 transactions timespan) ===
    print("  Computing recent velocity features...")

    # Get timestamp of 5th previous transaction
    df["sender_5th_prev_ts"] = df.groupby("from_account_id")["timestamp"].shift(5)
    df["sender_last5_timespan"] = (
        (df["timestamp"] - df["sender_5th_prev_ts"]).dt.total_seconds() / 3600
    ).fillna(-1).astype(np.float32)

    # Recent velocity: 5 transactions / timespan (tx per hour)
    df["sender_recent_velocity"] = np.where(
        df["sender_last5_timespan"] > 0,
        5.0 / df["sender_last5_timespan"],
        0
    ).astype(np.float32)

    # Same for receiver
    df["receiver_5th_prev_ts"] = df.groupby("to_account_id")["timestamp"].shift(5)
    df["receiver_last5_timespan"] = (
        (df["timestamp"] - df["receiver_5th_prev_ts"]).dt.total_seconds() / 3600
    ).fillna(-1).astype(np.float32)

    df["receiver_recent_velocity"] = np.where(
        df["receiver_last5_timespan"] > 0,
        5.0 / df["receiver_last5_timespan"],
        0
    ).astype(np.float32)

    # === Velocity change (acceleration) ===
    print("  Computing velocity change features...")

    # Compare current velocity to previous velocity
    df["sender_prev_velocity"] = df.groupby("from_account_id")["sender_recent_velocity"].shift(1).fillna(0)
    df["sender_velocity_change"] = (df["sender_recent_velocity"] - df["sender_prev_velocity"]).astype(np.float32)

    # Is velocity accelerating?
    df["sender_accelerating"] = (df["sender_velocity_change"] > 0).astype(np.int8)

    df["receiver_prev_velocity"] = df.groupby("to_account_id")["receiver_recent_velocity"].shift(1).fillna(0)
    df["receiver_velocity_change"] = (df["receiver_recent_velocity"] - df["receiver_prev_velocity"]).astype(np.float32)
    df["receiver_accelerating"] = (df["receiver_velocity_change"] > 0).astype(np.int8)

    # === Cleanup temporary columns ===
    temp_cols = [
        "sender_prev_timestamp", "receiver_prev_timestamp",
        "sender_5th_prev_ts", "receiver_5th_prev_ts",
        "sender_prev_velocity", "receiver_prev_velocity",
    ]
    df = df.drop(columns=temp_cols)

    print(f"  Added {26} temporal behavior features")

    return df


def engineer_features(save_processed: bool = True) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    Returns DataFrame with all engineered features ready for model training.
    """
    # Load data
    df_tx, df_accounts = load_raw_data()

    # Apply feature engineering steps (in-place modifications where possible)
    df = df_tx  # Start with raw data
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
    df = add_behavioral_deviation_features(df)
    df = add_network_features(df)
    df = add_entity_features(df, df_accounts)
    df = add_payment_format_features(df)
    df = add_temporal_behavior_features(df)  # Enhanced temporal features

    # Fill any remaining NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print(f"\nFeature engineering complete. Shape: {df.shape}")
    print(f"Target distribution:\n{df['is_laundering'].value_counts(normalize=True)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    if save_processed:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        output_path = DATA_PROCESSED / "features.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved processed features to {output_path}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature columns for model training."""
    exclude_cols = [
        "timestamp", "from_bank", "from_account", "to_bank", "to_account",
        "amount_received", "receiving_currency", "amount_paid", "payment_currency",
        "payment_format", "is_laundering", "from_account_id", "to_account_id",
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


if __name__ == "__main__":
    df = engineer_features()
    print(f"\nFeature columns: {get_feature_columns(df)}")
