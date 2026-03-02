"""
Graph Feature Preprocessor (GFP) for AML Detection

Based on: "Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction
for Financial Crime Detection" (2024)

Extracts graph-based features without using GNNs, then feeds to LightGBM.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import networkx as nx
from tqdm import tqdm


def build_transaction_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build directed multigraph from transactions."""
    G = nx.MultiDiGraph()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graph"):
        G.add_edge(
            row['from_account'],
            row['to_account'],
            amount=row['amount_usd'],
            timestamp=row['timestamp'] if 'timestamp' in row else 0,
            tx_id=row.name
        )

    return G


def extract_fan_features(df: pd.DataFrame, time_windows: List[int] = [1, 7, 30]) -> pd.DataFrame:
    """
    Extract fan-in and fan-out features at different time windows.

    Fan-out: One sender -> many receivers (placement)
    Fan-in: Many senders -> one receiver (layering)
    """
    features = {}

    # Sort by timestamp for time-window calculations
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Group by sender and receiver
    sender_groups = df.groupby('from_account')
    receiver_groups = df.groupby('to_account')

    # Pre-compute aggregations
    print("Computing fan-out features...")
    sender_stats = sender_groups.agg({
        'to_account': ['count', 'nunique'],
        'amount_usd': ['sum', 'mean', 'std', 'min', 'max'],
    })
    sender_stats.columns = ['_'.join(col) for col in sender_stats.columns]
    sender_stats = sender_stats.rename(columns={
        'to_account_count': 'sender_total_txns',
        'to_account_nunique': 'sender_unique_receivers_all',
        'amount_usd_sum': 'sender_total_amount',
        'amount_usd_mean': 'sender_avg_amount',
        'amount_usd_std': 'sender_std_amount',
        'amount_usd_min': 'sender_min_amount',
        'amount_usd_max': 'sender_max_amount',
    })

    print("Computing fan-in features...")
    receiver_stats = receiver_groups.agg({
        'from_account': ['count', 'nunique'],
        'amount_usd': ['sum', 'mean', 'std', 'min', 'max'],
    })
    receiver_stats.columns = ['_'.join(col) for col in receiver_stats.columns]
    receiver_stats = receiver_stats.rename(columns={
        'from_account_count': 'receiver_total_txns',
        'from_account_nunique': 'receiver_unique_senders_all',
        'amount_usd_sum': 'receiver_total_amount',
        'amount_usd_mean': 'receiver_avg_amount',
        'amount_usd_std': 'receiver_std_amount',
        'amount_usd_min': 'receiver_min_amount',
        'amount_usd_max': 'receiver_max_amount',
    })

    # Merge back
    df = df.merge(sender_stats, left_on='from_account', right_index=True, how='left')
    df = df.merge(receiver_stats, left_on='to_account', right_index=True, how='left')

    # Fill NaN with 0
    df = df.fillna(0)

    return df


def extract_2hop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 2-hop neighborhood features.

    For each transaction A->B:
    - How many accounts did A receive from? (A's in-degree)
    - How many accounts does B send to? (B's out-degree)
    - Is there a path B->...->A? (cycle detection)
    """
    print("Computing 2-hop features...")

    # Build adjacency info
    sender_to_receivers = df.groupby('from_account')['to_account'].apply(set).to_dict()
    receiver_to_senders = df.groupby('to_account')['from_account'].apply(set).to_dict()

    # For each transaction, compute 2-hop features
    features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="2-hop features"):
        sender = row['from_account']
        receiver = row['to_account']

        # Sender's in-neighbors (who sends to sender)
        sender_in_neighbors = receiver_to_senders.get(sender, set())

        # Receiver's out-neighbors (who receiver sends to)
        receiver_out_neighbors = sender_to_receivers.get(receiver, set())

        # 2-hop cycle: Does receiver (or receiver's out-neighbors) send back to sender?
        has_direct_cycle = sender in sender_to_receivers.get(receiver, set())
        has_2hop_cycle = any(
            sender in sender_to_receivers.get(out_neighbor, set())
            for out_neighbor in receiver_out_neighbors
        )

        # Scatter-gather: sender receives from few, sends to many (scatter)
        # or sender receives from many, sends to few (gather)
        sender_in_degree = len(sender_in_neighbors)
        sender_out_degree = len(sender_to_receivers.get(sender, set()))

        receiver_in_degree = len(receiver_to_senders.get(receiver, set()))
        receiver_out_degree = len(sender_to_receivers.get(receiver, set()))

        features.append({
            'sender_in_degree': sender_in_degree,
            'sender_out_degree': sender_out_degree,
            'receiver_in_degree': receiver_in_degree,
            'receiver_out_degree': receiver_out_degree,
            'has_direct_cycle': int(has_direct_cycle),
            'has_2hop_cycle': int(has_2hop_cycle),
            'sender_fan_ratio': sender_out_degree / (sender_in_degree + 1),  # >1 = fan-out
            'receiver_fan_ratio': receiver_out_degree / (receiver_in_degree + 1),  # <1 = fan-in
            'sender_2hop_out': len(receiver_out_neighbors),  # How far does money go?
            'receiver_2hop_in': len(sender_in_neighbors),  # Where did money come from?
        })

    return pd.DataFrame(features, index=df.index)


def extract_temporal_graph_features(df: pd.DataFrame, windows_hours: List[int] = [1, 24, 168]) -> pd.DataFrame:
    """
    Extract temporal graph features within time windows.

    For each transaction, look at the sender's/receiver's activity in past N hours.
    """
    print("Computing temporal graph features...")

    df = df.sort_values('timestamp').reset_index(drop=True)

    features = []

    # Convert to numpy for speed
    timestamps = df['timestamp'].values
    senders = df['from_account'].values
    receivers = df['to_account'].values
    amounts = df['amount_usd'].values

    # Build index for fast lookup
    sender_txn_indices = defaultdict(list)
    receiver_txn_indices = defaultdict(list)

    for i, (s, r) in enumerate(zip(senders, receivers)):
        sender_txn_indices[s].append(i)
        receiver_txn_indices[r].append(i)

    for i in tqdm(range(len(df)), desc="Temporal features"):
        ts = timestamps[i]
        sender = senders[i]
        receiver = receivers[i]

        row_features = {}

        for window in windows_hours:
            window_seconds = window * 3600

            # Sender's past transactions in window
            sender_past = [
                j for j in sender_txn_indices[sender]
                if j < i and (ts - timestamps[j]) <= window_seconds
            ]

            # Receiver's past transactions in window
            receiver_past = [
                j for j in receiver_txn_indices[receiver]
                if j < i and (ts - timestamps[j]) <= window_seconds
            ]

            row_features[f'sender_txn_count_{window}h'] = len(sender_past)
            row_features[f'sender_amount_sum_{window}h'] = amounts[sender_past].sum() if sender_past else 0
            row_features[f'sender_unique_receivers_{window}h'] = len(set(receivers[j] for j in sender_past)) if sender_past else 0

            row_features[f'receiver_txn_count_{window}h'] = len(receiver_past)
            row_features[f'receiver_amount_sum_{window}h'] = amounts[receiver_past].sum() if receiver_past else 0
            row_features[f'receiver_unique_senders_{window}h'] = len(set(senders[j] for j in receiver_past)) if receiver_past else 0

        features.append(row_features)

    return pd.DataFrame(features, index=df.index)


def extract_amount_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features about amount flows through accounts.

    - Does the account receive similar amounts to what it sends? (pass-through)
    - Is there a burst of similar amounts? (structuring)
    """
    print("Computing amount flow features...")

    # Total in/out per account
    total_out = df.groupby('from_account')['amount_usd'].sum()
    total_in = df.groupby('to_account')['amount_usd'].sum()

    # Flow ratio for sender: out / in
    sender_in = df['from_account'].map(total_in).fillna(0)
    sender_out = df['from_account'].map(total_out).fillna(0)

    receiver_in = df['to_account'].map(total_in).fillna(0)
    receiver_out = df['to_account'].map(total_out).fillna(0)

    features = pd.DataFrame({
        'sender_flow_ratio': sender_out / (sender_in + 1),  # >1 = net sender
        'receiver_flow_ratio': receiver_out / (receiver_in + 1),  # <1 = net receiver
        'sender_net_flow': sender_out - sender_in,
        'receiver_net_flow': receiver_out - receiver_in,
        'sender_total_in': sender_in,
        'sender_total_out': sender_out,
        'receiver_total_in': receiver_in,
        'receiver_total_out': receiver_out,
    }, index=df.index)

    return features


def extract_all_graph_features(df: pd.DataFrame, fast_mode: bool = True) -> pd.DataFrame:
    """
    Extract all GFP features.

    Args:
        df: Transaction dataframe with from_account, to_account, amount_usd, timestamp
        fast_mode: If True, skip expensive temporal features

    Returns:
        DataFrame with graph features
    """
    print("="*60)
    print("GRAPH FEATURE PREPROCESSOR (GFP)")
    print("="*60)

    # Start with original df
    result = df.copy()

    # 1. Fan features
    result = extract_fan_features(result)

    # 2. 2-hop features
    hop_features = extract_2hop_features(df)
    result = pd.concat([result, hop_features], axis=1)

    # 3. Amount flow features
    flow_features = extract_amount_flow_features(df)
    result = pd.concat([result, flow_features], axis=1)

    # 4. Temporal features (expensive, skip in fast mode)
    if not fast_mode:
        temporal_features = extract_temporal_graph_features(df, windows_hours=[1, 24])
        result = pd.concat([result, temporal_features], axis=1)

    print(f"\nExtracted {len(result.columns) - len(df.columns)} graph features")

    return result


if __name__ == "__main__":
    # Test on small sample
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.models.train import load_features, temporal_split

    df = load_features()
    df_train, df_val, df_test = temporal_split(df)

    # Test on small sample
    sample = df_train.head(1000)
    result = extract_all_graph_features(sample, fast_mode=True)
    print(f"\nResult shape: {result.shape}")
    print(f"New columns: {[c for c in result.columns if c not in sample.columns]}")
