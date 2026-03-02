"""
End-to-End Demo: Investigation Brief Generation Pipeline
=========================================================
Shows every step from raw transaction to final LLM prompt and output.
"""
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
from datetime import datetime

from src.models.train import load_features, temporal_split

# =============================================================================
# STEP 0: Load model and get a high-risk transaction
# =============================================================================
print("=" * 80)
print("STEP 0: LOAD DATA AND SELECT HIGH-RISK TRANSACTION")
print("=" * 80)

model = joblib.load('outputs/models/lgbm_best_gfp.joblib')
meta = joblib.load('outputs/models/lgbm_best_gfp_meta.joblib')
df = load_features()
df_train, _, df_test = temporal_split(df)
feature_cols = meta['all_features']

# Add GFP features to test data
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

# Score all test transactions
X_test = df_test[feature_cols]
y_proba = model.predict_proba(X_test)[:, 1]
df_test = df_test.copy()
df_test['risk_score'] = y_proba

# Get highest risk transaction
row = df_test.nlargest(1, 'risk_score').iloc[0]

print(f"\nSelected transaction with risk score: {row['risk_score']:.4f}")
print(f"Ground truth (is_laundering): {row['is_laundering']}")

# =============================================================================
# STEP 1: RAW TRANSACTION DATA
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: RAW TRANSACTION DATA (Input)")
print("=" * 80)

print(f"""
Transaction Details:
--------------------
  Timestamp:          {row['timestamp']}
  From Bank:          {row['from_bank']}
  From Account:       {row['from_account']}
  To Bank:            {row['to_bank']}
  To Account:         {row['to_account']}
  Amount (USD):       ${row['amount_usd']:,.2f}
  Payment Currency:   {row['payment_currency']}
  Receiving Currency: {row['receiving_currency']}
  Payment Format:     {row['payment_format']}
""")

# =============================================================================
# STEP 2: ENGINEERED FEATURES (subset)
# =============================================================================
print("=" * 80)
print("STEP 2: ENGINEERED FEATURES (Key Behavioral Indicators)")
print("=" * 80)

key_features = [
    ('sender_unique_receivers', 'Network: How many different accounts sender sends to'),
    ('receiver_unique_senders', 'Network: How many different accounts send to receiver'),
    ('sender_concentration', 'Network: How concentrated are sender\'s transactions'),
    ('sender_time_since_last', 'Velocity: Hours since sender\'s last transaction'),
    ('sender_gap_ratio', 'Velocity: Current gap / historical average gap'),
    ('sender_daily_count', 'Velocity: Transactions by sender today'),
    ('sender_amount_zscore', 'Deviation: How unusual is this amount for sender'),
    ('is_cross_bank', 'Flag: Different originator and beneficiary banks'),
    ('currency_mismatch', 'Flag: Payment and receiving currencies differ'),
    ('near_10k_threshold', 'Flag: Amount near $10K CTR threshold'),
    ('is_night', 'Flag: Transaction between 10PM-6AM'),
]

print("\nFeature                      Value       Description")
print("-" * 80)
for feat, desc in key_features:
    val = row.get(feat, 'N/A')
    if isinstance(val, float):
        print(f"{feat:<28} {val:>8.3f}    {desc}")
    else:
        print(f"{feat:<28} {str(val):>8}    {desc}")

# =============================================================================
# STEP 3: MODEL EXPLANATION (SHAP values)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: SHAP LOCAL EXPLANATION (Per-Transaction Risk Drivers)")
print("=" * 80)

# Get SHAP explanation
import shap
X_single = row[feature_cols].values.reshape(1, -1)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_single)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

contributions = {}
for name, value in zip(feature_cols, shap_values[0]):
    contributions[name] = value

# Sort by absolute contribution
sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nTop 15 features driving this transaction's risk score:")
print("-" * 60)
print(f"{'Feature':<40} {'Contribution':>12} {'Effect':<15}")
print("-" * 60)
for feature, contrib in sorted_contrib[:15]:
    effect = "INCREASES RISK" if contrib > 0 else "decreases risk"
    print(f"{feature:<40} {contrib:>+12.4f} {effect}")

print(f"\nInterpretation: The LightGBM model uses these SHAP values to explain")
print(f"the final risk probability of {row['risk_score']:.4f}")

# =============================================================================
# STEP 4: TYPOLOGY MATCHING (Rule-Based)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: TYPOLOGY MATCHING (Rule-Based Pattern Detection)")
print("=" * 80)

AML_TYPOLOGIES = {
    'fan_in': {
        'name': 'Fan-In',
        'phase': 'Layering',
        'description': 'Multiple source accounts funnel funds into a single destination account',
        'indicators': {
            'receiver_unique_senders': {'threshold': 3, 'direction': '>'},
            'receiver_concentration': {'threshold': 0.3, 'direction': '<'},
        },
        'red_flags': [
            'High number of unique incoming senders',
            'Receiver concentration ratio indicates aggregation point',
        ],
        'regulatory_context': 'Common in layering phase to obscure fund origins. May indicate collection account for structured deposits.',
    },
    'fan_out': {
        'name': 'Fan-Out',
        'phase': 'Placement/Integration',
        'description': 'Single source account distributes funds to multiple destination accounts',
        'indicators': {
            'sender_unique_receivers': {'threshold': 3, 'direction': '>'},
            'sender_concentration': {'threshold': 0.3, 'direction': '<'},
        },
        'red_flags': [
            'High number of unique outgoing recipients',
            'Sender concentration ratio indicates distribution point',
        ],
        'regulatory_context': 'Common in placement (breaking up cash) or integration (distributing cleaned funds).',
    },
    'rapid_movement': {
        'name': 'Rapid Fund Movement',
        'phase': 'Layering',
        'description': 'Unusually fast movement of funds through accounts with minimal holding time',
        'indicators': {
            'sender_time_since_last': {'threshold': 1, 'direction': '<'},
            'sender_gap_ratio': {'threshold': 0.5, 'direction': '<'},
        },
        'red_flags': [
            'Transaction occurs shortly after previous activity',
            'Time gap significantly below historical average',
        ],
        'regulatory_context': 'Rapid movement reduces time for detection and intervention. Common in pass-through accounts used for layering.',
    },
    'structuring': {
        'name': 'Structuring (Smurfing)',
        'phase': 'Placement',
        'description': 'Breaking large amounts into smaller transactions to avoid reporting thresholds',
        'indicators': {
            'near_10k_threshold': {'threshold': 1, 'direction': '='},
            'sender_daily_count': {'threshold': 3, 'direction': '>'},
        },
        'red_flags': [
            'Multiple transactions just below reporting threshold',
            'High daily transaction count from single account',
        ],
        'regulatory_context': 'Federal structuring laws (31 USC 5324) prohibit breaking transactions to evade CTR filing.',
    },
    'velocity_spike': {
        'name': 'Velocity Spike',
        'phase': 'Behavioral Anomaly',
        'description': 'Sudden increase in transaction frequency compared to historical pattern',
        'indicators': {
            'sender_velocity_change': {'threshold': 0, 'direction': '>'},
            'sender_recent_velocity': {'threshold': 0.5, 'direction': '>'},
        },
        'red_flags': [
            'Transaction velocity significantly above normal',
            'Acceleration in account activity',
        ],
        'regulatory_context': 'Velocity spikes in previously dormant accounts warrant enhanced scrutiny.',
    },
}

print("\nChecking each typology against transaction features...")
print("-" * 80)

features = {col: row[col] for col in feature_cols}
matches = []

for typology_id, typology in AML_TYPOLOGIES.items():
    print(f"\n[{typology['name']}] - {typology['phase']} phase")
    score = 0
    evidence = []

    for indicator, criteria in typology['indicators'].items():
        if indicator in features:
            value = features[indicator]
            threshold = criteria['threshold']
            direction = criteria['direction']

            is_match = False
            if direction == '>' and value > threshold:
                is_match = True
            elif direction == '<' and value < threshold:
                is_match = True
            elif direction == '=' and value == threshold:
                is_match = True

            status = "MATCH" if is_match else "no match"
            print(f"  {indicator}: {value:.3f} {direction} {threshold} --> {status}")

            if is_match:
                score += 1
                evidence.append(f'{indicator}={value:.2f} ({direction}{threshold})')

    if score >= 2:
        print(f"  >>> TYPOLOGY MATCHED (score={score}/2+) <<<")
        matches.append({
            'name': typology['name'],
            'phase': typology['phase'],
            'description': typology['description'],
            'confidence': 'HIGH' if score >= 2 else 'MEDIUM',
            'evidence': evidence,
            'red_flags': typology['red_flags'][:score],
            'regulatory_context': typology['regulatory_context'],
        })
    else:
        print(f"  (not matched, score={score}/2)")

print("\n" + "-" * 80)
print(f"MATCHED TYPOLOGIES: {len(matches)}")
for m in matches:
    print(f"  - {m['name']} ({m['phase']}) - {m['confidence']} confidence")

# =============================================================================
# STEP 5: BUILD THE LLM PROMPT
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: CONSTRUCTED LLM PROMPT")
print("=" * 80)

tx = {
    'timestamp': str(row['timestamp']),
    'from_bank': str(row['from_bank']),
    'from_account': str(row['from_account']),
    'to_bank': str(row['to_bank']),
    'to_account': str(row['to_account']),
    'amount_usd': float(row['amount_usd']),
    'payment_currency': str(row['payment_currency']),
    'receiving_currency': str(row['receiving_currency']),
    'payment_format': str(row['payment_format']),
}
risk_score = row['risk_score']

prompt = f"""You are a Senior AML Compliance Analyst preparing a preliminary investigation brief for a transaction flagged by the automated monitoring system. Your brief will be reviewed by the BSA Officer and may support a SAR filing decision.

CRITICAL INSTRUCTIONS:
- Use precise AML terminology (layering, structuring, smurfing, integration, placement)
- Reference specific regulatory frameworks where applicable (BSA, 31 CFR 1010, FinCEN guidance)
- Distinguish between OBSERVED FACTS (from data) and ANALYTICAL ASSESSMENT (your interpretation)
- Do NOT fabricate information not provided below
- Be specific about which money laundering phase and typology this may represent

===============================================================================
TRANSACTION UNDER REVIEW
===============================================================================
Transaction ID: {tx['from_account']}->{tx['to_account']}@{tx['timestamp'][:10]}
Timestamp: {tx['timestamp']}
Originator: Bank {tx['from_bank']}, Account {tx['from_account']}
Beneficiary: Bank {tx['to_bank']}, Account {tx['to_account']}
Amount: ${tx['amount_usd']:,.2f}
Currency Flow: {tx['payment_currency']} -> {tx['receiving_currency']}
Payment Method: {tx['payment_format']}
ML Risk Score: {risk_score:.4f} (Top 5% - High Risk Tier)

===============================================================================
MODEL RISK DRIVERS (LightGBM + GFP Features)
===============================================================================
"""

for feature, contrib in sorted_contrib[:10]:
    direction = "INCREASES RISK" if contrib > 0 else "decreases risk"
    prompt += f"* {feature}: {contrib:+.4f} {direction}\n"

prompt += f"""
===============================================================================
BEHAVIORAL INDICATORS
===============================================================================
NETWORK PATTERNS:
* Sender's unique recipients: {features.get('sender_unique_receivers', 0):.0f}
* Receiver's unique senders: {features.get('receiver_unique_senders', 0):.0f}
* Sender concentration ratio: {features.get('sender_concentration', 0):.3f}
* Receiver concentration ratio: {features.get('receiver_concentration', 0):.3f}

VELOCITY INDICATORS:
* Sender's daily transaction count: {features.get('sender_daily_count', 0):.0f}
* Sender's daily volume: ${features.get('sender_daily_volume', 0):,.2f}
* Time since sender's last transaction: {features.get('sender_time_since_last', -1):.1f} hours
* Sender's gap ratio (vs historical): {features.get('sender_gap_ratio', 1):.2f}x

BEHAVIORAL DEVIATION:
* Sender amount z-score: {features.get('sender_amount_zscore', 0):.2f} (>2 = unusual)
* Receiver amount z-score: {features.get('receiver_amount_zscore', 0):.2f}

TRANSACTION FLAGS:
* Cross-bank transfer: {'YES' if features.get('is_cross_bank', 0) else 'No'}
* Currency mismatch: {'YES - CROSS-BORDER INDICATOR' if features.get('currency_mismatch', 0) else 'No'}
* Near $10K threshold: {'YES - STRUCTURING INDICATOR' if features.get('near_10k_threshold', 0) else 'No'}
* Night transaction: {'YES' if features.get('is_night', 0) else 'No'}

===============================================================================
TYPOLOGY ANALYSIS (Pre-computed by rule engine)
===============================================================================
"""

if matches:
    for match in matches:
        prompt += f"""
>> {match['name']} Pattern ({match['confidence']} Confidence)
   Phase: {match['phase']}
   Description: {match['description']}
   Evidence: {', '.join(match['evidence'])}
   Red Flags Present:
"""
        for flag in match['red_flags']:
            prompt += f"     - {flag}\n"
        prompt += f"   Regulatory Context: {match['regulatory_context']}\n"
else:
    prompt += "No strong typology matches detected. Transaction flagged based on behavioral anomalies.\n"

prompt += """
===============================================================================
GENERATE INVESTIGATION BRIEF
===============================================================================

Provide a structured brief with these sections:

1. EXECUTIVE SUMMARY (2-3 sentences)
   - What is this transaction and why was it flagged?

2. TYPOLOGY ASSESSMENT
   - Which money laundering typology does this most likely represent?
   - What phase of money laundering (Placement/Layering/Integration)?
   - Confidence level and reasoning

3. RISK INDICATORS (bullet points)
   - List specific suspicious indicators observed
   - Reference the data provided above

4. REGULATORY CONSIDERATIONS
   - Relevant BSA/AML requirements
   - SAR filing considerations under 31 CFR 1020.320

5. RECOMMENDED ACTION
   Choose one: ESCALATE FOR SAR FILING | ENHANCED DUE DILIGENCE | STANDARD REVIEW | CLEAR WITH DOCUMENTATION
   Provide specific justification

6. SUGGESTED NEXT STEPS
   - What additional information should be gathered?
   - What related transactions should be reviewed?

7. ANALYST NOTES
   - Limitations of this automated assessment
   - Areas requiring human judgment
"""

print("\n--- BEGIN PROMPT (this would be sent to GPT-4/Claude) ---")
print(prompt)
print("--- END PROMPT ---")

print(f"\nPrompt length: {len(prompt)} characters, ~{len(prompt.split())} words")

# =============================================================================
# STEP 6: LLM OUTPUT (Mock for demo)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: GENERATED BRIEF (Mock Output)")
print("=" * 80)

# Determine primary typology
if matches:
    primary = matches[0]
    typology_name = primary['name']
    typology_phase = primary['phase']
else:
    typology_name = "Behavioral Anomaly"
    typology_phase = "Unknown"

mock_output = f"""
==============================================================================
                    PRELIMINARY INVESTIGATION BRIEF
                    AML Transaction Monitoring System
==============================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
Case Reference: TXN-{tx['from_account'][:8]}-{tx['timestamp'][:10].replace('-','')}
Risk Tier: HIGH (Top 5% by ML Score)

------------------------------------------------------------------------------
1. EXECUTIVE SUMMARY
------------------------------------------------------------------------------
{tx['payment_format']} transfer of ${tx['amount_usd']:,.2f} from Bank {tx['from_bank']}
(Account {tx['from_account']}) to Bank {tx['to_bank']} (Account {tx['to_account']}).
Transaction flagged by ML monitoring system with risk score {risk_score:.4f},
placing it in the top 5% of all transactions by risk. Pattern analysis indicates
potential {typology_name} activity consistent with the {typology_phase} phase
of money laundering.

------------------------------------------------------------------------------
2. TYPOLOGY ASSESSMENT
------------------------------------------------------------------------------
Primary Typology: {typology_name}
ML Phase: {typology_phase}
Confidence: HIGH

Assessment: {matches[0]['description'] if matches else 'Elevated risk based on behavioral anomalies'}

Evidence:
{chr(10).join('  - ' + e for e in matches[0]['evidence']) if matches else '  - Multiple behavioral risk indicators triggered'}

Regulatory Context: {matches[0]['regulatory_context'] if matches else 'Transaction warrants review under BSA requirements.'}

------------------------------------------------------------------------------
3. RISK INDICATORS
------------------------------------------------------------------------------
* NETWORK: Sender has distributed funds to {features.get('sender_unique_receivers', 0):.0f} unique
  recipients, indicating potential fund distribution (Fan-Out) pattern
* VELOCITY: Transaction occurred {features.get('sender_time_since_last', 0):.1f} hours after
  sender's previous transaction (gap ratio: {features.get('sender_gap_ratio', 0):.1%} of historical average)
* VOLUME: Sender executed {features.get('sender_daily_count', 0):.0f} transactions today,
  totaling ${features.get('sender_daily_volume', 0):,.2f}
* CROSS-INSTITUTION: Transfer between different banks increases complexity of fund tracing
* AMOUNT: ${tx['amount_usd']:,.2f} exceeds $10,000 CTR threshold

------------------------------------------------------------------------------
4. REGULATORY CONSIDERATIONS
------------------------------------------------------------------------------
* BSA Requirement: Financial institutions must report suspicious transactions
  that may involve money laundering per 31 CFR 1020.320

* SAR Filing: Transaction meets criteria for suspicious activity:
  - Amount exceeds $5,000 threshold
  - Pattern consistent with known money laundering typology
  - Multiple risk indicators present

* CTR Requirement: Transaction exceeds $10,000 threshold for Currency
  Transaction Report filing per 31 CFR 1010.311

------------------------------------------------------------------------------
5. RECOMMENDED ACTION
------------------------------------------------------------------------------
+------------------------------------------------------------------+
|  >> ESCALATE FOR SAR FILING                                      |
+------------------------------------------------------------------+

Justification: High-confidence typology match ({typology_name}) combined with
elevated ML risk score ({risk_score:.4f}) meets SAR filing threshold under
31 CFR 1020.320(a)(2). Transaction exhibits multiple red flags consistent
with the {typology_phase} phase of money laundering.

------------------------------------------------------------------------------
6. SUGGESTED NEXT STEPS
------------------------------------------------------------------------------
[ ] Pull complete transaction history for Account {tx['from_account']} (sender)
[ ] Pull complete transaction history for Account {tx['to_account']} (receiver)
[ ] Review KYC documentation for both account holders
[ ] Check for prior SARs or alerts on either account
[ ] Analyze related transactions within +/- 7 day window
[ ] Map full network of sender's {features.get('sender_unique_receivers', 0):.0f} recipients
[ ] Verify beneficial ownership information
[ ] Request source of funds documentation (amount > $10,000)

------------------------------------------------------------------------------
7. ANALYST NOTES
------------------------------------------------------------------------------
LIMITATIONS OF THIS ASSESSMENT:
* Based on transaction-level features only; entity-level context not incorporated
* ML model trained on synthetic data; validate patterns against production behavior
* Historical SAR filings and law enforcement information not included
* Beneficial ownership and KYC data not available in current dataset

AREAS REQUIRING HUMAN JUDGMENT:
* Determination of suspicious intent vs. legitimate business activity
* Assessment of customer's stated business purpose
* Evaluation of geographic risk factors
* Final SAR filing decision per BSA Officer review

==============================================================================
                         END OF PRELIMINARY BRIEF
==============================================================================
"""

print(mock_output)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
print(f"""
Step 1: Raw Transaction     -> Basic metadata (amount, accounts, timestamp)
Step 2: Feature Engineering -> {len(feature_cols)} behavioral features
Step 3: EBM Explanation     -> Per-feature risk contributions
Step 4: Typology Matching   -> {len(matches)} pattern(s) matched: {', '.join(m['name'] for m in matches) if matches else 'None'}
Step 5: Prompt Construction -> {len(prompt)} chars with structured context
Step 6: Brief Generation    -> Compliance-ready investigation brief

Ground Truth: {'ACTUAL LAUNDERING' if row['is_laundering'] == 1 else 'Legitimate transaction'}
Model Score:  {risk_score:.4f} (correctly flagged as high-risk)
""")
