"""
AI-Powered Investigation Assistant for AML.

Generates preliminary investigation briefs for high-risk transactions
using LLM capabilities with AML-specific terminology and typology identification.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import OUTPUTS_BRIEFS


# =============================================================================
# AML TYPOLOGY DEFINITIONS
# =============================================================================
# Thresholds calibrated against IBM AML dataset ground truth labels.
# Performance validated: FAN-IN 43% recall, FAN-OUT 52% recall, GATHER-SCATTER 25% recall.
#
# DETECTABLE with transaction-level features:
#   - FAN-IN, FAN-OUT, GATHER-SCATTER, RAPID-MOVEMENT, STRUCTURING
#
# NOT DETECTABLE (require graph analysis):
#   - CYCLE, STACK, RANDOM, BIPARTITE
# =============================================================================

AML_TYPOLOGIES = {
    "fan_in": {
        "name": "Fan-In",
        "phase": "Layering",
        "description": "Multiple source accounts funnel funds into a single destination account",
        "detection_confidence": "MEDIUM",  # 43% recall on IBM dataset
        "red_flags": [
            "High number of unique incoming senders (5+)",
            "Receiver concentration ratio above 0.5 indicates aggregation point",
            "Sender-to-receiver fan ratio below 0.6",
        ],
        "indicators": {
            # Data-driven thresholds from IBM AML dataset analysis
            # FAN-IN mean: receiver_unique_senders=9.83, sender_unique_receivers=4.76
            "receiver_unique_senders": {"threshold": 5, "direction": ">="},
            "receiver_concentration": {"threshold": 0.5, "direction": ">"},
        },
        "regulatory_context": "Common in layering phase to obscure fund origins. May indicate collection account for structured deposits (31 CFR 1010.311).",
    },
    "fan_out": {
        "name": "Fan-Out",
        "phase": "Placement/Integration",
        "description": "Single source account distributes funds to multiple destination accounts",
        "detection_confidence": "MEDIUM-HIGH",  # 52% recall on IBM dataset
        "red_flags": [
            "High number of unique outgoing recipients (5+)",
            "Sender concentration ratio above 0.4 indicates distribution point",
            "Sender-to-receiver fan ratio above 2.0",
        ],
        "indicators": {
            # Data-driven thresholds from IBM AML dataset analysis
            # FAN-OUT mean: sender_unique_receivers=13.53, receiver_unique_senders=3.43
            "sender_unique_receivers": {"threshold": 5, "direction": ">="},
            "sender_concentration": {"threshold": 0.4, "direction": ">"},
        },
        "regulatory_context": "Common in placement (breaking up cash) or integration (distributing cleaned funds). May indicate payroll fraud or kickback schemes.",
    },
    "gather_scatter": {
        "name": "Gather-Scatter / Scatter-Gather",
        "phase": "Layering",
        "description": "Account acts as intermediary: gathering from multiple sources AND scattering to multiple destinations",
        "detection_confidence": "LOW-MEDIUM",  # 25% recall - hard to distinguish from FAN patterns
        "red_flags": [
            "Account acts as both aggregation and distribution point",
            "High velocity of funds through the account",
            "Both sender and receiver have multiple counterparties (5+)",
        ],
        "indicators": {
            # Both FAN-IN and FAN-OUT characteristics present
            # Note: GATHER-SCATTER and SCATTER-GATHER are indistinguishable without temporal sequence analysis
            "receiver_unique_senders": {"threshold": 5, "direction": ">="},
            "sender_unique_receivers": {"threshold": 5, "direction": ">="},
        },
        "regulatory_context": "Classic layering pattern. Intermediary account breaks audit trail.",
    },
    "scatter_gather": {
        "name": "Scatter-Gather",
        "phase": "Layering",
        "description": "Funds scattered to multiple intermediaries then gathered to single destination",
        "detection_confidence": "MEDIUM",  # ML classifier can distinguish from gather-scatter
        "red_flags": [
            "Multiple source accounts funnel through intermediaries to single destination",
            "Receiver has unusually high number of unique senders",
            "Pattern indicates coordinated fund aggregation",
        ],
        "indicators": {
            "receiver_unique_senders": {"threshold": 8, "direction": ">="},
            "sender_unique_receivers": {"threshold": 4, "direction": ">="},
        },
        "regulatory_context": "Layering pattern where funds are first distributed then re-aggregated. Often used to obscure beneficial ownership.",
    },
    "rapid_movement": {
        "name": "Rapid Fund Movement",
        "phase": "Layering",
        "description": "Unusually fast movement of funds through accounts with minimal holding time",
        "detection_confidence": "HIGH",  # Clear behavioral signal
        "red_flags": [
            "Transaction occurs within 1 hour of previous activity",
            "Time gap significantly below historical average (gap_ratio < 0.5)",
            "Burst of transactions in short window",
        ],
        "indicators": {
            "sender_time_since_last": {"threshold": 1, "direction": "<"},  # < 1 hour
            "sender_gap_ratio": {"threshold": 0.5, "direction": "<"},
        },
        "regulatory_context": "Rapid movement reduces time for detection and intervention. Common in pass-through accounts used for layering.",
    },
    "structuring": {
        "name": "Structuring (Smurfing)",
        "phase": "Placement",
        "description": "Breaking large amounts into smaller transactions to avoid reporting thresholds",
        "detection_confidence": "MEDIUM",  # Depends on threshold proximity
        "red_flags": [
            "Multiple transactions just below $10K CTR threshold",
            "High daily transaction count from single account (3+)",
            "Amounts clustered near reporting thresholds",
        ],
        "indicators": {
            "near_10k_threshold": {"threshold": 1, "direction": "="},
            "sender_daily_count": {"threshold": 3, "direction": ">"},
        },
        "regulatory_context": "Federal structuring laws (31 USC 5324) prohibit breaking transactions to evade CTR filing. Strong indicator of willful evasion.",
    },
    "unusual_timing": {
        "name": "Unusual Timing Pattern",
        "phase": "Behavioral Anomaly",
        "description": "Transaction timing deviates from account's historical behavior pattern",
        "detection_confidence": "MEDIUM",
        "red_flags": [
            "Transaction at unusual hour for this account (6+ hours from typical)",
            "Night transaction (10PM-6AM)",
            "Weekend activity for typically weekday account",
        ],
        "indicators": {
            "sender_hour_deviation": {"threshold": 6, "direction": ">"},
            "is_night": {"threshold": 1, "direction": "="},
        },
        "regulatory_context": "Behavioral anomalies may indicate account takeover, coercion, or use by unauthorized party.",
    },
    "velocity_spike": {
        "name": "Velocity Spike",
        "phase": "Behavioral Anomaly",
        "description": "Sudden increase in transaction frequency compared to historical pattern",
        "detection_confidence": "MEDIUM",
        "red_flags": [
            "Transaction velocity significantly above normal",
            "Recent velocity above 0.5 transactions/day",
            "New account with immediate high activity",
        ],
        "indicators": {
            "sender_velocity_change": {"threshold": 0.1, "direction": ">"},
            "sender_recent_velocity": {"threshold": 0.5, "direction": ">"},
        },
        "regulatory_context": "Velocity spikes in previously dormant accounts warrant enhanced scrutiny. May indicate compromised credentials or mule account activation.",
    },
}

# Typologies that CANNOT be detected with transaction-level features
# (require graph neural networks or network analysis)
UNDETECTABLE_TYPOLOGIES = {
    "cycle": "Requires detecting loops in transaction graph",
    "stack": "Requires multi-hop path analysis across transactions",
    "random": "Requires chain detection across multiple hops",
    "bipartite": "Requires graph structure analysis",
}


# =============================================================================
# CONTEXT BUILDING
# =============================================================================

def build_transaction_context(
    row: pd.Series,
    model_contributions: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Build structured context for a single transaction.

    Args:
        row: Transaction row with features
        model_contributions: Feature contributions from EBM or SHAP
    """
    context = {
        "transaction": {
            "timestamp": str(row.get("timestamp", "N/A")),
            "from_bank": str(row.get("from_bank", "N/A")),
            "from_account": str(row.get("from_account", "N/A")),
            "to_bank": str(row.get("to_bank", "N/A")),
            "to_account": str(row.get("to_account", "N/A")),
            "amount_usd": float(row.get("amount_usd", 0)),
            "payment_currency": str(row.get("payment_currency", "N/A")),
            "receiving_currency": str(row.get("receiving_currency", "N/A")),
            "payment_format": str(row.get("payment_format", "N/A")),
        },
        "risk_score": float(row.get("risk_score", 0)),
        "features": {},
        "model_contributions": model_contributions or {},
    }

    # Extract all numeric features
    for col in row.index:
        if col not in ["is_laundering", "risk_score", "timestamp", "from_bank", "to_bank",
                       "from_account", "to_account", "payment_currency", "receiving_currency",
                       "payment_format"]:
            try:
                val = row[col]
                if pd.notna(val) and isinstance(val, (int, float, np.number)):
                    context["features"][col] = float(val)
            except (TypeError, ValueError):
                pass

    return context


def identify_typology_matches(context: Dict[str, Any], use_ml: bool = True) -> List[Dict[str, Any]]:
    """
    Match transaction patterns to known AML typologies.

    Args:
        context: Transaction context with features
        use_ml: If True, use ML classifier (84% accuracy). If False, use rules (52% accuracy).

    Returns list of potential typology matches with confidence and evidence.
    """
    features = context.get("features", {})

    # Use ML classifier if available (recommended)
    if use_ml:
        try:
            from src.llm.typology_classifier import classify_typology_ml, ML_FEATURE_COLS
            ml_features = {col: features.get(col, 0) for col in ML_FEATURE_COLS}
            result = classify_typology_ml(ml_features)

            typology_id = result["typology"].lower()
            if typology_id == "unknown":
                # No clear typology detected
                return []

            # Map to typology definition
            if typology_id in AML_TYPOLOGIES:
                typology = AML_TYPOLOGIES[typology_id]
                return [{
                    "typology_id": typology_id,
                    "name": typology["name"],
                    "phase": typology["phase"],
                    "description": typology["description"],
                    "confidence": result["confidence"],
                    "ml_confidence": result.get("probability", 0),  # Include actual probability
                    "detection_confidence": result["confidence"],
                    "matched_indicators": [],
                    "evidence": [result["reasoning"]],
                    "red_flags": typology["red_flags"][:2],
                    "regulatory_context": typology["regulatory_context"],
                }]
            return []
        except Exception as e:
            # Fall back to rule-based if ML fails
            pass

    # Rule-based matching (fallback)
    matches = []

    # Calculate fan ratio for FAN-IN/FAN-OUT distinction
    sender_unique = features.get("sender_unique_receivers", 0)
    receiver_unique = features.get("receiver_unique_senders", 0)

    if receiver_unique > 0:
        fan_ratio = sender_unique / receiver_unique
    else:
        fan_ratio = sender_unique if sender_unique > 0 else 1.0

    # Add fan_ratio to features for rule checking
    features["_fan_ratio"] = fan_ratio

    for typology_id, typology in AML_TYPOLOGIES.items():
        score = 0
        matched_indicators = []
        evidence = []

        for indicator, criteria in typology["indicators"].items():
            if indicator in features:
                value = features[indicator]
                threshold = criteria["threshold"]
                direction = criteria["direction"]

                # Check if indicator matches
                is_match = False
                if direction == ">" and value > threshold:
                    is_match = True
                elif direction == "<" and value < threshold:
                    is_match = True
                elif direction == "=" and value == threshold:
                    is_match = True
                elif direction == ">=" and value >= threshold:
                    is_match = True

                if is_match:
                    score += 1
                    matched_indicators.append(indicator)
                    evidence.append(f"{indicator}={value:.2f} ({direction}{threshold})")

        # Determine minimum required indicators
        num_indicators = len(typology["indicators"])
        min_required = min(2, num_indicators)

        # Check if pattern matches
        if score >= min_required:
            # Additional fan_ratio checks for network patterns
            should_include = True

            if typology_id == "fan_in":
                # FAN-IN: receiver-heavy pattern (fan_ratio < 0.6)
                if fan_ratio >= 0.6:
                    should_include = False
                    evidence.append(f"fan_ratio={fan_ratio:.2f} (need <0.6 for FAN-IN)")

            elif typology_id == "fan_out":
                # FAN-OUT: sender-heavy pattern (fan_ratio > 2.0)
                if fan_ratio <= 2.0:
                    should_include = False
                    evidence.append(f"fan_ratio={fan_ratio:.2f} (need >2.0 for FAN-OUT)")

            if should_include:
                # Use detection_confidence from typology definition if available
                base_confidence = typology.get("detection_confidence", "MEDIUM")
                confidence = "HIGH" if score >= num_indicators else base_confidence

                matches.append({
                    "typology_id": typology_id,
                    "name": typology["name"],
                    "phase": typology["phase"],
                    "description": typology["description"],
                    "confidence": confidence,
                    "detection_confidence": typology.get("detection_confidence", "UNKNOWN"),
                    "matched_indicators": matched_indicators,
                    "evidence": evidence,
                    "red_flags": typology["red_flags"][:max(score, 2)],
                    "regulatory_context": typology["regulatory_context"],
                })

    # Remove GATHER-SCATTER if more specific FAN-IN or FAN-OUT matched
    fan_patterns = [m for m in matches if m["typology_id"] in ("fan_in", "fan_out")]
    if fan_patterns:
        matches = [m for m in matches if m["typology_id"] != "gather_scatter"] + \
                  [m for m in matches if m["typology_id"] == "gather_scatter"][:0]  # Remove gather_scatter
        matches = [m for m in matches if m["typology_id"] != "gather_scatter"]

    # Sort by confidence and number of indicators
    confidence_order = {"HIGH": 0, "MEDIUM-HIGH": 1, "MEDIUM": 2, "LOW-MEDIUM": 3, "LOW": 4}
    matches.sort(key=lambda x: (
        confidence_order.get(x["confidence"], 5),
        -len(x["matched_indicators"])
    ))

    return matches


# =============================================================================
# PROMPT ENGINEERING - SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_FULL = """You are a Senior AML Compliance Analyst at a major financial institution.

ROLE & EXPERTISE:
- 10+ years experience in BSA/AML compliance
- Certified Anti-Money Laundering Specialist (CAMS)
- Expert in SAR filing under 31 CFR 1020.320
- Familiar with FinCEN guidance and FATF typologies

TASK:
Generate preliminary investigation briefs for transactions flagged by the ML monitoring system.
Your briefs will be reviewed by the BSA Officer and may support SAR filing decisions.

CRITICAL RULES - FOLLOW EXACTLY:
1. ONLY reference data explicitly provided in the user message
2. If information is not provided, state "Not available in source data"
3. NEVER invent account holder names, business types, or transaction purposes
4. NEVER speculate about specific criminal activities or name individuals
5. Distinguish OBSERVED FACTS (from data) vs ANALYTICAL ASSESSMENT (your interpretation)
6. Use precise AML terminology: layering, structuring, smurfing, integration, placement
7. Reference specific regulations where applicable: BSA, 31 CFR 1010, FinCEN guidance

OUTPUT FORMAT - Follow this exact structure:
```
EXECUTIVE SUMMARY:
[2-3 sentences max. What transaction, why flagged, primary concern]

TYPOLOGY ASSESSMENT:
- Most likely represents a [typology] typology.
- Phase: [Placement/Layering/Integration]
- Confidence: [HIGH/MEDIUM/LOW] ([X]% from classifier)
- Reasoning: [1-2 sentences based on PROVIDED indicators only]

RISK INDICATORS:
- [Feature name]: [value] ([direction] RISK)
[List top 5-8 risk drivers from SHAP/model contributions]

REGULATORY CONSIDERATIONS:
- BSA/AML requirements: [relevant requirement]
- SAR filing considerations: [threshold analysis]

RECOMMENDED ACTION:
[ESCALATE FOR SAR FILING | ENHANCED DUE DILIGENCE | STANDARD REVIEW | CLEAR WITH DOCUMENTATION]
Justification: [1-2 sentences]

SUGGESTED NEXT STEPS:
- [Action 1]
- [Action 2]
- [Action 3]

ANALYST NOTES:
- Limitations: [what data is missing]
- Human judgment needed: [what cannot be determined from data alone]
```

BANNED PHRASES - Never use these:
- "The account holder is likely involved in..."
- "This appears to be drug trafficking / terrorism / fraud..."
- "John Smith" or any fabricated names
- "Based on my experience..." (you have no experience beyond this data)
- Any specific criminal accusations

CRITICAL - AVOID THESE COMMON ERRORS:
1. CURRENCY: Always show amount in USD ($). Original currency is informational only.
2. FEATURE VALUES vs SHAP:
   - SHAP contributions (like +0.86) show RISK CONTRIBUTION, not the feature value
   - Always check the BEHAVIORAL FEATURES section for actual values
   - Example: is_self_transfer SHAP=+0.86 but feature value might be 0 - use the feature value!
3. SELF-TRANSFER: Only claim "self-transfer" if is_self_transfer=1 in BEHAVIORAL FEATURES.
4. MISSING DATA: If a feature is not listed, write "Not available" - do NOT invent values."""

SYSTEM_PROMPT_LOCAL = """You are an AML Compliance Analyst generating investigation briefs.

RULES:
1. ONLY use data from the user message - never invent facts
2. If data is missing, say "Not available"
3. Never fabricate names, business types, or criminal accusations
4. Currency is USD ($) unless source says otherwise
5. is_cross_bank=1 means DIFFERENT banks (not self-transfer)
6. Only list SHAP values that appear in the source data

OUTPUT exactly these sections:
EXECUTIVE SUMMARY: [2-3 sentences]
TYPOLOGY ASSESSMENT: [typology, phase, confidence, brief reasoning]
RISK INDICATORS: [bullet list of top 5 risk factors with values FROM SOURCE]
REGULATORY CONSIDERATIONS: [BSA/SAR relevance]
RECOMMENDED ACTION: [one of: ESCALATE/EDD/REVIEW/CLEAR + justification]
SUGGESTED NEXT STEPS: [3 bullet points]
ANALYST NOTES: [limitations, areas needing human judgment]"""

# Few-shot example for better output quality
FEW_SHOT_EXAMPLE = """
EXAMPLE INPUT:
Transaction: $8,500 from Bank 123 Account A to Bank 456 Account B
Risk Score: 0.92, Payment: Wire, Cross-bank: Yes
Top risk drivers: near_10k_threshold (+2.1), sender_daily_count (+1.8), sender_unique_receivers (+1.2)
Typology: Fan-Out (LOW confidence, 45%)
Features: sender_unique_receivers=12, sender_daily_count=5, near_10k_threshold=1

EXAMPLE OUTPUT:
EXECUTIVE SUMMARY:
Wire transfer of $8,500 from Bank 123 to Bank 456 flagged with risk score 0.92. Transaction exhibits potential structuring indicators (near $10K threshold) combined with fan-out distribution pattern.

TYPOLOGY ASSESSMENT:
- Most likely represents a Fan-Out typology.
- Phase: Placement/Integration
- Confidence: LOW (45% from ML classifier)
- Reasoning: Sender has 12 unique recipients indicating fund distribution, though confidence is limited.

RISK INDICATORS:
- near_10k_threshold: 1 (↑ INCREASES RISK) - Amount $8,500 is 85% of CTR threshold
- sender_daily_count: 5 (↑ INCREASES RISK) - Elevated same-day activity
- sender_unique_receivers: 12 (↑ INCREASES RISK) - High recipient diversity
- Cross-bank transfer increases tracing complexity

REGULATORY CONSIDERATIONS:
- BSA/AML requirements: Near-threshold amount warrants structuring analysis per 31 USC 5324
- SAR filing considerations: Pattern of near-threshold transactions may meet SAR criteria under 31 CFR 1020.320

RECOMMENDED ACTION:
ENHANCED DUE DILIGENCE
Justification: Near-threshold amount combined with fan-out pattern requires additional investigation before SAR determination.

SUGGESTED NEXT STEPS:
- Review sender's 30-day transaction history for threshold patterns
- Verify source of funds documentation
- Check for prior alerts on either account

ANALYST NOTES:
- Limitations: Single transaction view; entity-level risk profile not available
- Human judgment needed: Determination of legitimate business purpose vs. structuring intent
"""


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_investigation_prompt(
    context: Dict[str, Any],
    typology_matches: List[Dict],
    model_type: str = "EBM",
    compact: bool = False,
) -> str:
    """
    Build prompt for LLM investigation brief generation.

    Args:
        context: Transaction context with features
        typology_matches: Matched typologies from ML classifier
        model_type: Model type for display (LightGBM or EBM)
        compact: If True, use shorter format for local LLMs with limited context

    Returns:
        User prompt string (system prompt is separate)
    """
    tx = context["transaction"]
    features = context["features"]
    contributions = context.get("model_contributions", {})

    # Build compact prompt for local LLMs (fits in ~1500 tokens)
    if compact:
        return _build_compact_prompt(context, typology_matches, contributions)

    # Full prompt for API-based LLMs
    prompt = f"""Generate an investigation brief for this flagged transaction.

TRANSACTION DATA:
- ID: {tx['from_account']}→{tx['to_account']}
- Timestamp: {tx['timestamp']}
- Amount: ${tx['amount_usd']:,.2f}
- From: Bank {tx['from_bank']}, Account {tx['from_account']}
- To: Bank {tx['to_bank']}, Account {tx['to_account']}
- Payment Method: {tx['payment_format']}
- Currency: {tx['payment_currency']} → {tx['receiving_currency']}
- ML Risk Score: {context['risk_score']:.4f} (Top 5% = High Risk)

SHAP RISK CONTRIBUTIONS (how much each feature affects the risk score):
NOTE: These are contribution values, NOT feature values. Check BEHAVIORAL FEATURES for actual values.
"""
    # Add top model contributions (limit to top 8)
    if contributions:
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, contrib in sorted_contrib[:8]:
            direction = "increases risk" if contrib > 0 else "decreases risk"
            prompt += f"- {feature}: {contrib:+.4f} ({direction})\n"
    else:
        prompt += "- No SHAP contributions available\n"

    prompt += f"""
BEHAVIORAL FEATURES:
- sender_unique_receivers: {features.get('sender_unique_receivers', 0):.0f}
- receiver_unique_senders: {features.get('receiver_unique_senders', 0):.0f}
- sender_concentration: {features.get('sender_concentration', 0):.3f}
- receiver_concentration: {features.get('receiver_concentration', 0):.3f}
- sender_daily_count: {features.get('sender_daily_count', 0):.0f}
- sender_time_since_last: {features.get('sender_time_since_last', -1):.1f} hours
- sender_gap_ratio: {features.get('sender_gap_ratio', 1):.2f}
- sender_amount_zscore: {features.get('sender_amount_zscore', 0):.2f}

FLAGS:
- is_cross_bank: {1 if features.get('is_cross_bank', 0) else 0}
- currency_mismatch: {1 if features.get('currency_mismatch', 0) else 0}
- near_10k_threshold: {1 if features.get('near_10k_threshold', 0) else 0}
- is_night: {1 if features.get('is_night', 0) else 0}
- is_weekend: {1 if features.get('is_weekend', 0) else 0}

TYPOLOGY CLASSIFICATION (ML Classifier - 84% accuracy):
"""
    if typology_matches:
        match = typology_matches[0]
        prompt += f"""- Typology: {match['name']}
- Phase: {match['phase']}
- Confidence: {match['confidence']}
- Evidence: {', '.join(match['evidence'][:3])}
"""
    else:
        prompt += "- No typology match (behavioral anomaly only)\n"

    prompt += """
INSTRUCTIONS:
Using ONLY the data above, generate the investigation brief following the exact format from your system instructions. Do not invent any information not provided above."""

    return prompt


def _build_compact_prompt(
    context: Dict[str, Any],
    typology_matches: List[Dict],
    contributions: Dict[str, float],
) -> str:
    """Build compact prompt for local LLMs with limited context (~1000 tokens)."""
    tx = context["transaction"]
    features = context["features"]

    prompt = f"""Transaction to analyze:
Amount: ${tx['amount_usd']:,.2f} | Risk Score: {context['risk_score']:.4f}
From: Bank {tx['from_bank']} Acct {tx['from_account']}
To: Bank {tx['to_bank']} Acct {tx['to_account']}
Method: {tx['payment_format']} | Time: {tx['timestamp']}

Top Risk Drivers:
"""
    if contributions:
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, contrib in sorted_contrib[:5]:
            prompt += f"- {feature}: {contrib:+.3f}\n"

    prompt += f"""
Key Features:
- sender_unique_receivers: {features.get('sender_unique_receivers', 0):.0f}
- receiver_unique_senders: {features.get('receiver_unique_senders', 0):.0f}
- sender_daily_count: {features.get('sender_daily_count', 0):.0f}
- near_10k_threshold: {1 if features.get('near_10k_threshold', 0) else 0}
- is_cross_bank: {1 if features.get('is_cross_bank', 0) else 0}

"""
    if typology_matches:
        match = typology_matches[0]
        prompt += f"Typology: {match['name']} ({match['confidence']}) - {match['phase']}\n"
    else:
        prompt += "Typology: None matched\n"

    prompt += """
Generate brief with these sections:
EXECUTIVE SUMMARY, TYPOLOGY ASSESSMENT, RISK INDICATORS, REGULATORY CONSIDERATIONS, RECOMMENDED ACTION, SUGGESTED NEXT STEPS, ANALYST NOTES

Use ONLY data provided above. Never invent facts."""

    return prompt


# =============================================================================
# BRIEF GENERATION
# =============================================================================

def generate_brief_with_llm(
    context: Dict[str, Any],
    typology_matches: List[Dict],
    use_mock: bool = True,
    llm_provider: str = "anthropic",
    model_type: str = "EBM",
) -> str:
    """Generate investigation brief using LLM or mock response.

    Args:
        context: Transaction context
        typology_matches: List of matched typologies (from ML classifier)
        use_mock: If True, use template-based mock. If False, use LLM.
        llm_provider: 'anthropic', 'openai', or 'local' (Phi-3)
        model_type: Model type for prompt (LightGBM or EBM)

    Prompt Engineering:
        - API LLMs (OpenAI/Anthropic): Full system prompt + few-shot example + detailed user prompt
        - Local LLM (Phi-3): Compact system prompt + condensed user prompt (fits 4k context)
    """
    if use_mock:
        return _generate_mock_brief(context, typology_matches)

    # Use compact prompt for local LLMs to fit in limited context window
    compact = (llm_provider == "local")
    prompt = build_investigation_prompt(context, typology_matches, model_type, compact=compact)

    if llm_provider == "local":
        return _call_local_llm(prompt)
    elif llm_provider == "openai":
        return _call_openai(prompt)
    elif llm_provider == "anthropic":
        return _call_anthropic(prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}. Use 'local', 'anthropic', or 'openai'")


def _call_openai(prompt: str, include_example: bool = True) -> str:
    """Call OpenAI API with proper system prompt and optional few-shot example."""
    try:
        from openai import OpenAI
        client = OpenAI()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FULL},
        ]

        # Add few-shot example for better output quality
        if include_example:
            messages.append({"role": "user", "content": FEW_SHOT_EXAMPLE.split("EXAMPLE INPUT:")[1].split("EXAMPLE OUTPUT:")[0].strip()})
            messages.append({"role": "assistant", "content": FEW_SHOT_EXAMPLE.split("EXAMPLE OUTPUT:")[1].strip()})

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.2,
            max_tokens=1200,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating brief: {e}"


def _call_anthropic(prompt: str, include_example: bool = True) -> str:
    """Call Anthropic API with proper system prompt and optional few-shot example."""
    try:
        from anthropic import Anthropic
        client = Anthropic()

        messages = []

        # Add few-shot example for better output quality
        if include_example:
            example_input = FEW_SHOT_EXAMPLE.split("EXAMPLE INPUT:")[1].split("EXAMPLE OUTPUT:")[0].strip()
            example_output = FEW_SHOT_EXAMPLE.split("EXAMPLE OUTPUT:")[1].strip()
            messages.append({"role": "user", "content": f"Here's an example:\n\n{example_input}"})
            messages.append({"role": "assistant", "content": example_output})

        messages.append({"role": "user", "content": prompt})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=messages,
            system=SYSTEM_PROMPT_FULL,
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating brief: {e}"


def _call_local_llm(prompt: str, model_name: str = "microsoft/Phi-3-mini-4k-instruct") -> str:
    """Call local Hugging Face LLM for brief generation with optimized prompt."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Use cached model if available
        global _LOCAL_BRIEF_MODEL, _LOCAL_BRIEF_TOKENIZER
        if '_LOCAL_BRIEF_MODEL' not in globals() or _LOCAL_BRIEF_MODEL is None:
            print(f"    Loading local LLM for brief generation: {model_name}")
            _LOCAL_BRIEF_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _LOCAL_BRIEF_MODEL = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        # Use compact system prompt for local LLMs (saves context tokens)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LOCAL},
            {"role": "user", "content": prompt}
        ]

        chat_prompt = _LOCAL_BRIEF_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Check token count and warn if near limit
        inputs = _LOCAL_BRIEF_TOKENIZER(chat_prompt, return_tensors="pt", truncation=True, max_length=3000)
        input_len = inputs['input_ids'].shape[1]
        if input_len > 2500:
            print(f"    Warning: Input length {input_len} tokens, may truncate output")

        with torch.no_grad():
            outputs = _LOCAL_BRIEF_MODEL.generate(
                **inputs,
                max_new_tokens=1000,  # Increased for full brief
                temperature=0.3,
                do_sample=True,
                pad_token_id=_LOCAL_BRIEF_TOKENIZER.eos_token_id,
                repetition_penalty=1.1,  # Reduce repetitive text
            )

        response = _LOCAL_BRIEF_TOKENIZER.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    except Exception as e:
        return f"Error generating brief with local LLM: {e}"

# Global cache for local brief model
_LOCAL_BRIEF_MODEL = None
_LOCAL_BRIEF_TOKENIZER = None


def _generate_mock_brief(context: Dict[str, Any], typology_matches: List[Dict]) -> str:
    """Generate structured mock brief demonstrating expected output format."""
    tx = context["transaction"]
    features = context["features"]
    risk_score = context["risk_score"]

    # Determine primary typology
    if typology_matches:
        primary = typology_matches[0]
        typology_name = primary["name"]
        typology_phase = primary["phase"]
        typology_desc = primary["description"]
        confidence = primary["confidence"]
    else:
        typology_name = "Behavioral Anomaly"
        typology_phase = "Unknown"
        typology_desc = "Transaction exhibits unusual patterns warranting review"
        confidence = "LOW"

    # Build risk indicators based on features
    risk_indicators = []

    if features.get("sender_unique_receivers", 0) > 3:
        risk_indicators.append(f"Fan-out pattern: Sender has {features['sender_unique_receivers']:.0f} unique recipients (indicative of fund distribution)")

    if features.get("receiver_unique_senders", 0) > 3:
        risk_indicators.append(f"Fan-in pattern: Receiver has {features['receiver_unique_senders']:.0f} unique senders (indicative of fund aggregation)")

    if features.get("sender_time_since_last", -1) >= 0 and features.get("sender_time_since_last", 999) < 1:
        risk_indicators.append(f"Rapid movement: Only {features['sender_time_since_last']:.1f} hours since sender's last transaction")

    if features.get("sender_gap_ratio", 1) < 0.5:
        risk_indicators.append(f"Velocity anomaly: Transaction gap is {features['sender_gap_ratio']:.1%} of sender's historical average")

    if features.get("currency_mismatch", 0):
        risk_indicators.append("Cross-border indicator: Payment and receiving currencies differ")

    if features.get("is_cross_bank", 0):
        risk_indicators.append("Inter-institutional transfer increases complexity of fund tracing")

    if features.get("near_10k_threshold", 0):
        risk_indicators.append("Amount near $10,000 CTR threshold - potential structuring indicator per 31 USC 5324")

    if features.get("sender_amount_zscore", 0) > 2:
        risk_indicators.append(f"Amount deviation: Transaction is {features['sender_amount_zscore']:.1f} standard deviations above sender's typical amount")

    if features.get("is_night", 0):
        risk_indicators.append("Non-business hours transaction may indicate unauthorized access")

    if not risk_indicators:
        risk_indicators.append("Elevated ML risk score warrants manual review")

    # Determine recommended action
    if risk_score > 0.9 and typology_matches and typology_matches[0]["confidence"] == "HIGH":
        action = "ESCALATE FOR SAR FILING"
        action_reason = "High-confidence typology match combined with elevated risk score meets SAR filing threshold under 31 CFR 1020.320(a)(2)."
    elif risk_score > 0.8 or (typology_matches and len(typology_matches) > 0):
        action = "ENHANCED DUE DILIGENCE"
        action_reason = "Risk indicators warrant additional investigation including related party analysis and historical transaction review."
    elif risk_score > 0.6:
        action = "STANDARD REVIEW"
        action_reason = "Moderate risk signals present. Document findings and monitor for pattern development."
    else:
        action = "CLEAR WITH DOCUMENTATION"
        action_reason = "Risk factors present but below escalation threshold. Maintain documentation for audit trail."

    brief = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PRELIMINARY INVESTIGATION BRIEF                            ║
║                    AML Transaction Monitoring System                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
Case Reference: TXN-{tx['from_account'][:8]}-{tx['timestamp'][:10].replace('-','')}
Risk Tier: HIGH (Top 5% by ML Score)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{tx['payment_format']} transfer of ${tx['amount_usd']:,.2f} from Bank {tx['from_bank']}
(Account {tx['from_account']}) to Bank {tx['to_bank']} (Account {tx['to_account']}).
Transaction flagged by ML monitoring system with risk score {risk_score:.4f}, placing it
in the top 5% of all transactions. Pattern analysis indicates potential {typology_name}
activity consistent with the {typology_phase} phase of money laundering.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. TYPOLOGY ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Primary Typology: {typology_name}
ML Phase: {typology_phase}
Confidence: {confidence}

Assessment: {typology_desc}

{"".join([f'''
Secondary Match: {m["name"]} ({m["confidence"]})
  - {m["description"]}
''' for m in typology_matches[1:3]]) if len(typology_matches) > 1 else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. RISK INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join('▸ ' + indicator for indicator in risk_indicators)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. REGULATORY CONSIDERATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• BSA Requirement: Financial institutions must report suspicious transactions
  that may involve money laundering per 31 CFR 1020.320

• SAR Threshold: Transactions of $5,000+ that the institution knows, suspects,
  or has reason to suspect involve funds from illegal activity

• Structuring (31 USC 5324): If near-threshold amounts detected, consider whether
  pattern suggests intentional evasion of CTR requirements

• FinCEN Advisory: {"Cross-border transactions require enhanced scrutiny per FinCEN Advisory FIN-2021-A001" if features.get('currency_mismatch', 0) else "Domestic transaction - standard BSA requirements apply"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. RECOMMENDED ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
╔═══════════════════════════════════════════════════════════════════╗
║  ▶ {action:<58} ║
╚═══════════════════════════════════════════════════════════════════╝

Justification: {action_reason}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6. SUGGESTED NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Pull complete transaction history for Account {tx['from_account']} (sender)
□ Pull complete transaction history for Account {tx['to_account']} (receiver)
□ Review KYC documentation for both account holders
□ Check for prior SARs or alerts on either account
□ Analyze related transactions within ±7 day window
□ Verify beneficial ownership information
□ {"Request source of funds documentation" if tx['amount_usd'] > 10000 else "Document review findings"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

7. ANALYST NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIMITATIONS OF THIS ASSESSMENT:
• Based on transaction-level features only; entity-level context not incorporated
• ML model trained on synthetic data; validate patterns against production behavior
• Historical SAR filings and law enforcement information not included
• Beneficial ownership and KYC data not available in current dataset

AREAS REQUIRING HUMAN JUDGMENT:
• Determination of suspicious intent vs. legitimate business activity
• Assessment of customer's stated business purpose
• Evaluation of geographic risk factors
• Final SAR filing decision per BSA Officer review

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                         END OF PRELIMINARY BRIEF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    return brief


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

@dataclass
class BriefResult:
    """Result of generating and verifying an investigation brief."""
    brief: str
    context: Dict[str, Any]
    typology_matches: List[Dict]
    verification: Dict[str, Any]
    source_context_str: str


def generate_investigation_briefs(
    high_risk_df: pd.DataFrame,
    model,
    feature_columns: List[str],
    n_examples: int = 5,
    use_mock: bool = True,
    llm_provider: str = "anthropic",
    model_type: str = "LightGBM",
    verify_briefs: bool = True,
    use_llm_judge: bool = True,
) -> List[BriefResult]:
    """
    Generate and verify investigation briefs for top-risk transactions.

    Pipeline:
    1. Extract transaction context and model explanations
    2. Match against AML typologies (rule-based)
    3. Generate brief (LLM or mock)
    4. Verify brief for hallucinations (programmatic + LLM judge)
    5. Return results with verification status

    Args:
        high_risk_df: DataFrame of high-risk transactions (sorted by risk score)
        model: Trained model (LightGBM or EBM)
        feature_columns: List of feature column names
        n_examples: Number of briefs to generate
        use_mock: If True, use mock responses instead of LLM API
        llm_provider: 'openai' or 'anthropic'
        model_type: 'LightGBM' or 'EBM' for contribution extraction
        verify_briefs: If True, run hallucination verification pipeline
        use_llm_judge: If True, use LLM-as-a-Judge for verification

    Returns:
        List of BriefResult objects with briefs and verification status
    """
    # Import verification module
    from src.llm.verification import full_verification_pipeline

    print(f"\nGenerating investigation briefs for {n_examples} high-risk transactions...")
    if verify_briefs:
        print(f"  Verification: ENABLED (LLM Judge: {'ON' if use_llm_judge else 'OFF'})")
    OUTPUTS_BRIEFS.mkdir(parents=True, exist_ok=True)

    top_transactions = high_risk_df.head(n_examples)
    results = []

    verification_summary = {"approved": 0, "needs_review": 0, "rejected": 0}

    for idx, (row_idx, row) in enumerate(top_transactions.iterrows()):
        print(f"\n  [{idx + 1}/{n_examples}] Processing transaction...")

        # Get model contributions
        contributions = {}
        if model_type == "EBM":
            try:
                X_single = row[feature_columns].values.reshape(1, -1)
                local_expl = model.explain_local(pd.DataFrame(X_single, columns=feature_columns))
                local_data = local_expl.data(0)
                for name, score in zip(local_data['names'], local_data['scores']):
                    if name != 'intercept':
                        contributions[name] = score
            except Exception as e:
                print(f"    Warning: Could not get EBM contributions: {e}")
        else:
            # Use SHAP for LightGBM
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                X_single = row[feature_columns].values.reshape(1, -1)
                shap_values = explainer.shap_values(X_single)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                for name, value in zip(feature_columns, shap_values[0]):
                    contributions[name] = value
            except Exception as e:
                print(f"    Warning: Could not get SHAP contributions: {e}")

        # Build context
        context = build_transaction_context(row, contributions)

        # Identify typology matches
        typology_matches = identify_typology_matches(context)
        print(f"    Typologies matched: {len(typology_matches)}")
        for m in typology_matches:
            print(f"      - {m['name']} ({m['phase']}) [{m['confidence']}]")

        # Build source context string for verification
        source_context_str = _build_source_context_string(context, typology_matches)

        # Generate brief
        brief = generate_brief_with_llm(
            context,
            typology_matches,
            use_mock=use_mock,
            llm_provider=llm_provider,
            model_type=model_type,
        )

        # Verify brief for hallucinations
        verification = {"status": "skipped", "verdict": "NOT_VERIFIED"}
        if verify_briefs:
            print(f"    Verifying brief...")
            source_data = {col: row[col] for col in feature_columns}
            source_data['risk_score'] = row['risk_score']
            source_data['amount_usd'] = row['amount_usd']

            verification = full_verification_pipeline(
                generated_brief=brief,
                source_data=source_data,
                source_context_str=source_context_str,
                use_llm_judge=use_llm_judge,
                use_mock=use_mock,
            )

            verdict = verification.get('overall_verdict', 'UNKNOWN')
            confidence = verification.get('confidence', 'UNKNOWN')
            print(f"    Verification: {verdict} (confidence: {confidence})")

            if verification.get('issues'):
                for issue in verification['issues']:
                    print(f"      [!] {issue}")

            # Track summary
            if verdict == "APPROVE":
                verification_summary["approved"] += 1
            elif verdict == "REJECT":
                verification_summary["rejected"] += 1
            else:
                verification_summary["needs_review"] += 1

        # Create result object
        result = BriefResult(
            brief=brief,
            context=context,
            typology_matches=typology_matches,
            verification=verification,
            source_context_str=source_context_str,
        )
        results.append(result)

        # Save individual brief with verification status
        brief_path = OUTPUTS_BRIEFS / f"brief_{idx + 1}.txt"
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(_format_brief_with_verification(result))

    # Save all briefs
    all_briefs_path = OUTPUTS_BRIEFS / "all_investigation_briefs.txt"
    with open(all_briefs_path, "w", encoding="utf-8") as f:
        for i, result in enumerate(results):
            f.write(f"\n{'#'*80}\n")
            f.write(f"# BRIEF {i+1} of {len(results)}\n")
            f.write(f"{'#'*80}\n")
            f.write(_format_brief_with_verification(result))
            f.write("\n\n")

    # Save verification report
    if verify_briefs:
        report_path = OUTPUTS_BRIEFS / "verification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("VERIFICATION SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total briefs generated: {len(results)}\n")
            f.write(f"  Approved:     {verification_summary['approved']}\n")
            f.write(f"  Needs Review: {verification_summary['needs_review']}\n")
            f.write(f"  Rejected:     {verification_summary['rejected']}\n\n")

            for i, result in enumerate(results):
                v = result.verification
                f.write(f"\nBrief {i+1}:\n")
                f.write(f"  Verdict: {v.get('overall_verdict', 'N/A')}\n")
                f.write(f"  Confidence: {v.get('confidence', 'N/A')}\n")
                if v.get('summary'):
                    f.write(f"  Summary: {v['summary']}\n")
                if v.get('llm_judge'):
                    j = v['llm_judge']
                    if j.get('verified'):
                        f.write(f"  Verified claims: {j.get('verified', 0)}\n")
                    if j.get('hallucinations'):
                        f.write(f"  Hallucinations: {j.get('hallucinations', 0)}\n")
                if v.get('issues'):
                    f.write(f"  Issues:\n")
                    for issue in v['issues']:
                        f.write(f"    - {issue}\n")

    print(f"\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Briefs saved to: {OUTPUTS_BRIEFS}")
    if verify_briefs:
        print(f"\nVerification Summary:")
        print(f"  Approved:     {verification_summary['approved']}")
        print(f"  Needs Review: {verification_summary['needs_review']}")
        print(f"  Rejected:     {verification_summary['rejected']}")

    return results


def _build_source_context_string(context: Dict[str, Any], typology_matches: List[Dict]) -> str:
    """Build source context string for verification - includes all data provided to brief generator."""
    tx = context["transaction"]
    features = context["features"]
    contributions = context.get("model_contributions", {})

    # Note: amount_usd is always in USD (converted from payment_currency)
    lines = [
        "TRANSACTION DATA:",
        f"- Amount: ${tx['amount_usd']:,.2f} (USD equivalent)",
        f"- Original Currency: {tx['payment_currency']} -> {tx['receiving_currency']}",
        f"- Risk Score: {context['risk_score']:.4f}",
        f"- From Bank: {tx['from_bank']}, Account: {tx['from_account']}",
        f"- To Bank: {tx['to_bank']}, Account: {tx['to_account']}",
        f"- Payment Format: {tx['payment_format']}",
        f"- Timestamp: {tx['timestamp']}",
        "",
        "BEHAVIORAL FEATURES:",
    ]

    # Include all features that were provided
    for feat, value in sorted(features.items()):
        if isinstance(value, float):
            lines.append(f"- {feat}: {value:.4f}")
        else:
            lines.append(f"- {feat}: {value}")

    # Include SHAP contributions (these ARE part of the source data for the brief)
    if contributions:
        lines.append("")
        lines.append("SHAP CONTRIBUTIONS (model explanations - these values are valid):")
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, value in sorted_contrib[:10]:
            lines.append(f"- {name}: {value:+.4f}")

    if typology_matches:
        lines.append("")
        lines.append("TYPOLOGY CLASSIFICATION (from ML classifier):")
        for m in typology_matches:
            # Include the actual confidence details if available
            conf_str = m['confidence']
            if 'ml_confidence' in m:
                conf_str += f" ({m['ml_confidence']*100:.0f}%)"
            lines.append(f"- {m['name']} ({conf_str})")
            lines.append(f"  Phase: {m['phase']}")

    return "\n".join(lines)


def _format_brief_with_verification(result: BriefResult) -> str:
    """Format brief with verification status header."""
    v = result.verification
    verdict = v.get('overall_verdict', 'NOT_VERIFIED')
    confidence = v.get('confidence', 'N/A')

    header = []
    header.append("+" + "-" * 68 + "+")
    header.append(f"| VERIFICATION: {verdict:<15} Confidence: {confidence:<10}            |")

    if v.get('llm_judge'):
        j = v['llm_judge']
        verified = j.get('verified', 0)
        halluc = j.get('hallucinations', 0)
        if verified or halluc:
            header.append(f"| Verified: {verified}, Hallucinations: {halluc:<36} |")

    if v.get('summary'):
        summary = v['summary'][:60]
        header.append(f"| {summary:<67} |")

    header.append("+" + "-" * 68 + "+")

    return "\n".join(header) + "\n\n" + result.brief


if __name__ == "__main__":
    from src.models.train import load_model, load_features, temporal_split
    import joblib
    import argparse

    parser = argparse.ArgumentParser(description="Generate AML investigation briefs")
    parser.add_argument("--n-briefs", type=int, default=5, help="Number of briefs to generate")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification step")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM-as-a-Judge verification")
    parser.add_argument("--use-llm", action="store_true", help="Use real LLM API instead of mock")
    parser.add_argument("--provider", type=str, default="anthropic", choices=["anthropic", "openai"])
    args = parser.parse_args()

    # Load model and data
    print("=" * 60)
    print("AML INVESTIGATION BRIEF GENERATOR")
    print("=" * 60)
    print("\nLoading model and data...")

    model, meta = load_model()  # Loads lgbm_best_gfp by default
    feature_cols = meta['all_features']
    print(f"  Model: LightGBM + GFP ({len(feature_cols)} features)")

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

    # Get predictions and add risk score
    X_test = df_test[feature_cols]
    y_proba = model.predict_proba(X_test)[:, 1]
    df_test["risk_score"] = y_proba

    # Get top risk transactions
    top_risk = df_test.nlargest(100, "risk_score")

    print(f"  Test transactions: {len(df_test):,}")
    print(f"  Top risk selected: {len(top_risk)}")
    print(f"  Briefs to generate: {args.n_briefs}")

    # Generate briefs with verification
    results = generate_investigation_briefs(
        top_risk,
        model,
        feature_cols,
        n_examples=args.n_briefs,
        use_mock=not args.use_llm,
        llm_provider=args.provider,
        model_type="LightGBM",
        verify_briefs=not args.no_verify,
        use_llm_judge=not args.no_llm_judge,
    )

    print("\n" + "=" * 80)
    print("EXAMPLE BRIEF (with verification header):")
    print("=" * 80)
    print(_format_brief_with_verification(results[0]))
