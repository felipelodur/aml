"""
Hallucination Detection & Verification for AML Investigation Briefs.

Implements multiple verification strategies:
1. Programmatic fact-checking (exact match against source data)
2. LLM-as-a-Judge (secondary LLM validates claims)
3. Self-consistency checking (multiple generations, check agreement)

References:
- SelfCheckGPT (Manakul et al., 2023)
- G-Eval (Liu et al., 2023)
- RAGAS framework for RAG evaluation
"""

import re
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class VerificationStatus(Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    HALLUCINATION = "hallucination"
    UNCERTAIN = "uncertain"


@dataclass
class Claim:
    """A single factual claim extracted from LLM output."""
    text: str
    claim_type: str  # "numeric", "categorical", "temporal", "existence"
    field_referenced: str  # Which source field this claim is about
    value_claimed: Any
    source_value: Any = None
    status: VerificationStatus = VerificationStatus.UNVERIFIED


@dataclass
class VerificationReport:
    """Full verification report for a generated brief."""
    total_claims: int
    verified_claims: int
    hallucinated_claims: int
    uncertain_claims: int
    claims: List[Claim]
    overall_score: float  # 0-1, proportion verified
    recommendation: str  # "approve", "review", "reject"


# =============================================================================
# STRATEGY 1: PROGRAMMATIC FACT-CHECKING
# =============================================================================

def extract_numeric_claims(text: str) -> List[Dict]:
    """
    Extract numeric claims from generated text.

    Handles multiple formats:
    - "16 unique recipients" / "sender_unique_receivers: 15"
    - "$16,037.19" / "Amount: $13,155.94"
    - "risk score 0.9932" / "Risk Score: 0.9979"
    - "sender_concentration: 0.625"
    - SHAP values: "+5.83 RISK"
    """
    claims = []

    # Expanded patterns for new brief format
    patterns = [
        # Network counts - both formats
        (r'(\d+)\s+unique\s+(recipients|receivers)', 'sender_unique_receivers'),
        (r'(\d+)\s+unique\s+senders', 'receiver_unique_senders'),
        (r'sender_unique_receivers[:\s]+(\d+)', 'sender_unique_receivers'),
        (r'receiver_unique_senders[:\s]+(\d+)', 'receiver_unique_senders'),

        # Amounts
        (r'\$\s*([\d,]+\.?\d*)', 'amount_usd'),
        (r'amount_usd[:\s]+\$?([\d,]+\.?\d*)', 'amount_usd'),

        # Risk score
        (r'[Rr]isk\s+[Ss]core[:\s]+(\d+\.?\d*)', 'risk_score'),
        (r'(\d+\.\d{3,})\s*\((?:top|high)', 'risk_score'),

        # Concentration ratios
        (r'sender_concentration[:\s]+(\d+\.?\d*)', 'sender_concentration'),
        (r'receiver_concentration[:\s]+(\d+\.?\d*)', 'receiver_concentration'),
        (r'concentration[:\s]+(\d+\.?\d*)', 'sender_concentration'),

        # Daily counts
        (r'sender_daily_count[:\s]+(\d+)', 'sender_daily_count'),
        (r'(\d+)\s+transactions?\s+(?:today|same[- ]day)', 'sender_daily_count'),

        # Time features
        (r'sender_time_since_last[:\s]+(\d+\.?\d*)', 'sender_time_since_last'),
        (r'(\d+\.?\d*)\s+hours?\s+(?:since|after)', 'sender_time_since_last'),

        # Gap ratio
        (r'sender_gap_ratio[:\s]+(\d+\.?\d*)', 'sender_gap_ratio'),
        (r'gap[_\s]ratio[:\s]+(\d+\.?\d*)', 'sender_gap_ratio'),

        # Z-scores
        (r'sender_amount_zscore[:\s]+(-?\d+\.?\d*)', 'sender_amount_zscore'),
        (r'z-score[:\s]+(-?\d+\.?\d*)', 'sender_amount_zscore'),

        # Binary flags
        (r'is_cross_bank[:\s]+(\d)', 'is_cross_bank'),
        (r'near_10k_threshold[:\s]+(\d)', 'near_10k_threshold'),
        (r'is_self_transfer[:\s]+(\d)', 'is_self_transfer'),
    ]

    seen_values = set()  # Avoid duplicate claims

    for pattern, field_name in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group(1).replace(',', '') if match.group(1) else ''
            if value and value.strip():
                # Create unique key to avoid duplicates
                key = (field_name, value)
                if key not in seen_values:
                    seen_values.add(key)
                    claims.append({
                        'text': match.group(0),
                        'value': value,
                        'type': field_name,  # Use actual field name
                        'position': match.start(),
                    })

    return claims


def verify_claim_against_source(
    claim: Dict,
    source_data: Dict[str, Any],
    tolerance: float = 0.05  # 5% tolerance for rounding
) -> Tuple[VerificationStatus, Any]:
    """
    Verify a single claim against source data.

    Args:
        claim: Extracted claim dict with 'type' as the field name
        source_data: Original transaction features
        tolerance: Relative tolerance for numeric comparison (5% default)

    Returns:
        (status, source_value)
    """
    field_name = claim['type']
    try:
        claimed_value = float(claim['value'])
    except ValueError:
        return VerificationStatus.UNCERTAIN, None

    # Direct field lookup (new approach - claim type IS the field name)
    if field_name in source_data:
        source_value = source_data[field_name]

        if source_value is None:
            return VerificationStatus.UNCERTAIN, None

        try:
            source_value = float(source_value)
        except (ValueError, TypeError):
            return VerificationStatus.UNCERTAIN, None

        # Check if values match within tolerance
        if source_value == 0:
            if abs(claimed_value) < 0.01:  # Both effectively zero
                return VerificationStatus.VERIFIED, source_value
            else:
                return VerificationStatus.HALLUCINATION, source_value
        else:
            relative_error = abs(claimed_value - source_value) / abs(source_value)
            if relative_error <= tolerance:
                return VerificationStatus.VERIFIED, source_value
            else:
                # Value doesn't match - this is a hallucination
                return VerificationStatus.HALLUCINATION, source_value

    # Field not in source data - can't verify
    return VerificationStatus.UNCERTAIN, None


def programmatic_verification(
    generated_text: str,
    source_data: Dict[str, Any]
) -> VerificationReport:
    """
    Run programmatic fact-checking on generated brief.

    Extracts all numeric/factual claims and verifies against source.

    Verdict logic:
    - APPROVE: Most verifiable claims match source data
    - REVIEW: Too few claims could be verified (missing source fields)
    - REJECT: Claims contradict source data (hallucinations)
    """
    extracted_claims = extract_numeric_claims(generated_text)

    claims = []
    verified = 0
    hallucinated = 0
    uncertain = 0

    for claim_dict in extracted_claims:
        status, source_value = verify_claim_against_source(claim_dict, source_data)

        claim = Claim(
            text=claim_dict['text'],
            claim_type=claim_dict['type'],
            field_referenced=claim_dict['type'],
            value_claimed=claim_dict['value'],
            source_value=source_value,
            status=status,
        )
        claims.append(claim)

        if status == VerificationStatus.VERIFIED:
            verified += 1
        elif status == VerificationStatus.HALLUCINATION:
            hallucinated += 1
        else:
            uncertain += 1

    total = len(claims)

    # Calculate score based on verifiable claims only
    verifiable = verified + hallucinated
    score = verified / verifiable if verifiable > 0 else 1.0

    # Determine recommendation
    if hallucinated > 0:
        recommendation = "reject"
    elif verifiable == 0:
        # No claims could be checked - need manual review
        recommendation = "review"
    elif verified >= verifiable * 0.8:
        # 80%+ of verifiable claims match
        recommendation = "approve"
    else:
        recommendation = "review"

    return VerificationReport(
        total_claims=total,
        verified_claims=verified,
        hallucinated_claims=hallucinated,
        uncertain_claims=uncertain,
        claims=claims,
        overall_score=score,
        recommendation=recommendation,
    )


# =============================================================================
# STRATEGY 2: LLM-AS-A-JUDGE
# =============================================================================

def build_judge_prompt(
    generated_brief: str,
    source_context: str,
) -> str:
    """Build prompt for LLM judge to verify factual consistency."""
    prompt = f"""SOURCE DATA (ground truth):
{source_context}

BRIEF TO VERIFY:
{generated_brief}

Check if the brief matches the source data. Respond with JSON only."""
    return prompt


def llm_judge_verification(
    generated_brief: str,
    source_context: str,
    llm_provider: str = "anthropic",
    use_mock: bool = True,
) -> Dict[str, Any]:
    """
    Use a second LLM to verify factual consistency.

    Args:
        generated_brief: The AI-generated investigation brief
        source_context: The original structured context provided to generator
        llm_provider: "openai" or "anthropic"
        use_mock: If True, return mock verification result

    Returns:
        Verification result dict
    """
    prompt = build_judge_prompt(generated_brief, source_context)

    if use_mock:
        return _mock_judge_response(generated_brief, source_context)

    # Real LLM call would go here
    if llm_provider == "anthropic":
        return _call_anthropic_judge(prompt)
    elif llm_provider == "openai":
        return _call_openai_judge(prompt)


def _mock_judge_response(brief: str, context: str) -> Dict[str, Any]:
    """Generate mock judge response for demo."""
    # Simple heuristic: check if key numbers from context appear in brief
    import re

    # Extract numbers from context
    context_numbers = set(re.findall(r'\d+\.?\d*', context))
    brief_numbers = set(re.findall(r'\d+\.?\d*', brief))

    # Numbers in brief but not in context = potential hallucinations
    suspicious = brief_numbers - context_numbers

    # Filter out common false positives (dates, section numbers, etc.)
    suspicious = {n for n in suspicious if float(n) > 1 and float(n) != 2022}

    verified_count = len(brief_numbers & context_numbers)
    total_count = len(brief_numbers)

    return {
        "claims_checked": total_count,
        "verified": verified_count,
        "hallucinations": len(suspicious),
        "uncertain": total_count - verified_count - len(suspicious),
        "suspicious_values": list(suspicious)[:5],
        "verdict": "APPROVE" if len(suspicious) == 0 else "NEEDS REVIEW",
        "confidence": "HIGH" if len(suspicious) == 0 else "MEDIUM",
        "explanation": f"Found {len(suspicious)} values in brief not present in source context.",
    }


LLM_JUDGE_SYSTEM_PROMPT = """You fact-check AML investigation briefs against source data.

VERIFY:
1. Amount ($) - must match within $1
2. Risk score - must match within 0.001
3. Integer features (counts) - must match exactly
4. Decimal features - allow rounding to 2-3 decimal places (0.204 = 0.20370 is OK)
5. Typology and phase must match

NOT HALLUCINATIONS:
- Rounding differences (0.625 vs 0.6253 is fine)
- SHAP contribution values (these come from model, not source data)
- Regulatory references (BSA, SAR, CFR citations)

REAL HALLUCINATIONS:
- Wrong currency (EUR when source shows USD)
- Claims about features not in source (e.g., "self-transfer" when not mentioned)
- Invented account holder names or business types
- Values completely different from source (e.g., $5000 vs $15000)

Respond with ONLY JSON:
{"verdict":"APPROVE","verified":5,"hallucinations":0,"issues":[],"summary":"All claims match source data"}"""


def _call_anthropic_judge(prompt: str) -> Dict[str, Any]:
    """Call Anthropic API for LLM-as-a-Judge verification."""
    try:
        from anthropic import Anthropic
        import json
        import re

        client = Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            system=LLM_JUDGE_SYSTEM_PROMPT,
        )

        raw_text = response.content[0].text.strip()

        # Extract JSON from response (handle markdown code blocks)
        json_text = raw_text
        if "```json" in raw_text:
            json_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            json_text = raw_text.split("```")[1].split("```")[0].strip()

        # Parse JSON
        try:
            result = json.loads(json_text)
            return {
                "verdict": result.get("verdict", "APPROVE"),
                "verified": result.get("verified", 0),
                "hallucinations": result.get("hallucinations", 0),
                "hallucination_details": result.get("issues", []),
                "confidence": "HIGH" if result.get("hallucinations", 0) == 0 else "MEDIUM",
                "summary": result.get("summary", ""),
                "raw_response": raw_text,
            }
        except json.JSONDecodeError:
            # Try to extract verdict from text
            verdict = "APPROVE" if "APPROVE" in raw_text.upper() else "REJECT" if "REJECT" in raw_text.upper() else "NEEDS REVIEW"
            return {
                "verdict": verdict,
                "raw_response": raw_text,
                "confidence": "LOW",
                "summary": raw_text[:100],
            }

    except Exception as e:
        return {"error": str(e), "verdict": "ERROR", "confidence": "LOW"}


def _call_openai_judge(prompt: str) -> Dict[str, Any]:
    """Call OpenAI API for judge verification."""
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for verification
            messages=[
                {"role": "system", "content": "You are a meticulous fact-checker. Be strict."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return {"raw_response": response.choices[0].message.content, "verdict": "NEEDS REVIEW"}
    except Exception as e:
        return {"error": str(e), "verdict": "ERROR"}


# =============================================================================
# STRATEGY 3: SELF-CONSISTENCY (Multiple Samples)
# =============================================================================

def self_consistency_check(
    context: Dict[str, Any],
    generator_func,
    n_samples: int = 3,
    agreement_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Generate multiple briefs and check for consistency.

    Based on SelfCheckGPT approach: if the model is hallucinating,
    different samples will disagree. If grounded in facts, they'll agree.

    Args:
        context: Transaction context dict
        generator_func: Function that generates a brief given context
        n_samples: Number of samples to generate
        agreement_threshold: Required agreement ratio to pass

    Returns:
        Consistency report
    """
    samples = []
    for i in range(n_samples):
        brief = generator_func(context)
        samples.append(brief)

    # Extract key claims from each sample
    all_claims = [extract_numeric_claims(s) for s in samples]

    # Check agreement on numeric values
    agreements = []
    for claim_type in ['amount', 'risk_score', 'daily_count']:
        values_per_sample = []
        for claims in all_claims:
            type_claims = [c for c in claims if c['type'] == claim_type]
            if type_claims:
                values_per_sample.append(float(type_claims[0]['value']))

        if len(values_per_sample) >= 2:
            # Check if all values are the same
            if len(set(values_per_sample)) == 1:
                agreements.append(1.0)
            else:
                # Calculate agreement ratio
                from collections import Counter
                most_common = Counter(values_per_sample).most_common(1)[0][1]
                agreements.append(most_common / len(values_per_sample))

    avg_agreement = sum(agreements) / len(agreements) if agreements else 1.0

    return {
        "n_samples": n_samples,
        "agreement_score": avg_agreement,
        "passed": avg_agreement >= agreement_threshold,
        "recommendation": "APPROVE" if avg_agreement >= agreement_threshold else "REVIEW",
    }


# =============================================================================
# COMBINED VERIFICATION PIPELINE
# =============================================================================

def full_verification_pipeline(
    generated_brief: str,
    source_data: Dict[str, Any],
    source_context_str: str,
    use_llm_judge: bool = True,
    use_mock: bool = True,
) -> Dict[str, Any]:
    """
    Run LLM-as-a-Judge verification pipeline.

    Uses a second LLM call to fact-check the generated brief against source data.

    Args:
        generated_brief: The AI-generated investigation brief
        source_data: Original transaction features dict (unused when using LLM judge)
        source_context_str: The context string that was provided to the generator
        use_llm_judge: Whether to use LLM-as-a-Judge (if False, skip verification)
        use_mock: If True, use mock responses for LLM calls

    Returns:
        Verification report dict
    """
    results = {
        "llm_judge": None,
        "overall_verdict": None,
        "confidence": None,
        "issues": [],
    }

    if not use_llm_judge:
        results["overall_verdict"] = "NOT_VERIFIED"
        results["confidence"] = "N/A"
        return results

    # Use LLM-as-a-Judge for fact-checking
    judge_result = llm_judge_verification(
        generated_brief,
        source_context_str,
        use_mock=use_mock,
    )
    results["llm_judge"] = judge_result

    # Extract verdict from LLM judge
    verdict = judge_result.get("verdict", "NEEDS REVIEW")
    confidence = judge_result.get("confidence", "MEDIUM")

    results["overall_verdict"] = verdict
    results["confidence"] = confidence

    # Add issues if any hallucinations found
    if judge_result.get("hallucinations", 0) > 0:
        results["issues"].append(f"Found {judge_result['hallucinations']} hallucination(s)")
        for detail in judge_result.get("hallucination_details", []):
            results["issues"].append(f"  - {detail}")

    if judge_result.get("summary"):
        results["summary"] = judge_result["summary"]

    if judge_result.get("error"):
        results["overall_verdict"] = "ERROR"
        results["issues"].append(f"Verification error: {judge_result['error']}")

    return results


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Example usage
    source_data = {
        "amount_usd": 16037.19,
        "risk_score": 0.9932,
        "sender_unique_receivers": 16,
        "receiver_unique_senders": 3,
        "sender_time_since_last": 0.0,
        "sender_daily_count": 6,
        "sender_gap_ratio": 0.0,
        "sender_amount_zscore": -0.21,
    }

    # Simulated generated brief (with one hallucination for demo)
    generated_brief = """
    ACH transfer of $16,037.19 from Bank 27 to Bank 16866.
    Transaction flagged with risk score 0.9932.
    Sender has 16 unique recipients indicating Fan-Out pattern.
    Transaction occurred 0.0 hours after previous activity.
    Sender executed 6 transactions today.
    """

    source_context = """
    Amount: $16,037.19
    Risk Score: 0.9932
    sender_unique_receivers: 16
    receiver_unique_senders: 3
    sender_time_since_last: 0.0
    sender_daily_count: 6
    """

    print("=" * 60)
    print("VERIFICATION PIPELINE DEMO")
    print("=" * 60)

    result = full_verification_pipeline(
        generated_brief,
        source_data,
        source_context,
        use_llm_judge=True,
        use_mock=True,
    )

    if result['llm_judge']:
        print(f"\nLLM Judge Verification:")
        print(f"  Verdict: {result['llm_judge']['verdict']}")
        print(f"  Confidence: {result['llm_judge'].get('confidence', 'N/A')}")
        print(f"  Verified: {result['llm_judge'].get('verified', 0)}")
        print(f"  Hallucinations: {result['llm_judge'].get('hallucinations', 0)}")

    print(f"\nOVERALL VERDICT: {result['overall_verdict']}")
    print(f"CONFIDENCE: {result['confidence']}")

    if result['issues']:
        print(f"\nIssues found:")
        for issue in result['issues']:
            print(f"  - {issue}")
