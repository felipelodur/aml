# AI-Powered Investigation Assistant

## End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PART 1: ML RISK SCORING                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Raw Transactions (1M+)                                                     │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │    Feature      │    │   LightGBM      │    │   Risk Score    │         │
│  │   Engineering   │───▶│    + GFP        │───▶│   (0.0 - 1.0)   │         │
│  │  (64 features)  │    │                 │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                        │                    │
│                                                        ▼                    │
│                                               ┌─────────────────┐           │
│                                               │    Top 5%       │           │
│                                               │   (~52K txns)   │           │
│                                               └────────┬────────┘           │
│                                                        │                    │
└────────────────────────────────────────────────────────┼────────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PART 2: INVESTIGATION ASSISTANT                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Transaction   │    │  ML Typology    │    │    Typology     │         │
│  │    Features     │───▶│   Classifier    │───▶│  + Confidence   │         │
│  │                 │    │ (Random Forest) │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│         │                     84% acc                  │                    │
│         │                                              │                    │
│         ▼                                              ▼                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ SHAP Feature    │    │   Local LLM     │    │  Investigation  │         │
│  │ Contributions   │───▶│   (Phi-3)       │───▶│     Brief       │         │
│  │                 │    │                 │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Risk Scoring Model (LightGBM + GFP)

**Purpose**: Score transactions 0.0-1.0 based on fraud likelihood

**Features** (64 total):
| Category | Example Features | SHAP Contribution |
|----------|------------------|-------------------|
| Network | sender_unique_receivers, receiver_unique_senders, GFP degrees | 32% |
| Timing | time_since_last, gap_ratio, velocity_change | 28% |
| Amount | amount_usd, log_amount, near_threshold flags | 24% |
| Velocity | daily_count, daily_volume, cumulative_count | 10% |
| Deviation | amount_zscore, unusual_hour, unusual_amount | 6% |

**Output**: Top 5% by risk score (~52K transactions) sent to Investigation Assistant

---

### 2. Typology Classification (Random Forest)

**Purpose**: Classify transaction pattern into money laundering typology

**Why ML over LLM?**
| Approach | Accuracy | Speed | Notes |
|----------|----------|-------|-------|
| Rule-based | 13% | Fast | Only Fan-Out worked |
| Local LLM (Phi-3) | 50% | Slow | Struggles with rule-following |
| **ML Classifier** | **84%** | **Fast** | Best accuracy + speed |

**Output Labels**:
| Label | Description | Accuracy |
|-------|-------------|----------|
| `FAN_OUT` | Single source distributes to multiple destinations | 87% |
| `FAN_IN` | Multiple sources aggregate to single destination | 79% |
| `GATHER_SCATTER` | Intermediary gathers then scatters funds | 93% |
| `SCATTER_GATHER` | Funds scattered to intermediaries then gathered | 91% |
| `UNKNOWN` | Graph-based pattern (cycle, stack, random) - not detectable | 100% |

**Confidence Calibration**:
| Confidence | Accuracy |
|------------|----------|
| HIGH | 100% |
| MEDIUM | 97% |
| LOW | 86% |

**Training Data**: 297 labeled transactions from IBM AML dataset patterns file

---

### 3. Brief Generation (Local LLM - Phi-3)

**Purpose**: Generate natural language investigation brief

**Model**: `microsoft/Phi-3-mini-4k-instruct` (~7GB, runs on CPU)

**Why Local LLM for briefs?**
- No API key required
- Runs offline
- Good at natural language generation (unlike classification)

**Input Prompt Structure**:
```
TRANSACTION UNDER REVIEW
├── Amount, accounts, timestamp, payment method
├── Risk score and percentile

MODEL RISK DRIVERS (SHAP)
├── Top 10 features driving risk score

BEHAVIORAL INDICATORS
├── Network patterns (fan-in/out metrics)
├── Velocity indicators
├── Deviation flags

TYPOLOGY ANALYSIS (from ML classifier)
├── Classified typology + confidence
├── ML phase (Placement/Layering/Integration)

GENERATE BRIEF
├── Executive Summary
├── Typology Assessment
├── Risk Indicators
├── Regulatory Considerations
├── Recommended Action
├── Suggested Next Steps
└── Analyst Notes
```

---

## File Structure

```
aml-risk-scoring/
├── run_investigator.py              # Entry point
├── src/
│   ├── models/
│   │   └── train.py                 # LightGBM training
│   ├── features/
│   │   └── engineering.py           # 64 feature engineering
│   └── llm/
│       ├── investigator.py          # Brief generation pipeline
│       ├── typology_classifier.py   # ML + LLM typology classification
│       └── verification.py          # Hallucination detection
├── outputs/
│   ├── models/
│   │   ├── lgbm_best_gfp.joblib           # Risk scoring model
│   │   └── typology_classifier.joblib      # Typology classifier
│   └── briefs/
│       ├── brief_1.txt ... brief_N.txt     # Generated briefs
│       └── all_investigation_briefs.txt    # Combined output
└── docs/
    └── investigation_assistant.md          # This file
```

---

## Usage

```bash
# Generate briefs with local LLM (Phi-3)
python run_investigator.py --n-briefs 5 --use-llm

# Generate briefs with mock templates (faster, no LLM)
python run_investigator.py --n-briefs 5

# More options
python run_investigator.py --help
```

**Requirements for local LLM**:
- ~10GB disk space (for Phi-3 model cache)
- ~8GB RAM
- First run downloads model (~7GB)

---

## Example Output

```
+--------------------------------------------------------------------+
| VERIFICATION STATUS: NOT_VERIFIED         Confidence: N/A         |
+--------------------------------------------------------------------+

EXECUTIVE SUMMARY:
Transaction ID: 800774890→800278A90, involving a transfer of $13,155.94
from Bank 1595 to Bank 14, flagged by high ML risk score and behavioral
indicators suggesting a Fan-Out Pattern.

TYPOLOGY ASSESSMENT:
- Most likely represents a Fan-Out Pattern typology.
- Phase: Placement/Integration
- Confidence level: Low (48% confidence)
- Reasoning: High number of unique outgoing recipients...

RISK INDICATORS:
- Payment_ACH: +5.8330 (↑ RISK)
- Sender_concentration: +1.9531 (↑ RISK)
- Sender_unique_receivers: +0.7225 (↑ RISK)
...

RECOMMENDED ACTION:
- ESCALATE FOR SAR FILING
```

---

## Architecture Decisions

### Why Hybrid (ML + LLM)?

| Task | Best Tool | Reason |
|------|-----------|--------|
| Risk Scoring | LightGBM | Handles tabular data, fast, interpretable via SHAP |
| Typology Classification | Random Forest | 84% accuracy vs 50% for LLMs on rule-following |
| Brief Generation | Local LLM (Phi-3) | Good at natural language, explains findings |

### Why Not Pure LLM?

1. **Classification accuracy**: Small LLMs (0.5B-7B) struggle with conditional rule-following
2. **Speed**: ML classifiers are 100x faster than LLM inference
3. **Consistency**: ML gives same output for same input
4. **Cost**: No API fees, runs locally

### Why Not Pure ML?

1. **Natural language**: ML can't write coherent investigation briefs
2. **Flexibility**: LLM adapts explanation to context
3. **Analyst readability**: Human-readable output for compliance review
