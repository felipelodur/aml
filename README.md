# AML Risk Scoring System

ML-powered anti-money laundering transaction risk scoring and investigation assistant.

## Project Structure

```
aml-risk-scoring/
├── data/
│   ├── raw/                    # IBM AML dataset (LI-Small)
│   └── processed/              # Engineered features
├── src/
│   ├── features/
│   │   ├── engineering.py      # Feature engineering pipeline (64 features)
│   │   └── graph_features.py   # Graph Feature Preprocessor (GFP)
│   ├── models/
│   │   ├── train.py            # Model training with imbalance handling
│   │   └── evaluate.py         # Evaluation & SHAP interpretation
│   └── llm/
│       ├── investigator.py     # Brief generation pipeline
│       ├── typology_classifier.py  # ML typology classification (84% acc)
│       └── verification.py     # LLM-as-a-Judge fact-checking
├── outputs/
│   ├── models/                 # Trained model artifacts
│   ├── reports/                # Evaluation plots & interpretation
│   └── briefs/                 # Generated investigation briefs
├── docs/
│   └── investigation_assistant.md  # Architecture documentation
├── run_investigator.py         # Investigation assistant entry point
├── train_best_model.py         # Model training entry point
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key for LLM (optional - mock mode available)
export ANTHROPIC_API_KEY=your_key

# Run Part 1: Train risk scoring model
python train_best_model.py

# Run Part 2: Generate investigation briefs
python run_investigator.py --n-briefs 5 --use-llm
```

## Part 1: Transaction Risk Scoring Model

**Model:** LightGBM + Graph Feature Preprocessor (GFP)

**Performance:**
- PR-AUC: 0.73 (42% lift at 5% FPR)
- Recall at 5% FPR: 42%
- Top 5% risk tier captures 1.2% fraud (60x baseline)

**Features (70 total):**
| Category | Examples | SHAP Contribution |
|----------|----------|-------------------|
| Network | sender_unique_receivers, GFP degrees | 32% |
| Timing | time_since_last, gap_ratio | 28% |
| Amount | amount_usd, near_threshold flags | 24% |
| Velocity | daily_count, cumulative_count | 10% |
| Deviation | amount_zscore, unusual_hour | 6% |

**Key files:**
- `src/features/engineering.py` - 64 behavioral features
- `src/features/graph_features.py` - GFP network features
- `src/models/train.py` - LightGBM + SMOTE + class weights

## Part 2: AI-Powered Investigation Assistant

**Architecture:** Hybrid ML + LLM

```
High-Risk Transaction
        │
        ▼
┌───────────────┐     ┌────────────────┐     ┌────────────────┐
│  ML Typology  │────▶│  LLM Brief     │────▶│  LLM Judge     │
│  Classifier   │     │  Generation    │     │  Verification  │
│  (84% acc)    │     │  (Claude/Phi-3)│     │  (fact-check)  │
└───────────────┘     └────────────────┘     └────────────────┘
```

**Typology Classification:**
| Typology | Accuracy |
|----------|----------|
| FAN_OUT | 87% |
| FAN_IN | 79% |
| GATHER_SCATTER | 93% |
| SCATTER_GATHER | 91% |

**LLM Options:**
- `--llm-provider anthropic` - Claude API (recommended)
- `--llm-provider openai` - GPT-4 API
- `--llm-provider local` - Phi-3 (offline, ~8GB RAM)
- Mock mode (default) - Template-based, no API needed

**Verification:**
- LLM-as-a-Judge: Second LLM call fact-checks the brief
- Catches hallucinations (wrong values, fabricated claims)
- Returns APPROVE/REJECT verdict with confidence

**Key files:**
- `src/llm/investigator.py` - Brief generation with prompt engineering
- `src/llm/typology_classifier.py` - ML + rule-based classification
- `src/llm/verification.py` - LLM-as-a-Judge verification

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| LightGBM + GFP | Best PR-AUC; GFP adds network context |
| ML Typology Classifier | 84% accuracy vs 50% for LLMs on rule-following |
| LLM-as-a-Judge | Catches hallucinations before analyst review |
| Hybrid ML + LLM | ML for accuracy, LLM for natural language |
| PR-AUC metric | Better for imbalanced compliance data |

## Usage Examples

```bash
# Generate 5 briefs with Claude API + verification
python run_investigator.py --n-briefs 5 --use-llm --llm-provider anthropic

# Generate briefs with local LLM (no API key needed)
python run_investigator.py --n-briefs 5 --use-llm --llm-provider local

# Mock mode (fastest, template-based)
python run_investigator.py --n-briefs 5

# Skip verification (faster)
python run_investigator.py --n-briefs 5 --use-llm --no-verify
```

## Requirements

- Python 3.10+
- ~8GB RAM (for local LLM)
- ~10GB disk (for Phi-3 model cache, optional)

## Limitations

- Synthetic IBM dataset may not capture all production patterns
- Local LLM (Phi-3) less capable than API models
- LLM briefs require human review before SAR filing
- Graph features limited to 1-hop connections
