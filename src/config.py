"""Configuration constants for the AML risk scoring system."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_BRIEFS = PROJECT_ROOT / "outputs" / "briefs"

# Data files
TRANSACTIONS_FILE = DATA_RAW / "LI-Small_Trans.csv"
ACCOUNTS_FILE = DATA_RAW / "LI-Small_accounts.csv"

# Feature engineering parameters
VELOCITY_WINDOWS = {
    "1h": 1,
    "6h": 6,
    "24h": 24,
    "7d": 168,
}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_RISK_PERCENTILE = 0.05  # Top 5% for LLM investigation

# Column mappings (handle duplicate column names in raw data)
TRANSACTION_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]
