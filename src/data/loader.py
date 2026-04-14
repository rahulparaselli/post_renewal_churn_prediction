"""
src/data/loader.py
──────────────────
Single place to load all four raw CSVs.
Every notebook imports from here — never reads files directly.

Key facts baked in:
  - billings     : 122,082 rows × 59 cols  — one row per customer per Renewal_Year
  - cc_calls     : 32,882  rows × 33 cols  — support/care calls (starts 2024)
  - emails       : 123,389 rows × 27 cols  — CRM email interactions (starts 2025 effectively)
  - renewal_calls: 186,534 rows × 41 cols  — renewal negotiation calls (starts 2024)

CRITICAL year mapping:
  emails.year      = Renewal_Year + 1   ← emails sent in the year AFTER the billing period
  cc_calls.Call_Year    = Renewal_Year  ← same year
  renewal_calls.Call_Year = Renewal_Year ← same year

LEAKAGE columns in renewal_calls (NEVER use as model features):
  Membership_Renewal_Decision, Churn_Category,
  Desire_To_Cancel, Customer_Renewal_Response_Category
"""

import pandas as pd
from pathlib import Path

# Project root is two levels up from this file  (src/data/loader.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"


def load_billings() -> pd.DataFrame:
    """
    Load billings.csv — the backbone table.

    One row per customer (Co_Ref) per Renewal_Year.
    Prospect_Outcome: Won | Churned | Open
    Renewal_Year range in data: 2023, 2024, 2025, 2026 (+ 5 garbage rows: 2027/2050)

    Returns raw DataFrame — call clean_billings() before use.
    """
    return pd.read_csv(RAW / "billings.csv", low_memory=False)


def load_cc_calls() -> pd.DataFrame:
    """
    Load cc_calls.csv — customer care / support calls.

    No data for 2023. Sparse in 2024 (5,431 rows). Rich in 2025 (26,705 rows).
    Sentiment score columns (start/end/overall/issues) are stored as str — need numeric conversion.
    cc_contractor_suggest_leave has messy long AI strings mixed with Yes/No.

    Returns raw DataFrame — call clean_cc_calls() before use.
    """
    return pd.read_csv(RAW / "cc_calls.csv", low_memory=False)


def load_emails() -> pd.DataFrame:
    """
    Load emails.csv — CRM email interaction log.

    CRITICAL: emails.year = Renewal_Year + 1
    Effectively no data for 2023 (0 rows) or 2024 (2 rows).
    Rich from 2025 (81,625 rows) and 2026 (41,762 rows).

    Time_to_Renewal windows: prior_year | 45_out | 14_out | pre_renewal
    crm_contractor_sentiment has messy values — needs cleaning to Positive/Neutral/Negative.
    crm_contractor_sentiment_score is stored as str — needs numeric conversion.

    Returns raw DataFrame — call clean_emails() before use.
    """
    return pd.read_csv(RAW / "emails.csv", low_memory=False)


def load_renewal_calls() -> pd.DataFrame:
    """
    Load renewal_calls.csv — renewal negotiation calls.

    No data for 2023. 85,751 rows in 2024. 89,819 rows in 2025.
    Call_Direction has two inconsistent formats: 'Outbound'/'Inbound' AND 'OUT_BOUND'/'IN_BOUND'.
    Discount_or_Waiver_Requested has 30+ messy text variants — needs cleaning to Yes/No.
    Explicit_Competitor_Mention has 'XXXX', 'UNAVAILABLE' noise values.

    ⚠ LEAKAGE columns — never use as features:
        Membership_Renewal_Decision, Churn_Category,
        Desire_To_Cancel, Customer_Renewal_Response_Category

    Returns raw DataFrame — call clean_renewal_calls() before use.
    """
    return pd.read_csv(RAW / "renewal_calls.csv", low_memory=False)


def load_all() -> dict:
    """
    Load all four raw files and return as a named dict.

    Usage:
        raw = load_all()
        bills = clean_billings(raw["billings"])
        ...
    """
    return {
        "billings":      load_billings(),
        "cc_calls":      load_cc_calls(),
        "emails":        load_emails(),
        "renewal_calls": load_renewal_calls(),
    }
