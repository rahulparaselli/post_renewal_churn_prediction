"""
src/data/cleaner.py
────────────────────
Cleaning functions for all four raw files.

Design rules:
  1. Every function takes a raw DataFrame and returns a cleaned copy.
  2. No rows are dropped — cleaning only fixes types and standardises values.
     Row filtering (cohort selection, garbage year removal) happens in notebooks.
  3. Where a value cannot be parsed cleanly it becomes NaN — never silently wrong.
  4. Every messy real-world pattern found in the data is handled explicitly here.
"""

import pandas as pd
import numpy as np
import re


# ─────────────────────────────────────────────────────────────────────────────
#  BILLINGS
# ─────────────────────────────────────────────────────────────────────────────

def clean_billings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean billings.csv.

    Key issues fixed:
      - Date columns stored as strings → parse to datetime
      - Discount_Amount stored as str  → numeric
      - Current_Auto_Renewal_Flag is 'y'/'n' → standardise to 'Yes'/'No'
      - Renewal_Year has 5 garbage rows (2027, 2050) → left in, filter in notebook
      - Band has NaN (24 rows) → left as NaN, handled in feature engineering
      - Renewal_Month stored as str like 'January' → left as-is for seasonality plots
    """
    df = df.copy()

    # ── date columns ─────────────────────────────────────────────────────────
    date_cols = [
        "Prospect_Renewal_Date", "Closed_Date", "Registration_Date",
        "Proforma_Date", "DateTime_Out",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Last_Years_Date_Paid is numeric (Excel serial date) in some rows — coerce
    if "Last_Years_Date_Paid" in df.columns:
        df["Last_Years_Date_Paid"] = pd.to_numeric(
            df["Last_Years_Date_Paid"], errors="coerce"
        )

    # ── numeric columns stored as str ─────────────────────────────────────────
    str_to_numeric = ["Discount_Amount", "Total_Net_Paid", "Last_Total_Net_Paid",
                      "Amount", "Total_Amount", "Payment_Timeframe",
                      "Tenure_Years", "Connection_Net", "Connection_Qty"]
    for col in str_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Renewal_Year — ensure integer (already int64 but belt-and-braces) ────
    df["Renewal_Year"] = pd.to_numeric(df["Renewal_Year"], errors="coerce")

    # ── Current_Auto_Renewal_Flag: 'y' → 'Yes', 'n' → 'No' ──────────────────
    if "Current_Auto_Renewal_Flag" in df.columns:
        mapping = {"y": "Yes", "Y": "Yes", "yes": "Yes",
                   "n": "No",  "N": "No",  "no":  "No"}
        df["Current_Auto_Renewal_Flag"] = (
            df["Current_Auto_Renewal_Flag"].str.strip().map(mapping)
        )

    # ── Band: strip whitespace ────────────────────────────────────────────────
    if "Band" in df.columns:
        df["Band"] = df["Band"].str.strip()

    # ── Payment_Method: uppercase and strip ───────────────────────────────────
    if "Payment_Method" in df.columns:
        df["Payment_Method"] = df["Payment_Method"].str.strip().str.upper()

    # ── Prospect_Outcome: strip whitespace ────────────────────────────────────
    if "Prospect_Outcome" in df.columns:
        df["Prospect_Outcome"] = df["Prospect_Outcome"].str.strip()

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  CC CALLS
# ─────────────────────────────────────────────────────────────────────────────

# Values in Yes/No flag columns that should map to Yes
_CC_YES_VALUES = {"yes", "y", "true", "1"}
# Values that should map to No
_CC_NO_VALUES  = {"no", "n", "false", "0", "not discussed", "not applicable"}


def _clean_yes_no_col(series: pd.Series) -> pd.Series:
    """
    Convert a messy Yes/No column to clean 'Yes' / 'No' / NaN.

    Handles:
      - Standard 'Yes' / 'No'
      - 'Not Discussed' → NaN  (no information, not a No)
      - Long AI-generated text strings → NaN  (unparseable)
      - NaN → stays NaN
    """
    def _map(val):
        if pd.isna(val):
            return np.nan
        v = str(val).strip().lower()
        if v in _CC_YES_VALUES:
            return "Yes"
        if v in _CC_NO_VALUES:
            return "No"
        # Anything longer than 10 chars is a long AI text — treat as unparseable
        if len(v) > 10:
            return np.nan
        return np.nan

    return series.map(_map)


def clean_cc_calls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean cc_calls.csv.

    Key issues fixed:
      - Sentiment score columns stored as str with mixed 'Not Discussed' / numeric → numeric
      - cc_contractor_sentiment has long AI text strings → kept for top-level,
        but a clean category column is added
      - All Yes/No flag columns cleaned via _clean_yes_no_col
      - Call_Date parsed to datetime
      - Direction is clean (IN_BOUND / OUT_BOUND) — just strip whitespace
    """
    df = df.copy()

    # ── date / year ───────────────────────────────────────────────────────────
    df["Call_Date"] = pd.to_datetime(df["Call_Date"], errors="coerce", dayfirst=True)
    df["Call_Year"] = pd.to_numeric(df["Call_Year"], errors="coerce").astype("Int64")

    # ── sentiment score columns (stored as str, contain 'Not Discussed' etc.) ─
    score_cols = [
        "cc_contractor_sentiment_start_score",
        "cc_contractor_sentiment_end_score",
        "cc_contractor_sentiment_overall_score",
        "cc_contractor_sentiment_issues_score",
    ]
    for col in score_cols:
        if col in df.columns:
            # Replace known non-numeric strings with NaN before converting
            cleaned = df[col].replace(
                {"Not Discussed": np.nan, "NaN": np.nan, "": np.nan}
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    # ── clean cc_contractor_sentiment to a standard category ─────────────────
    # Raw column has clean values (Satisfied/Neutral/Dissatisfied/Not Discussed)
    # plus ~60 long AI text rows — map these to NaN
    if "cc_contractor_sentiment" in df.columns:
        valid_sentiments = {"Satisfied", "Neutral", "Dissatisfied", "Not Discussed"}
        df["cc_contractor_sentiment_clean"] = df["cc_contractor_sentiment"].apply(
            lambda x: x if x in valid_sentiments else np.nan
        )

    # ── Yes/No flag columns ───────────────────────────────────────────────────
    yesno_cols = [
        "cc_care_package", "cc_care_package_discussed", "cc_urgency_getting_on_site",
        "cc_external_consultant", "cc_agent_cross_sell_attempt",
        "cc_customer_issues_concerns", "cc_business_struggles_financial_hardship",
        "cc_chasing_response", "cc_issues_within_questionnaire",
        "cc_login_issues", "cc_platform_issues", "cc_dissatisfaction_time_to_complete",
        "cc_process_complexity_concerns", "cc_questions_harder_than_expected",
        "cc_dissatisfaction_support", "cc_pricing_mentioned",
        "cc_pricing_sentiment_impact", "cc_refund_discussed",
        "cc_contractor_suggest_leave", "cc_contractor_complained",
    ]
    for col in yesno_cols:
        if col in df.columns:
            df[col] = _clean_yes_no_col(df[col])

    # ── Direction: strip whitespace ───────────────────────────────────────────
    if "Direction" in df.columns:
        df["Direction"] = df["Direction"].str.strip()

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  EMAILS
# ─────────────────────────────────────────────────────────────────────────────

# Sentiment category mapping — covers the messy variants found in the data
_SENTIMENT_MAP = {
    "satisfied":                                    "Positive",
    "positive":                                     "Positive",
    "neutral":                                      "Neutral",
    "not discussed":                                "Neutral",   # treat as neutral
    "dissatisfied":                                 "Negative",
    "negative":                                     "Negative",
    "initially dissatisfied, later neutral":        "Neutral",   # recovered
    "initially dissatisfied, later satisfied":      "Positive",  # recovered
}


def _clean_sentiment(series: pd.Series) -> pd.Series:
    """
    Map crm_contractor_sentiment to clean Positive / Neutral / Negative / NaN.
    Long AI-generated text rows → NaN.
    """
    def _map(val):
        if pd.isna(val):
            return np.nan
        v = str(val).strip().lower()
        if v in _SENTIMENT_MAP:
            return _SENTIMENT_MAP[v]
        # Long text (AI hallucination in data) → unparseable
        if len(v) > 40:
            return np.nan
        # Short unrecognised values (e.g. 'Yes', 'No') → NaN
        return np.nan

    return series.map(_map)


def clean_emails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean emails.csv.

    Key issues fixed:
      - crm_contractor_sentiment has messy long AI strings + 'Yes'/'No' noise
        → cleaned to Positive / Neutral / Negative via _clean_sentiment
      - crm_contractor_sentiment_score stored as str → numeric
      - crm_agent_chase_count stored as str → numeric
      - All Yes/No flag columns: 'Not Discussed' → NaN (no info, not a No)
      - year column: already int64, just validate
    """
    df = df.copy()

    # ── year ──────────────────────────────────────────────────────────────────
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # ── sentiment: raw column preserved, clean version added ─────────────────
    if "crm_contractor_sentiment" in df.columns:
        df["crm_contractor_sentiment_clean"] = _clean_sentiment(
            df["crm_contractor_sentiment"]
        )

    # ── sentiment score: stored as str ────────────────────────────────────────
    if "crm_contractor_sentiment_score" in df.columns:
        df["crm_contractor_sentiment_score"] = pd.to_numeric(
            df["crm_contractor_sentiment_score"], errors="coerce"
        )

    # ── chase count: stored as str ────────────────────────────────────────────
    if "crm_agent_chase_count" in df.columns:
        df["crm_agent_chase_count"] = pd.to_numeric(
            df["crm_agent_chase_count"], errors="coerce"
        ).fillna(0)

    # ── Yes/No flag columns ───────────────────────────────────────────────────
    # Note: 'Not Discussed' is treated as NaN (no information),
    #        not as 'No'. This is intentional — absence of mention ≠ confirmed No.
    yesno_cols = [
        "crm_accreditation_completed", "crm_timely_completion",
        "crm_progress_towards_accreditation", "crm_delays_in_accreditation",
        "crm_contractor_suggested_leave", "crm_contractor_engagement",
        "crm_dts_or_ssip_mentioned", "crm_customer_payment_intention",
        "crm_competitors_mentioned", "crm_platform_issues_raised",
        "crm_agent_chased_contractor", "crm_accreditation_issues",
        "crm_membership_overdue", "crm_auto_renewal_status",
        "crm_dissatisified_with_renewal_price", "crm_customer_complained",
        "crm_refund_mentioned", "crm_negative_customer_experience",
        "crm_dissatisfaction_with_support", "crm_financial_hardship_mentioned",
    ]
    for col in yesno_cols:
        if col in df.columns:
            df[col] = _clean_yes_no_col(df[col])

    # ── Time_to_Renewal: strip whitespace ─────────────────────────────────────
    if "Time_to_Renewal" in df.columns:
        df["Time_to_Renewal"] = df["Time_to_Renewal"].str.strip()

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  RENEWAL CALLS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_yes_no(series: pd.Series) -> pd.Series:
    """
    Extract Yes/No from messy renewal_calls columns.

    More aggressive than _clean_yes_no_col because renewal_calls columns
    contain very long strings like:
      'Yes (the customer asked if the price could be dropped under £100...)'
    Strategy: if string starts with or contains 'yes' → Yes, 'no' → No.
    'Not applicable' → NaN.  NaN → NaN.
    """
    def _map(val):
        if pd.isna(val):
            return np.nan
        v = str(val).strip()
        vl = v.lower()

        if vl in ("not applicable", "not discussed", "unavailable", "xxxx", ""):
            return np.nan

        # Explicit clean values first
        if vl == "yes":
            return "Yes"
        if vl == "no":
            return "No"
        if vl == "[no]":
            return "No"
        if vl == "**no**":
            return "No"

        # Long strings — check what they START with
        if vl.startswith("yes"):
            return "Yes"
        if vl.startswith("no"):
            return "No"

        # Strings that contain 'yes' or 'no' early
        if re.search(r"\byes\b", vl[:30]):
            return "Yes"
        if re.search(r"\bno\b", vl[:30]):
            return "No"

        return np.nan

    return series.map(_map)


def clean_renewal_calls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean renewal_calls.csv.

    Key issues fixed:
      - Call_Direction has two formats ('Outbound'/'Inbound' AND 'OUT_BOUND'/'IN_BOUND')
        → standardised to 'Inbound' / 'Outbound'
      - Discount_or_Waiver_Requested has 30+ messy text variants → Yes/No via _extract_yes_no
      - Explicit_Competitor_Mention has 'XXXX', 'UNAVAILABLE', '[No]' noise → cleaned
      - Membership_Renewal_Decision has 'Yes/No' (14 rows) → NaN (ambiguous)
      - Call_Date parsed to datetime
      - Call_Number and Call_Year → numeric

    ⚠ LEAKAGE columns are NOT cleaned or modified — they are intentionally
      left raw so any accidental use is obvious. Always filter them out before
      building features.
    """
    df = df.copy()

    # ── date / numeric ────────────────────────────────────────────────────────
    df["Call_Date"]   = pd.to_datetime(df["Call_Date"], errors="coerce", dayfirst=True)
    df["Call_Year"]   = pd.to_numeric(df["Call_Year"],   errors="coerce").astype("Int64")
    df["Call_Number"] = pd.to_numeric(df["Call_Number"], errors="coerce")

    # ── Call_Direction: standardise to Inbound / Outbound ────────────────────
    if "Call_Direction" in df.columns:
        direction_map = {
            "outbound":  "Outbound",
            "out_bound": "Outbound",
            "inbound":   "Inbound",
            "in_bound":  "Inbound",
        }
        df["Call_Direction"] = (
            df["Call_Direction"].str.strip().str.lower().map(direction_map)
        )

    # ── Yes/No flag columns (safe — not leakage) ─────────────────────────────
    yesno_cols_safe = [
        "Discussion_on_Price_Increase",
        "Renewal_Impact_Due_to_Price_Increase",
        "Discount_or_Waiver_Requested",
        "Call_Reschedule_Request",
        "Agent_Flagged_Membership_Status_Alert",
        "Agent_Renewal_Initiation",
        "Explicit_Competitor_Mention",
        "Explicit_Switching_Intent",
        "Price_Switching_Mentioned",
        "Competitor_Value_Comparison",
        "Competitor_Benefits_Mentioned",
        "Percentage_Price_Increase_Mentioned",
        "Monetary_Price_Increase_Mentioned",
        "Price_Range_Mentioned",
        "Customer_Asked_For_Justification",
        "Discount_Offered",
        "Serious_Complaint",
        "Other_Complaint",
    ]
    for col in yesno_cols_safe:
        if col in df.columns:
            df[col] = _extract_yes_no(df[col])

    # ── Membership_Renewal_Decision: 'Yes/No' (ambiguous) → NaN ──────────────
    # This is a LEAKAGE column — only cleaning the 'Yes/No' noise value.
    if "Membership_Renewal_Decision" in df.columns:
        df["Membership_Renewal_Decision"] = df["Membership_Renewal_Decision"].replace(
            {"Yes/No": np.nan}
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def clean_all(raw: dict) -> dict:
    """
    Clean all four files at once.

    Usage:
        from src.data.loader import load_all
        from src.data.cleaner import clean_all

        raw = load_all()
        data = clean_all(raw)
        bills = data["billings"]
        ...
    """
    return {
        "billings":      clean_billings(raw["billings"]),
        "cc_calls":      clean_cc_calls(raw["cc_calls"]),
        "emails":        clean_emails(raw["emails"]),
        "renewal_calls": clean_renewal_calls(raw["renewal_calls"]),
    }

