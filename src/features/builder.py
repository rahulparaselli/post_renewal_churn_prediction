"""
src/features/builder.py
────────────────────────
Builds the complete feature table for any cohort year.

The core function is build_cohort_features(bills, cc, rc, renewal_year).
Pass renewal_year=2024 for training, renewal_year=2025 for test/predict.
Both produce identical column structure.

Post-renewal cohort definition:
  A customer is included if their Closed_Date falls within 4 weeks (28 days)
  after their Prospect_Renewal_Date in the given Renewal_Year.
  Label: Prospect_Outcome == 'Churned' → 1, 'Won' → 0.
  'Open' outcomes are excluded (undecided within window).

NOTE: Email data is NOT used.
  emails.year = Renewal_Year + 1, meaning emails arrive AFTER the renewal
  decision is already made. Using them would be temporal leakage.

Feature groups:
  1. Billing structural     — band, tenure, anchorings, price, payment
  2. Billing derived        — price change %, first year flag, score delta
  3. CC call behavioural    — support call signals, sentiment trajectory
  4. Renewal call signals   — friction, discount, competitor (safe cols only)
  5. Cross-file composite   — combined signals across multiple files

LEAKAGE COLUMNS never used:
  Membership_Renewal_Decision, Churn_Category,
  Desire_To_Cancel, Customer_Renewal_Response_Category
"""

import pandas as pd
import numpy as np

LEAKAGE_COLS = [
    "Membership_Renewal_Decision",
    "Churn_Category",
    "Desire_To_Cancel",
    "Customer_Renewal_Response_Category",
]

BAND_ORDER = {
    "Band A": 1, "Band B": 2, "Band C1": 3, "Band C2": 4,
    "Band D": 5, "Band E": 6, "Band F": 7, "Band F1": 8,
    "Band F2": 9, "Band G": 10, "Band H": 11, "Band I": 12,
    "Band J": 13, "Group": 0,
}

# Columns to keep from billing data.
# POST-OUTCOME columns are EXCLUDED to prevent data leakage:
#   Total_Net_Paid         -- $0 for churned (they didn't pay)
#   Payment_Method         -- UNKNOWN for churned (no payment set up)
#   Payment_Timeframe      -- null for churned (no payment made)
#   Current_World_Pay_Token -- payment token (post-outcome)
#   Total_Renewal_Score_New -- updated after outcome decided (corr=-0.66)
#   Status_Scores          -- post-outcome status (corr=-0.66)
#   Tenure_Scores          -- post-outcome (corr=-0.43)
BILLING_KEEP_COLS = [
    "Co_Ref", "Renewal_Year", "Band", "Tenure_Years", "Current_Anchorings",
    "Connection_Qty", "Discount_Amount", "Last_Total_Net_Paid",
    "Current_Auto_Renewal_Flag", "Auto_Renewal_Score",
    "Renewal_Score_At_Release", "Sustainability_Score",
    "Last_Band", "Prospect_Outcome",
    "#_of_Connection", "Prospect_Renewal_Date", "Closed_Date",
]

DROP_BEFORE_TRAINING = [
    "Co_Ref", "Renewal_Year", "Prospect_Outcome", "outcome_post_renewal",
    "churn_label", "Band", "Current_Auto_Renewal_Flag",
    "Last_Band", "Current_Anchor_List", "Prospect_Status",
    "Prospect_Renewal_Date", "Closed_Date",
]


def build_billing_features(bills: pd.DataFrame, renewal_year: int) -> pd.DataFrame:
    """Extract customers whose deal closed within 4 weeks after Prospect_Renewal_Date.

    Includes Won and Churned outcomes only (Open excluded).
    """
    yr = bills[bills["Renewal_Year"] == renewal_year].copy()

    # ── 4-week post-renewal window ───────────────────────────────────────────
    yr["_closed_within_4w"] = (
        (yr["Closed_Date"] > yr["Prospect_Renewal_Date"]) &
        (yr["Closed_Date"] <= yr["Prospect_Renewal_Date"] + pd.Timedelta(days=28))
    )
    base = yr[
        (yr["_closed_within_4w"]) &
        (yr["Prospect_Outcome"].isin(["Won", "Churned"]))
    ][[c for c in BILLING_KEEP_COLS if c in yr.columns]].copy()

    if len(base) == 0:
        raise ValueError(f"No customers in 4-week post-renewal window for Renewal_Year={renewal_year}")

    base["is_first_year"] = base["Last_Total_Net_Paid"].isna().astype(int)
    base["auto_renewal_off"] = (~base["Current_Auto_Renewal_Flag"].str.strip().str.upper().eq("YES")).astype(int)

    # Removed (post-outcome leakage): payment_unknown, payment_timeframe_known,
    # has_world_pay_token, price_change_pct, price_increased/decreased,
    # Total_Renewal_Score_New derivatives (renewal_score_delta, low_renewal_score)

    base["anchoring_zero"] = (base["Current_Anchorings"] == 0).astype(int)
    base["anchoring_1plus"] = (base["Current_Anchorings"] >= 1).astype(int)
    base["anchoring_3plus"] = (base["Current_Anchorings"] >= 3).astype(int)

    base["band_enc"] = base["Band"].map(BAND_ORDER).fillna(0).astype(int)
    last_band_enc = base["Last_Band"].map(BAND_ORDER)
    base["band_downgraded"] = (base["band_enc"] < last_band_enc).astype(int)
    base["band_upgraded"] = (base["band_enc"] > last_band_enc).astype(int)

    q25 = base["Renewal_Score_At_Release"].quantile(0.25)
    base["low_release_score"] = (base["Renewal_Score_At_Release"] < q25).astype(int)

    base["tenure_0_1"] = (base["Tenure_Years"] <= 1).astype(int)
    base["tenure_2_3"] = ((base["Tenure_Years"] >= 2) & (base["Tenure_Years"] <= 3)).astype(int)
    base["tenure_4_7"] = ((base["Tenure_Years"] >= 4) & (base["Tenure_Years"] <= 7)).astype(int)
    base["tenure_8plus"] = (base["Tenure_Years"] >= 8).astype(int)

    return base.reset_index(drop=True)


# NOTE: build_email_features() has been removed.
# Email data (emails.year = Renewal_Year + 1) arrives AFTER the post-renewal
# decision window and would constitute temporal leakage if used as features.


def build_cc_features(cc: pd.DataFrame, co_ref_set: set, call_year: int) -> pd.DataFrame:
    """Aggregate cc_calls to one row per customer."""
    filtered = cc[(cc["Co_Ref"].isin(co_ref_set)) & (cc["Call_Year"]==call_year)].copy()
    if len(filtered) == 0:
        return pd.DataFrame(columns=["Co_Ref"])

    filtered["_sent_delta"] = (
        pd.to_numeric(filtered.get("cc_contractor_sentiment_end_score", pd.Series(dtype=float)), errors="coerce") -
        pd.to_numeric(filtered.get("cc_contractor_sentiment_start_score", pd.Series(dtype=float)), errors="coerce")
    )
    sent_col = "cc_contractor_sentiment_clean" if "cc_contractor_sentiment_clean" in filtered.columns else "cc_contractor_sentiment"

    feats = filtered.groupby("Co_Ref").agg(
        cc_call_count=("Call_Year","count"),
        cc_ever_suggest_leave=("cc_contractor_suggest_leave", lambda x: int((x=="Yes").any())),
        cc_ever_hardship=("cc_business_struggles_financial_hardship", lambda x: int((x=="Yes").any())),
        cc_ever_complained=("cc_contractor_complained", lambda x: int((x=="Yes").any())),
        cc_ever_platform_issues=("cc_platform_issues", lambda x: int((x=="Yes").any())),
        cc_ever_login_issues=("cc_login_issues", lambda x: int((x=="Yes").any())),
        cc_ever_pricing=("cc_pricing_mentioned", lambda x: int((x=="Yes").any())),
        cc_ever_refund=("cc_refund_discussed", lambda x: int((x=="Yes").any())),
        cc_pct_dissatisfied=(sent_col, lambda x: (x=="Dissatisfied").mean()),
        cc_avg_sentiment_delta=("_sent_delta","mean"),
        cc_min_overall_sentiment=("cc_contractor_sentiment_overall_score",
                                  lambda x: pd.to_numeric(x, errors="coerce").min()),
        cc_inbound_count=("Direction", lambda x: (x=="IN_BOUND").sum()),
    ).reset_index()

    feats["cc_inbound_ratio"] = (feats["cc_inbound_count"] / feats["cc_call_count"].replace(0, np.nan)).fillna(0)
    feats["cc_sentiment_worsened"] = (feats["cc_avg_sentiment_delta"] < 0).astype(int)
    feats["cc_multiple_calls"] = (feats["cc_call_count"] >= 3).astype(int)
    return feats


def build_rc_features(rc: pd.DataFrame, co_ref_set: set, call_year: int) -> pd.DataFrame:
    """Aggregate renewal_calls to one row per customer. Safe columns only."""
    safe_rc = rc.drop(columns=[c for c in LEAKAGE_COLS if c in rc.columns])
    filtered = safe_rc[(safe_rc["Co_Ref"].isin(co_ref_set)) & (safe_rc["Call_Year"]==call_year)].copy()
    if len(filtered) == 0:
        return pd.DataFrame(columns=["Co_Ref"])

    feats = filtered.groupby("Co_Ref").agg(
        rc_call_count=("Call_Year","count"),
        rc_max_call_number=("Call_Number", lambda x: pd.to_numeric(x, errors="coerce").max()),
        rc_discount_requested=("Discount_or_Waiver_Requested", lambda x: int((x=="Yes").any())),
        rc_price_discussed=("Discussion_on_Price_Increase", lambda x: int((x=="Yes").any())),
        rc_competitor_mentioned=("Explicit_Competitor_Mention", lambda x: int((x=="Yes").any())),
        rc_switching_intent=("Explicit_Switching_Intent", lambda x: int((x=="Yes").any())),
        rc_rescheduled=("Call_Reschedule_Request", lambda x: int((x=="Yes").any())),
        rc_agent_flagged=("Agent_Flagged_Membership_Status_Alert", lambda x: int((x=="Yes").any())),
        rc_price_switching=("Price_Switching_Mentioned", lambda x: int((x=="Yes").any())),
        rc_asked_justification=("Customer_Asked_For_Justification", lambda x: int((x=="Yes").any())),
        rc_discount_offered=("Discount_Offered", lambda x: int((x=="Yes").any())),
        rc_outbound_count=("Call_Direction", lambda x: (x=="Outbound").sum()),
    ).reset_index()

    feats["rc_high_friction"] = (feats["rc_max_call_number"] >= 3).astype(int)
    feats["rc_agent_chased"] = (feats["rc_outbound_count"] >= 2).astype(int)
    return feats


def build_labels(bills: pd.DataFrame, renewal_year: int) -> pd.DataFrame:
    """Build churn labels from same-year Prospect_Outcome within the 4-week
    post-renewal window.

    Won  → 0 (retained)
    Churned → 1 (churned)
    Open outcomes are excluded upstream in build_billing_features.
    """
    yr = bills[bills["Renewal_Year"] == renewal_year].copy()

    # ── apply the same 4-week post-renewal window ────────────────────────────
    yr["_closed_within_4w"] = (
        (yr["Closed_Date"] > yr["Prospect_Renewal_Date"]) &
        (yr["Closed_Date"] <= yr["Prospect_Renewal_Date"] + pd.Timedelta(days=28))
    )
    cohort = yr[
        (yr["_closed_within_4w"]) &
        (yr["Prospect_Outcome"].isin(["Won", "Churned"]))
    ][["Co_Ref", "Prospect_Outcome"]].copy()

    cohort["churn_label"] = (cohort["Prospect_Outcome"] == "Churned").astype(int)
    cohort = cohort.rename(columns={"Prospect_Outcome": "outcome_post_renewal"})
    return cohort[["Co_Ref", "churn_label", "outcome_post_renewal"]]


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-file composite features after all files are joined."""
    df = df.copy()
    z = pd.Series(0, index=df.index)

    df["any_leave_signal"] = (
        (df.get("cc_ever_suggest_leave", z)==1) |
        (df.get("rc_switching_intent", z)==1)
    ).astype(int)

    df["any_competitor_signal"] = (
        (df.get("rc_competitor_mentioned", z)==1)
    ).astype(int)

    df["any_financial_hardship"] = (
        (df.get("cc_ever_hardship", z)==1)
    ).astype(int)

    df["any_complaint"] = (
        (df.get("cc_ever_complained", z)==1)
    ).astype(int)

    neg_flag_cols = [
        "cc_ever_suggest_leave","cc_ever_hardship","cc_ever_complained",
        "cc_ever_platform_issues","cc_ever_pricing","rc_discount_requested",
        "rc_competitor_mentioned","rc_switching_intent","rc_rescheduled","rc_agent_flagged",
        "auto_renewal_off","anchoring_zero",
    ]
    existing = [c for c in neg_flag_cols if c in df.columns]
    df["total_negative_flags"] = df[existing].sum(axis=1)

    df["total_contact_count"] = (
        df.get("cc_call_count", z) +
        df.get("rc_call_count", z)
    )
    df["has_no_behavioural_data"] = (df["total_contact_count"]==0).astype(int)

    df["high_friction_score"] = (
        (df.get("rc_max_call_number", z) >= 3).astype(int) +
        (df.get("rc_discount_requested", z)==1).astype(int) +
        (df.get("rc_rescheduled", z)==1).astype(int) +
        (df.get("cc_multiple_calls", z)==1).astype(int)
    )
    df["critical_risk_billing"] = (
        (df.get("auto_renewal_off", z)==1) &
        (df.get("anchoring_zero", z)==1)
    ).astype(int)

    return df


def build_cohort_features(
    bills: pd.DataFrame,
    cc: pd.DataFrame,
    rc: pd.DataFrame,
    renewal_year: int,
    include_labels: bool = True,
) -> pd.DataFrame:
    """
    Build complete feature table for any cohort year.

    Cohort = customers whose deal closed within 4 weeks after
    Prospect_Renewal_Date, with Prospect_Outcome in {Won, Churned}.

    Uses billing, cc_calls, and renewal_calls only.
    Email data is excluded (temporal leakage — arrives after renewal decision).

    include_labels=True  → churn_label column added (Won→0, Churned→1)
    include_labels=False → production scoring mode (no label)
    """
    base = build_billing_features(bills, renewal_year)
    co_ref_set = set(base["Co_Ref"])
    n = len(base)
    print(f"[{renewal_year}] Base (4-week post-renewal window): {n:,} customers")

    cc_feats = build_cc_features(cc, co_ref_set, call_year=renewal_year)
    rc_feats = build_rc_features(rc, co_ref_set, call_year=renewal_year)
    print(f"[{renewal_year}] Coverage — cc:{len(cc_feats):,} rc:{len(rc_feats):,}")

    df = base.copy()
    df = df.merge(cc_feats,    on="Co_Ref", how="left")
    df = df.merge(rc_feats,    on="Co_Ref", how="left")

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    df = add_composite_features(df)

    assert len(df) == n, f"Row count changed: {len(df)} != {n}"
    assert df["Co_Ref"].duplicated().sum() == 0, "Duplicate Co_Refs"
    print(f"[{renewal_year}] Final: {df.shape} OK")

    if include_labels:
        labels = build_labels(bills, renewal_year)
        df = df.merge(labels, on="Co_Ref", how="left")
        print(f"[{renewal_year}] Churn rate: {df['churn_label'].mean()*100:.2f}%")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return model-ready numeric feature column names."""
    drop = set(DROP_BEFORE_TRAINING)
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
