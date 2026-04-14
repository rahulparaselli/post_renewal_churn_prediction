"""
src/features/builder.py
────────────────────────
Builds the complete feature table for any cohort year.

The core function is build_cohort_features(bills, emails, cc, rc, renewal_year).
Pass renewal_year=2024 for training, renewal_year=2025 for test/predict.
Both produce identical column structure.

Feature groups:
  1. Billing structural     — band, tenure, anchorings, price, payment
  2. Billing derived        — price change %, first year flag, score delta
  3. Email behavioural      — sentiment, flags, engagement (prior_year window)
  4. CC call behavioural    — support call signals, sentiment trajectory
  5. Renewal call signals   — friction, discount, competitor (safe cols only)
  6. Cross-file composite   — combined signals across multiple files

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

BILLING_KEEP_COLS = [
    "Co_Ref", "Renewal_Year", "Band", "Tenure_Years", "Current_Anchorings",
    "Connection_Qty", "Discount_Amount", "Total_Net_Paid", "Last_Total_Net_Paid",
    "Current_Auto_Renewal_Flag", "Payment_Method", "Auto_Renewal_Score",
    "Total_Renewal_Score_New", "Renewal_Score_At_Release", "Sustainability_Score",
    "Last_Band", "Prospect_Outcome", "Current_World_Pay_Token", "Payment_Timeframe",
    "#_of_Connection",
]

DROP_BEFORE_TRAINING = [
    "Co_Ref", "Renewal_Year", "Prospect_Outcome", "outcome_next_year",
    "churn_label", "Band", "Payment_Method", "Current_Auto_Renewal_Flag",
    "Last_Band", "Current_World_Pay_Token", "Current_Anchor_List", "Prospect_Status",
]


def build_billing_features(bills: pd.DataFrame, renewal_year: int) -> pd.DataFrame:
    """Extract Won customers and build all billing-derived features."""
    base = bills[
        (bills["Renewal_Year"] == renewal_year) &
        (bills["Prospect_Outcome"] == "Won")
    ][[c for c in BILLING_KEEP_COLS if c in bills.columns]].copy()

    if len(base) == 0:
        raise ValueError(f"No Won customers for Renewal_Year={renewal_year}")

    base["is_first_year"] = base["Last_Total_Net_Paid"].isna().astype(int)
    base["payment_unknown"] = base["Payment_Method"].str.strip().str.upper().eq("UNKNOWN").astype(int)
    base["auto_renewal_off"] = (~base["Current_Auto_Renewal_Flag"].str.strip().str.upper().eq("YES")).astype(int)
    base["has_world_pay_token"] = base["Current_World_Pay_Token"].notna().astype(int)
    base["payment_timeframe_known"] = base["Payment_Timeframe"].notna().astype(int)
    base["anchoring_zero"] = (base["Current_Anchorings"] == 0).astype(int)
    base["anchoring_1plus"] = (base["Current_Anchorings"] >= 1).astype(int)
    base["anchoring_3plus"] = (base["Current_Anchorings"] >= 3).astype(int)

    base["band_enc"] = base["Band"].map(BAND_ORDER).fillna(0).astype(int)
    last_band_enc = base["Last_Band"].map(BAND_ORDER)
    base["band_downgraded"] = (base["band_enc"] < last_band_enc).astype(int)
    base["band_upgraded"] = (base["band_enc"] > last_band_enc).astype(int)

    safe_last = base["Last_Total_Net_Paid"].replace(0, np.nan)
    base["price_change_pct"] = ((base["Total_Net_Paid"] - base["Last_Total_Net_Paid"]) / safe_last).fillna(0).clip(-1, 5)
    base["price_increased"] = (base["price_change_pct"] > 0.01).astype(int)
    base["price_increase_10pct"] = (base["price_change_pct"] > 0.10).astype(int)
    base["price_increase_20pct"] = (base["price_change_pct"] > 0.20).astype(int)
    base["price_decreased"] = (base["price_change_pct"] < -0.01).astype(int)

    base["renewal_score_delta"] = (base["Total_Renewal_Score_New"] - base["Renewal_Score_At_Release"]).fillna(0)
    q25 = base["Total_Renewal_Score_New"].quantile(0.25)
    base["low_renewal_score"] = (base["Total_Renewal_Score_New"] < q25).astype(int)

    base["tenure_0_1"] = (base["Tenure_Years"] <= 1).astype(int)
    base["tenure_2_3"] = ((base["Tenure_Years"] >= 2) & (base["Tenure_Years"] <= 3)).astype(int)
    base["tenure_4_7"] = ((base["Tenure_Years"] >= 4) & (base["Tenure_Years"] <= 7)).astype(int)
    base["tenure_8plus"] = (base["Tenure_Years"] >= 8).astype(int)

    return base.reset_index(drop=True)


def build_email_features(emails: pd.DataFrame, co_ref_set: set, time_window: str = "prior_year") -> pd.DataFrame:
    """Aggregate emails to one row per customer."""
    filtered = emails[
        (emails["Co_Ref"].isin(co_ref_set)) &
        (emails["Time_to_Renewal"] == time_window)
    ].copy()

    if len(filtered) == 0:
        return pd.DataFrame(columns=["Co_Ref"])

    sent_col = "crm_contractor_sentiment_clean" if "crm_contractor_sentiment_clean" in filtered.columns else "crm_contractor_sentiment"

    feats = filtered.groupby("Co_Ref").agg(
        email_ever_suggested_leave=("crm_contractor_suggested_leave", lambda x: int((x=="Yes").any())),
        email_ever_competitor=("crm_competitors_mentioned", lambda x: int((x=="Yes").any())),
        email_ever_price_dissatisfied=("crm_dissatisified_with_renewal_price", lambda x: int((x=="Yes").any())),
        email_ever_complained=("crm_customer_complained", lambda x: int((x=="Yes").any())),
        email_ever_financial_hardship=("crm_financial_hardship_mentioned", lambda x: int((x=="Yes").any())),
        email_ever_refund_mentioned=("crm_refund_mentioned", lambda x: int((x=="Yes").any())),
        email_ever_overdue=("crm_membership_overdue", lambda x: int((x=="Yes").any())),
        email_ever_negative_exp=("crm_negative_customer_experience", lambda x: int((x=="Yes").any())),
        email_ever_dissatisfied_support=("crm_dissatisfaction_with_support", lambda x: int((x=="Yes").any())),
        email_pct_negative=(sent_col, lambda x: (x=="Negative").mean()),
        email_pct_positive=(sent_col, lambda x: (x=="Positive").mean()),
        email_avg_sentiment_score=("crm_contractor_sentiment_score", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        email_min_sentiment_score=("crm_contractor_sentiment_score", lambda x: pd.to_numeric(x, errors="coerce").min()),
        email_count=("Co_Ref", "count"),
        email_agent_chase_total=("crm_agent_chase_count", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        email_pct_not_engaged=("crm_contractor_engagement", lambda x: (x=="No").mean()),
        email_ever_platform_issue=("crm_platform_issues_raised", lambda x: int((x=="Yes").any())),
    ).reset_index()

    feats["email_high_risk_flag"] = (
        (feats["email_ever_suggested_leave"]==1) |
        (feats["email_ever_competitor"]==1) |
        (feats["email_pct_negative"]>0.5)
    ).astype(int)

    feats["email_total_negative_flags"] = feats[[
        "email_ever_suggested_leave","email_ever_competitor","email_ever_price_dissatisfied",
        "email_ever_complained","email_ever_financial_hardship","email_ever_refund_mentioned",
        "email_ever_overdue","email_ever_negative_exp"
    ]].sum(axis=1)

    return feats


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
    """Build churn labels by looking up next year outcome."""
    next_year = renewal_year + 1
    next_bills = bills[bills["Renewal_Year"]==next_year][["Co_Ref","Prospect_Outcome"]]
    next_bills = next_bills.rename(columns={"Prospect_Outcome":"outcome_next_year"})
    base_ids = bills[(bills["Renewal_Year"]==renewal_year) & (bills["Prospect_Outcome"]=="Won")][["Co_Ref"]]
    merged = base_ids.merge(next_bills, on="Co_Ref", how="left")
    merged["churn_label"] = (merged["outcome_next_year"] != "Won").astype(int)
    return merged[["Co_Ref","churn_label","outcome_next_year"]]


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-file composite features after all files are joined."""
    df = df.copy()
    z = pd.Series(0, index=df.index)

    df["any_leave_signal"] = (
        (df.get("email_ever_suggested_leave", z)==1) |
        (df.get("cc_ever_suggest_leave", z)==1) |
        (df.get("rc_switching_intent", z)==1)
    ).astype(int)

    df["any_competitor_signal"] = (
        (df.get("email_ever_competitor", z)==1) |
        (df.get("rc_competitor_mentioned", z)==1)
    ).astype(int)

    df["any_financial_hardship"] = (
        (df.get("email_ever_financial_hardship", z)==1) |
        (df.get("cc_ever_hardship", z)==1)
    ).astype(int)

    df["any_complaint"] = (
        (df.get("email_ever_complained", z)==1) |
        (df.get("cc_ever_complained", z)==1)
    ).astype(int)

    neg_flag_cols = [
        "email_ever_suggested_leave","email_ever_competitor","email_ever_price_dissatisfied",
        "email_ever_complained","email_ever_financial_hardship","email_ever_refund_mentioned",
        "email_ever_overdue","cc_ever_suggest_leave","cc_ever_hardship","cc_ever_complained",
        "cc_ever_platform_issues","cc_ever_pricing","rc_discount_requested",
        "rc_competitor_mentioned","rc_switching_intent","rc_rescheduled","rc_agent_flagged",
        "auto_renewal_off","payment_unknown","anchoring_zero","price_increase_10pct",
    ]
    existing = [c for c in neg_flag_cols if c in df.columns]
    df["total_negative_flags"] = df[existing].sum(axis=1)

    df["total_contact_count"] = (
        df.get("email_count", z) +
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
    emails: pd.DataFrame,
    cc: pd.DataFrame,
    rc: pd.DataFrame,
    renewal_year: int,
    include_labels: bool = True,
) -> pd.DataFrame:
    """
    Build complete feature table for any cohort year.

    renewal_year=2024 → training set (labels from 2025)
    renewal_year=2025 → test + predict set (labels from 2026, partial)
    include_labels=False → production scoring mode
    """
    base = build_billing_features(bills, renewal_year)
    co_ref_set = set(base["Co_Ref"])
    n = len(base)
    print(f"[{renewal_year}] Base: {n:,} customers")

    email_feats = build_email_features(emails, co_ref_set)
    cc_feats = build_cc_features(cc, co_ref_set, call_year=renewal_year)
    rc_feats = build_rc_features(rc, co_ref_set, call_year=renewal_year)
    print(f"[{renewal_year}] Coverage — emails:{len(email_feats):,} "
          f"cc:{len(cc_feats):,} rc:{len(rc_feats):,}")

    df = base.copy()
    df = df.merge(email_feats, on="Co_Ref", how="left")
    df = df.merge(cc_feats,    on="Co_Ref", how="left")
    df = df.merge(rc_feats,    on="Co_Ref", how="left")

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    df = add_composite_features(df)

    assert len(df) == n, f"Row count changed: {len(df)} != {n}"
    assert df["Co_Ref"].duplicated().sum() == 0, "Duplicate Co_Refs"
    print(f"[{renewal_year}] Final: {df.shape} ✓")

    if include_labels:
        labels = build_labels(bills, renewal_year)
        df = df.merge(labels, on="Co_Ref", how="left")
        print(f"[{renewal_year}] Churn rate: {df['churn_label'].mean()*100:.2f}%")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return model-ready numeric feature column names."""
    drop = set(DROP_BEFORE_TRAINING)
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
