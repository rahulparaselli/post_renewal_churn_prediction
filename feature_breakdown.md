# Feature Breakdown — `builder.py`

Complete breakdown of every feature: which raw column it came from, what new features were derived, and how they were created.

---

## 1. Features from `billings.csv` — `build_billing_features()`

### Columns Directly Kept from Billing

| Column | Description |
|---|---|
| `Co_Ref` | Customer ID (join key) |
| `Renewal_Year` | Year of renewal |
| `Band` | Membership band (A–J, Group) |
| `Tenure_Years` | How many years the customer has been a member |
| `Current_Anchorings` | Number of anchoring services |
| `Connection_Qty` | Number of connections |
| `Discount_Amount` | Discount given |
| `Last_Total_Net_Paid` | What they paid last year (used to derive `is_first_year`) |
| `Current_Auto_Renewal_Flag` | Auto-renewal on or off |
| `Auto_Renewal_Score` | Internal auto-renewal score |
| `Renewal_Score_At_Release` | Score at time of release |
| `Sustainability_Score` | Customer sustainability metric |
| `Last_Band` | Their band last year (used to derive upgrades/downgrades) |
| `#_of_Connection` | Connection count |
| `Prospect_Renewal_Date` | Used to define the 4-week post-renewal window |
| `Closed_Date` | When the deal actually closed |

### New Features Created from Billing

| Feature | How It's Made |
|---|---|
| `is_first_year` | `1` if `Last_Total_Net_Paid` is NaN (no prior payment → new customer) |
| `auto_renewal_off` | `1` if `Current_Auto_Renewal_Flag` ≠ "YES" |
| `anchoring_zero` | `1` if `Current_Anchorings == 0` |
| `anchoring_1plus` | `1` if `Current_Anchorings >= 1` |
| `anchoring_3plus` | `1` if `Current_Anchorings >= 3` |
| `band_enc` | Numeric encoding of `Band` (Band A=1, Band B=2, … Band J=13, Group=0) |
| `band_downgraded` | `1` if current `band_enc` < last year's `band_enc` |
| `band_upgraded` | `1` if current `band_enc` > last year's `band_enc` |
| `low_release_score` | `1` if `Renewal_Score_At_Release` < 25th percentile |
| `tenure_0_1` | `1` if `Tenure_Years <= 1` |
| `tenure_2_3` | `1` if `Tenure_Years` is between 2 and 3 |
| `tenure_4_7` | `1` if `Tenure_Years` is between 4 and 7 |
| `tenure_8plus` | `1` if `Tenure_Years >= 8` |

### Excluded Billing Columns (Post-Outcome Leakage)

| Excluded Column | Reason |
|---|---|
| `Total_Net_Paid` | $0 for churned customers (they didn't pay) |
| `Payment_Method` | UNKNOWN for churned customers (no payment set up) |
| `Payment_Timeframe` | Null for churned customers (no payment made) |
| `Current_World_Pay_Token` | Payment token (post-outcome information) |
| `Total_Renewal_Score_New` | Updated after outcome decided (correlation = -0.66 with outcome) |
| `Status_Scores` | Post-outcome status score (correlation = -0.66) |
| `Tenure_Scores` | Post-outcome tenure score (correlation = -0.43) |

---

## 2. Features from `cc_calls.csv` — `build_cc_features()`

### Columns Used from CC Calls

| Source Column | Purpose |
|---|---|
| `Co_Ref` | Join key |
| `Call_Year` | Filter to matching renewal year |
| `cc_contractor_suggest_leave` | → `cc_ever_suggest_leave` |
| `cc_business_struggles_financial_hardship` | → `cc_ever_hardship` |
| `cc_contractor_complained` | → `cc_ever_complained` |
| `cc_platform_issues` | → `cc_ever_platform_issues` |
| `cc_login_issues` | → `cc_ever_login_issues` |
| `cc_pricing_mentioned` | → `cc_ever_pricing` |
| `cc_refund_discussed` | → `cc_ever_refund` |
| `cc_contractor_sentiment_clean` | → `cc_pct_dissatisfied` |
| `cc_contractor_sentiment_end_score` | → sentiment delta calculation |
| `cc_contractor_sentiment_start_score` | → sentiment delta calculation |
| `cc_contractor_sentiment_overall_score` | → `cc_min_overall_sentiment` |
| `Direction` | → `cc_inbound_count` |

### New Features Created from CC Calls (Aggregated per Customer)

| Feature | How It's Made |
|---|---|
| `cc_call_count` | Total number of support calls per customer |
| `cc_ever_suggest_leave` | `1` if ANY call has `cc_contractor_suggest_leave == "Yes"` |
| `cc_ever_hardship` | `1` if ANY call mentions financial hardship |
| `cc_ever_complained` | `1` if ANY call has a complaint |
| `cc_ever_platform_issues` | `1` if ANY call mentions platform issues |
| `cc_ever_login_issues` | `1` if ANY call mentions login issues |
| `cc_ever_pricing` | `1` if ANY call discusses pricing |
| `cc_ever_refund` | `1` if ANY call discusses refund |
| `cc_pct_dissatisfied` | Fraction of calls where sentiment = "Dissatisfied" |
| `cc_avg_sentiment_delta` | Mean of (end_score − start_score) across all calls |
| `cc_min_overall_sentiment` | Minimum overall sentiment score (worst interaction) |
| `cc_inbound_count` | Count of inbound calls |
| `cc_inbound_ratio` | `cc_inbound_count / cc_call_count` |
| `cc_sentiment_worsened` | `1` if `cc_avg_sentiment_delta < 0` |
| `cc_multiple_calls` | `1` if `cc_call_count >= 3` |

---

## 3. Features from `renewal_calls.csv` — `build_rc_features()`

### Columns Used from Renewal Calls

| Source Column | Purpose |
|---|---|
| `Co_Ref` | Join key |
| `Call_Year` | Filter to matching renewal year |
| `Call_Number` | → `rc_max_call_number` |
| `Discount_or_Waiver_Requested` | → `rc_discount_requested` |
| `Discussion_on_Price_Increase` | → `rc_price_discussed` |
| `Explicit_Competitor_Mention` | → `rc_competitor_mentioned` |
| `Explicit_Switching_Intent` | → `rc_switching_intent` |
| `Call_Reschedule_Request` | → `rc_rescheduled` |
| `Agent_Flagged_Membership_Status_Alert` | → `rc_agent_flagged` |
| `Price_Switching_Mentioned` | → `rc_price_switching` |
| `Customer_Asked_For_Justification` | → `rc_asked_justification` |
| `Discount_Offered` | → `rc_discount_offered` |
| `Call_Direction` | → `rc_outbound_count` |

### New Features Created from Renewal Calls (Aggregated per Customer)

| Feature | How It's Made |
|---|---|
| `rc_call_count` | Total number of renewal calls per customer |
| `rc_max_call_number` | Maximum call number (how many call attempts made) |
| `rc_discount_requested` | `1` if customer ever requested a discount or waiver |
| `rc_price_discussed` | `1` if price increase was ever discussed |
| `rc_competitor_mentioned` | `1` if competitor was explicitly mentioned |
| `rc_switching_intent` | `1` if explicit switching intent was shown |
| `rc_rescheduled` | `1` if a call was resheduled |
| `rc_agent_flagged` | `1` if the agent flagged membership status alert |
| `rc_price_switching` | `1` if price switching was mentioned |
| `rc_asked_justification` | `1` if customer asked for price justification |
| `rc_discount_offered` | `1` if discount was offered to the customer |
| `rc_outbound_count` | Count of outbound calls |
| `rc_high_friction` | `1` if `rc_max_call_number >= 3` (needed many call attempts) |
| `rc_agent_chased` | `1` if `rc_outbound_count >= 2` (agent had to chase the customer) |

### Excluded Renewal Call Columns (Leakage)

| Excluded Column | Reason |
|---|---|
| `Membership_Renewal_Decision` | Directly reveals the outcome |
| `Churn_Category` | Directly reveals the outcome |
| `Desire_To_Cancel` | Directly reveals the outcome |
| `Customer_Renewal_Response_Category` | Directly reveals the outcome |

---

## 4. Features from `emails.csv` — ❌ NOT USED

Email data is **intentionally excluded** from the feature set.

**Reason**: `emails.year = Renewal_Year + 1`, meaning email data arrives AFTER the renewal decision has already been made. Using these would be **temporal leakage**.

---

## 5. Cross-File Composite Features — `add_composite_features()`

These features combine signals from **multiple data sources** after the join.

| Feature | How It's Made | Sources Combined |
|---|---|---|
| `any_leave_signal` | `1` if `cc_ever_suggest_leave == 1` OR `rc_switching_intent == 1` | CC Calls + Renewal Calls |
| `any_competitor_signal` | `1` if `rc_competitor_mentioned == 1` | Renewal Calls |
| `any_financial_hardship` | `1` if `cc_ever_hardship == 1` | CC Calls |
| `any_complaint` | `1` if `cc_ever_complained == 1` | CC Calls |
| `total_negative_flags` | Sum of 12 binary risk flags (listed below) | Billing + CC + Renewal |
| `total_contact_count` | `cc_call_count + rc_call_count` | CC Calls + Renewal Calls |
| `has_no_behavioural_data` | `1` if `total_contact_count == 0` (no calls at all) | CC Calls + Renewal Calls |
| `high_friction_score` | Sum of 4 friction indicators (see below) | CC Calls + Renewal Calls |
| `critical_risk_billing` | `1` if `auto_renewal_off == 1` AND `anchoring_zero == 1` | Billing |

### The 12 Flags Summed in `total_negative_flags`

| Flag | Source |
|---|---|
| `cc_ever_suggest_leave` | CC Calls |
| `cc_ever_hardship` | CC Calls |
| `cc_ever_complained` | CC Calls |
| `cc_ever_platform_issues` | CC Calls |
| `cc_ever_pricing` | CC Calls |
| `rc_discount_requested` | Renewal Calls |
| `rc_competitor_mentioned` | Renewal Calls |
| `rc_switching_intent` | Renewal Calls |
| `rc_rescheduled` | Renewal Calls |
| `rc_agent_flagged` | Renewal Calls |
| `auto_renewal_off` | Billing |
| `anchoring_zero` | Billing |

### The 4 Indicators Summed in `high_friction_score`

| Indicator | Source |
|---|---|
| `rc_max_call_number >= 3` | Renewal Calls |
| `rc_discount_requested == 1` | Renewal Calls |
| `rc_rescheduled == 1` | Renewal Calls |
| `cc_multiple_calls == 1` | CC Calls |

---

## 6. Label / Target Variable — `build_labels()`

| Label | Value | Source Column |
|---|---|---|
| `churn_label` | `1` if `Prospect_Outcome == "Churned"` | `billings.csv → Prospect_Outcome` |
| `churn_label` | `0` if `Prospect_Outcome == "Won"` | `billings.csv → Prospect_Outcome` |
| Excluded | — | `Prospect_Outcome == "Open"` (undecided, removed) |

**Cohort filter**: Only customers whose `Closed_Date` falls within 4 weeks (28 days) after `Prospect_Renewal_Date`.

---

## Summary Table

| Source | Raw Columns Used | New Features Created |
|---|---|---|
| **billings.csv** | 16 columns | **13** new features |
| **cc_calls.csv** | 14 columns | **15** new features |
| **renewal_calls.csv** | 12 columns | **14** new features |
| **emails.csv** | ❌ Excluded (temporal leakage) | **0** |
| **Cross-file composites** | — | **9** new features |
| **Label** | 1 column (`Prospect_Outcome`) | **1** (`churn_label`) |
| **Total** | **43 raw columns** | **~52 features + 1 label** |
