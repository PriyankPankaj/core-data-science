"""
Phase 4: Feature engineering for churn prediction.

Every engineered feature is documented with its rationale below and in
the generated report. Original feature count vs. final count is tracked
explicitly per the spec.
"""
import pandas as pd
import numpy as np
import json

CLEAN_PATH = "data/telco_churn_clean.csv"
ENGINEERED_PATH = "data/telco_churn_engineered.csv"
RESULTS_DIR = "results"

FEATURE_RATIONALE = {
    "tenure_bucket": (
        "Groups raw tenure (0-72 months) into interpretable bands "
        "(New/Growing/Established/Loyal). Raw tenure showed a strong "
        "negative correlation with churn (-0.35 in Phase 3); bucketing "
        "helps capture non-linear risk thresholds a linear model alone "
        "might miss, and gives directly interpretable segments for "
        "business stakeholders."
    ),
    "avg_monthly_spend": (
        "TotalCharges / tenure (with tenure=0 handled as MonthlyCharges "
        "itself, since new customers have no historical average yet). "
        "Disentangles a customer's typical spend level from how long "
        "they've been a customer, since raw TotalCharges is confounded "
        "with tenure (Phase 2 finding)."
    ),
    "service_count": (
        "Count of add-on services subscribed (OnlineSecurity, "
        "OnlineBackup, DeviceProtection, TechSupport, StreamingTV, "
        "StreamingMovies, MultipleLines) that are 'Yes'. Phase 3 showed "
        "several individual services significantly associated with "
        "churn (Cramér's V 0.23-0.35); a combined count may capture "
        "overall customer engagement/lock-in more robustly than any "
        "single service."
    ),
    "has_internet_addons": (
        "Binary flag: True if the customer has any of OnlineSecurity, "
        "OnlineBackup, DeviceProtection, or TechSupport. These four "
        "services only apply to customers with InternetService != 'No', "
        "so this captures a meaningful engagement signal specifically "
        "within the internet-subscriber segment."
    ),
    "is_month_to_month": (
        "Binary flag for Contract == 'Month-to-month'. Contract type had "
        "the single strongest categorical association with churn in "
        "Phase 3 (Cramér's V = 0.41); isolating the highest-risk contract "
        "type as its own binary feature makes this the most direct "
        "possible signal for models that benefit from explicit flags "
        "over multi-category one-hot columns."
    ),
    "payment_delay_risk": (
        "Binary flag: True if PaymentMethod is 'Electronic check' "
        "(historically the highest-churn payment method in this dataset "
        "per common domain knowledge, confirmed by Phase 3's significant "
        "PaymentMethod association, Cramér's V = 0.30)."
    ),
    "charges_per_service": (
        "MonthlyCharges / (service_count + 1), avoiding division by "
        "zero for customers with 0 add-on services. Captures whether a "
        "customer is paying a premium relative to how many services "
        "they actually use — a proxy for perceived value, which isn't "
        "directly present in any single original column."
    ),
}


def engineer_features(df):
    original_cols = set(df.columns)

    # 1. Tenure bucket
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New (0-12mo)", "Growing (12-24mo)", "Established (24-48mo)", "Loyal (48-72mo)"],
    )

    # 2. Average monthly spend (handling tenure=0 case)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # 3. Service count
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = (df[service_cols] == "Yes").sum(axis=1)

    # 4. Has internet addons
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["has_internet_addons"] = (df[addon_cols] == "Yes").any(axis=1)

    # 5. Is month-to-month
    df["is_month_to_month"] = df["Contract"] == "Month-to-month"

    # 6. Payment delay risk
    df["payment_delay_risk"] = df["PaymentMethod"] == "Electronic check"

    # 7. Charges per service
    df["charges_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)

    engineered_cols = set(df.columns) - original_cols

    return df, len(original_cols), len(df.columns), engineered_cols


def write_report(original_count, final_count, engineered_cols):
    with open(f"{RESULTS_DIR}/../FEATURE_ENGINEERING_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Feature Engineering Report\n\n")
        f.write(f"- Original feature count: {original_count}\n")
        f.write(f"- Engineered feature count: {len(engineered_cols)}\n")
        f.write(f"- Final feature count: {final_count}\n\n")
        f.write("## Engineered Features & Rationale\n\n")
        for feat in sorted(engineered_cols):
            f.write(f"### `{feat}`\n\n{FEATURE_RATIONALE.get(feat, 'No rationale recorded.')}\n\n")

    print(f"Report written to FEATURE_ENGINEERING_REPORT.md")


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)

    df_engineered, original_count, final_count, engineered_cols = engineer_features(df)

    print(f"Original features: {original_count}")
    print(f"Engineered features: {len(engineered_cols)}")
    print(f"Final features: {final_count}")
    print(f"\nNew features added: {sorted(engineered_cols)}")

    print("\n=== Sanity check: engineered feature stats ===")
    print(f"\ntenure_bucket distribution:\n{df_engineered['tenure_bucket'].value_counts()}")
    print(f"\navg_monthly_spend: mean={df_engineered['avg_monthly_spend'].mean():.2f}, "
          f"min={df_engineered['avg_monthly_spend'].min():.2f}, "
          f"max={df_engineered['avg_monthly_spend'].max():.2f}")
    print(f"\nservice_count distribution:\n{df_engineered['service_count'].value_counts().sort_index()}")
    print(f"\nis_month_to_month: {df_engineered['is_month_to_month'].sum()} True, "
          f"{(~df_engineered['is_month_to_month']).sum()} False")

    df_engineered.to_csv(ENGINEERED_PATH, index=False)
    write_report(original_count, final_count, engineered_cols)

    print(f"\nEngineered dataset saved to {ENGINEERED_PATH}")