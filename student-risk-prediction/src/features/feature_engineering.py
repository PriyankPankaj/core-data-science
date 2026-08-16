"""
Phase 5: Feature engineering for student risk prediction.

Every feature is documented with rationale grounded in Phase 3/4 findings.
G1/G2/G3 remain excluded throughout (leakage prevention).
"""
import pandas as pd
import numpy as np
import json

DATA_PATH = "data/student_performance_with_target.csv"
ENGINEERED_PATH = "data/student_performance_engineered.csv"
RESULTS_DIR = "results"

FEATURE_RATIONALE = {
    "parental_education_avg": (
        "Average of Medu and Fedu. Both individually showed significant, "
        "similarly-directed negative correlation with risk in Phase 3/4 "
        "(Medu: -0.14, Fedu: -0.15); combining captures overall parental "
        "educational background as a single signal rather than two "
        "correlated ones."
    ),
    "failure_risk_flag": (
        "Binary flag: True if failures > 0. failures was the strongest "
        "single predictor (rank-biserial=-0.40, p<0.0001); a binary flag "
        "isolates 'has failed before' from the count magnitude, which "
        "may be a cleaner signal for models sensitive to the resulting "
        "skewed distribution (skewness=3.09)."
    ),
    "total_alcohol_consumption": (
        "Dalc + Walc combined. Both individually significant (p<0.005 "
        "each) with similar direction; combining into a single weekly "
        "consumption proxy reduces redundant correlated features."
    ),
    "study_efficiency": (
        "studytime / (failures + 1), avoiding division by zero. Captures "
        "whether study time is translating into avoided failures — a "
        "student with high studytime AND high failures may indicate "
        "different circumstances than raw studytime alone suggests."
    ),
    "social_engagement": (
        "goout + freetime combined. Both showed weaker but present "
        "correlations with risk; combined as a general social/leisure "
        "engagement proxy."
    ),
    "wants_higher_no_support": (
        "Binary flag: True if higher=='yes' AND schoolsup=='no'. higher "
        "was the strongest categorical predictor (Cramér's V=0.30); this "
        "flag identifies students who are motivated (want higher "
        "education) but lack formal school support — a potentially "
        "actionable intervention target."
    ),
}


def engineer_features(df):
    original_cols = set(df.columns)

    df["parental_education_avg"] = (df["Medu"] + df["Fedu"]) / 2
    df["failure_risk_flag"] = (df["failures"] > 0).astype(int)
    df["total_alcohol_consumption"] = df["Dalc"] + df["Walc"]
    df["study_efficiency"] = df["studytime"] / (df["failures"] + 1)
    df["social_engagement"] = df["goout"] + df["freetime"]
    df["wants_higher_no_support"] = ((df["higher"] == "yes") & (df["schoolsup"] == "no")).astype(int)

    engineered_cols = set(df.columns) - original_cols
    return df, len(original_cols), len(df.columns), engineered_cols


def write_report(original_count, final_count, engineered_cols):
    with open("FEATURE_ENGINEERING_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Feature Engineering Report\n\n")
        f.write(f"- Original feature count: {original_count}\n")
        f.write(f"- Engineered feature count: {len(engineered_cols)}\n")
        f.write(f"- Final feature count: {final_count}\n\n")
        f.write("## Engineered Features & Rationale\n\n")
        for feat in sorted(engineered_cols):
            f.write(f"### `{feat}`\n\n{FEATURE_RATIONALE.get(feat, 'No rationale recorded.')}\n\n")
    print("Report written to FEATURE_ENGINEERING_REPORT.md")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df_eng, original_count, final_count, engineered_cols = engineer_features(df)

    print(f"Original features: {original_count}")
    print(f"Engineered features: {len(engineered_cols)}")
    print(f"Final features: {final_count}")
    print(f"\nNew features: {sorted(engineered_cols)}")

    print("\n=== Sanity checks ===")
    print(f"parental_education_avg: mean={df_eng['parental_education_avg'].mean():.2f}")
    print(f"failure_risk_flag: {df_eng['failure_risk_flag'].sum()} True, {(~df_eng['failure_risk_flag'].astype(bool)).sum()} False")
    print(f"total_alcohol_consumption: mean={df_eng['total_alcohol_consumption'].mean():.2f}, "
          f"min={df_eng['total_alcohol_consumption'].min()}, max={df_eng['total_alcohol_consumption'].max()}")
    print(f"study_efficiency: mean={df_eng['study_efficiency'].mean():.2f}")
    print(f"wants_higher_no_support: {df_eng['wants_higher_no_support'].sum()} students")

    df_eng.to_csv(ENGINEERED_PATH, index=False)
    write_report(original_count, final_count, engineered_cols)

    print(f"\nSaved to {ENGINEERED_PATH}")