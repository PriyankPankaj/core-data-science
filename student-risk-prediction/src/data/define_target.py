"""
Phase 1 (continued): Target construction.

At-risk threshold: G3 < 10 (out of 20) — the standard passing threshold
used in Portuguese secondary education, matching the original UCI dataset
documentation's own framing of this dataset (grades are on a 0-20 scale,
10 is the conventional pass/fail boundary). This is a real, externally
justified threshold, not an arbitrary one chosen to produce a nice split.

LEAKAGE PREVENTION: G1, G2, AND G3 are all excluded from the feature set,
not just G3. G1/G2 are the same construct (grade performance) at earlier
timepoints and would leak strong signal about G3.
"""
import pandas as pd
import json

RAW_PATH = "data/student_performance_raw.csv"
RESULTS_DIR = "results"

AT_RISK_THRESHOLD = 10  # G3 < 10 = at risk (standard pass/fail boundary, 0-20 scale)
LEAKAGE_COLUMNS = ["G1", "G2", "G3"]


def construct_target(df):
    df["at_risk"] = (df["G3"] < AT_RISK_THRESHOLD).astype(int)
    return df


def sensitivity_analysis(df):
    """Since the threshold is a real but still a choice, check how the
    class balance shifts at nearby thresholds — documents that the
    threshold wasn't cherry-picked to produce a convenient split."""
    results = {}
    for threshold in [8, 9, 10, 11, 12]:
        at_risk_count = (df["G3"] < threshold).sum()
        results[threshold] = {
            "at_risk_count": int(at_risk_count),
            "at_risk_pct": float(at_risk_count / len(df) * 100),
        }
    return results


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    df = construct_target(df)

    print(f"At-risk threshold: G3 < {AT_RISK_THRESHOLD}")
    print(f"\nClass distribution:")
    print(df["at_risk"].value_counts())
    print(f"\nClass balance: {df['at_risk'].mean()*100:.2f}% at-risk")

    print(f"\n=== Threshold sensitivity analysis ===")
    sensitivity = sensitivity_analysis(df)
    for threshold, stats in sensitivity.items():
        print(f"  G3 < {threshold}: {stats['at_risk_count']} students ({stats['at_risk_pct']:.2f}%)")

    print(f"\n=== Leakage prevention ===")
    print(f"Excluded from features (target-derived): {LEAKAGE_COLUMNS}")

    df_features = df.drop(columns=LEAKAGE_COLUMNS)
    print(f"Feature columns retained: {len(df_features.columns) - 1}")  # -1 for at_risk itself

    df.to_csv("data/student_performance_with_target.csv", index=False)

    with open(f"{RESULTS_DIR}/target_construction.json", "w") as f:
        json.dump({
            "threshold_used": AT_RISK_THRESHOLD,
            "threshold_justification": "Standard pass/fail boundary on 0-20 grading scale",
            "class_distribution": df["at_risk"].value_counts().to_dict(),
            "class_balance_pct_at_risk": float(df["at_risk"].mean() * 100),
            "sensitivity_analysis": sensitivity,
            "leakage_columns_excluded": LEAKAGE_COLUMNS,
        }, f, indent=2)

    print(f"\nSaved to data/student_performance_with_target.csv")
    print(f"Target construction documented in {RESULTS_DIR}/target_construction.json")