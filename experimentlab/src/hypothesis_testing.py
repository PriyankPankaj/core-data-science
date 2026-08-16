"""
Phase 4: Primary hypothesis testing.

H0: The gate placement does not change retention.
H1: The gate placement changes retention.

Defined BEFORE looking at segment-level results (per spec: hypotheses
are not modified after viewing results). alpha=0.05, configurable.

Two-proportion z-test for the primary binary metric (retention_1), with
absolute lift, relative lift, 95% CI, and effect size.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import json

RAW_PATH = "data/cookie_cats_raw.csv"
RESULTS_DIR = "results"
ALPHA = 0.05

CONTROL_GROUP = "gate_30"
TREATMENT_GROUP = "gate_40"


def two_proportion_test(df, metric_col, control_name=CONTROL_GROUP, treatment_name=TREATMENT_GROUP):
    control = df.loc[df["version"] == control_name, metric_col]
    treatment = df.loc[df["version"] == treatment_name, metric_col]

    n_control, n_treatment = len(control), len(treatment)
    successes_control = control.sum()
    successes_treatment = treatment.sum()

    rate_control = successes_control / n_control
    rate_treatment = successes_treatment / n_treatment

    # Two-proportion z-test
    count = np.array([successes_treatment, successes_control])
    nobs = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(count, nobs)

    # Absolute and relative lift (treatment vs control)
    absolute_lift = rate_treatment - rate_control
    relative_lift = (absolute_lift / rate_control) * 100 if rate_control != 0 else None

    # 95% CI for the difference in proportions
    se_diff = np.sqrt(
        rate_control * (1 - rate_control) / n_control +
        rate_treatment * (1 - rate_treatment) / n_treatment
    )
    ci_low = absolute_lift - 1.96 * se_diff
    ci_high = absolute_lift + 1.96 * se_diff

    # Odds ratio (a standard effect size for binary outcomes)
    odds_control = rate_control / (1 - rate_control)
    odds_treatment = rate_treatment / (1 - rate_treatment)
    odds_ratio = odds_treatment / odds_control

    return {
        "metric": metric_col,
        "control_group": control_name,
        "treatment_group": treatment_name,
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "rate_control": float(rate_control),
        "rate_treatment": float(rate_treatment),
        "absolute_lift": float(absolute_lift),
        "relative_lift_pct": float(relative_lift) if relative_lift is not None else None,
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "odds_ratio": float(odds_ratio),
    }


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)

    print(f"H0: Gate placement does not change retention.")
    print(f"H1: Gate placement changes retention.")
    print(f"alpha = {ALPHA}\n")

    results = {}

    for metric in ["retention_1", "retention_7"]:
        print(f"=== {metric} ===")
        result = two_proportion_test(df, metric)
        results[metric] = result

        print(f"Control ({CONTROL_GROUP}) rate: {result['rate_control']*100:.2f}% (n={result['n_control']})")
        print(f"Treatment ({TREATMENT_GROUP}) rate: {result['rate_treatment']*100:.2f}% (n={result['n_treatment']})")
        print(f"Absolute lift: {result['absolute_lift']*100:.3f} pp")
        print(f"Relative lift: {result['relative_lift_pct']:.2f}%")
        print(f"Z-statistic: {result['z_statistic']:.4f}")
        print(f"P-value: {result['p_value']:.6f}")
        print(f"Significant at alpha={ALPHA}: {result['significant']}")
        print(f"95% CI for absolute lift: [{result['ci_95_low']*100:.3f}pp, {result['ci_95_high']*100:.3f}pp]")
        print(f"Odds ratio: {result['odds_ratio']:.4f}")
        print()

    print("=== Interpretation ===")
    r1 = results["retention_1"]
    if r1["significant"] and r1["absolute_lift"] < 0:
        print(f"retention_1 is significantly LOWER in treatment ({TREATMENT_GROUP}) "
              f"than control ({CONTROL_GROUP}) — moving the gate to level 40 "
              f"appears to HURT day-1 retention, the opposite of the likely "
              f"intended effect.")
    elif r1["significant"] and r1["absolute_lift"] > 0:
        print(f"retention_1 is significantly HIGHER in treatment.")
    else:
        print(f"No statistically significant difference in retention_1.")

    with open(f"{RESULTS_DIR}/hypothesis_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/hypothesis_tests.json")