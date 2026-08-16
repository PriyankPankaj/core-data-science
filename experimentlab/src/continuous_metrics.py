"""
Phase 6: Continuous metric testing.

sum_gamerounds is the available continuous/count metric in this dataset
(standing in for the spec's "session duration"/"order value" style
metrics). Checks normality assumptions before selecting Welch's t-test
vs Mann-Whitney U, computes effect size (Cohen's d or rank-biserial),
and reports mean, median, variance, CI, and p-value.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

RAW_PATH = "data/cookie_cats_raw.csv"
RESULTS_DIR = "results"
ALPHA = 0.05

CONTROL_GROUP = "gate_30"
TREATMENT_GROUP = "gate_40"


def check_normality(series, sample_size=500):
    sample = series.sample(min(len(series), sample_size), random_state=42)
    stat, p = stats.shapiro(sample)
    return bool(p > ALPHA)


def test_continuous_metric(df, metric_col, control_name=CONTROL_GROUP, treatment_name=TREATMENT_GROUP):
    control = df.loc[df["version"] == control_name, metric_col]
    treatment = df.loc[df["version"] == treatment_name, metric_col]

    is_normal_control = check_normality(control)
    is_normal_treatment = check_normality(treatment)
    both_normal = is_normal_control and is_normal_treatment

    if both_normal:
        test_name = "Welch's t-test"
        stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        pooled_std = np.sqrt((treatment.std()**2 + control.std()**2) / 2)
        effect_size = (treatment.mean() - control.mean()) / pooled_std
        effect_name = "Cohen's d"
    else:
        test_name = "Mann-Whitney U"
        stat, p_value = stats.mannwhitneyu(treatment, control, alternative="two-sided")
        n1, n2 = len(treatment), len(control)
        effect_size = 1 - (2 * stat) / (n1 * n2)
        effect_name = "Rank-biserial correlation"

    mean_diff = treatment.mean() - control.mean()
    se = np.sqrt(treatment.var() / len(treatment) + control.var() / len(control))
    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se

    return {
        "metric": metric_col,
        "test": test_name,
        "assumptions_normal": both_normal,
        "control_mean": float(control.mean()),
        "control_median": float(control.median()),
        "control_variance": float(control.var()),
        "treatment_mean": float(treatment.mean()),
        "treatment_median": float(treatment.median()),
        "treatment_variance": float(treatment.var()),
        "mean_difference": float(mean_diff),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
        "effect_size_name": effect_name,
        "effect_size": float(effect_size),
    }


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)

    print("=== sum_gamerounds (engagement) ===")

    # Note: this variable is heavily right-skewed (a few extreme outlier
    # players with thousands of rounds) — worth checking before testing
    print(f"Control skewness: {df.loc[df['version']==CONTROL_GROUP, 'sum_gamerounds'].skew():.2f}")
    print(f"Treatment skewness: {df.loc[df['version']==TREATMENT_GROUP, 'sum_gamerounds'].skew():.2f}")
    print(f"Max value: {df['sum_gamerounds'].max()} (investigate outliers, don't silently drop)\n")

    result = test_continuous_metric(df, "sum_gamerounds")

    print(f"Test used: {result['test']} (assumptions_normal={result['assumptions_normal']})")
    print(f"Control: mean={result['control_mean']:.2f}, median={result['control_median']:.2f}, "
          f"variance={result['control_variance']:.2f}")
    print(f"Treatment: mean={result['treatment_mean']:.2f}, median={result['treatment_median']:.2f}, "
          f"variance={result['treatment_variance']:.2f}")
    print(f"Mean difference: {result['mean_difference']:.3f}")
    print(f"95% CI: [{result['ci_95_low']:.3f}, {result['ci_95_high']:.3f}]")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"Significant: {result['significant']}")
    print(f"{result['effect_size_name']}: {result['effect_size']:.4f}")

    with open(f"{RESULTS_DIR}/continuous_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/continuous_metrics.json")