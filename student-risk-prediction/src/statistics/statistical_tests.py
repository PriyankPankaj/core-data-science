"""
Phase 4: Statistical hypothesis testing for at-risk associations.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json

DATA_PATH = "data/student_performance_with_target.csv"
RESULTS_DIR = "results"
ALPHA = 0.05

NUMERICAL_COLS = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                   "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
CATEGORICAL_COLS = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                     "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
                     "nursery", "higher", "internet", "romantic"]


def check_normality(series, sample_size=500):
    sample = series.sample(min(len(series), sample_size), random_state=42)
    stat, p = stats.shapiro(sample)
    return bool(p > ALPHA)


def test_numerical_feature(df, col, target_col="at_risk"):
    group_risk = df.loc[df[target_col] == 1, col].dropna()
    group_safe = df.loc[df[target_col] == 0, col].dropna()

    is_normal = check_normality(df[col].dropna())

    if is_normal:
        test_name = "Independent t-test"
        stat, p_value = stats.ttest_ind(group_risk, group_safe, equal_var=False)
        pooled_std = np.sqrt((group_risk.std()**2 + group_safe.std()**2) / 2)
        effect_size = (group_risk.mean() - group_safe.mean()) / pooled_std
        effect_name = "Cohen's d"
    else:
        test_name = "Mann-Whitney U"
        stat, p_value = stats.mannwhitneyu(group_risk, group_safe, alternative="two-sided")
        n1, n2 = len(group_risk), len(group_safe)
        effect_size = 1 - (2 * stat) / (n1 * n2)
        effect_name = "Rank-biserial correlation"

    return {
        "feature": col, "test": test_name, "normality_assumed": is_normal,
        "statistic": float(stat), "p_value": float(p_value),
        "mean_at_risk": float(group_risk.mean()), "mean_not_at_risk": float(group_safe.mean()),
        "effect_size_name": effect_name, "effect_size": float(effect_size),
    }


def test_categorical_feature(df, col, target_col="at_risk"):
    contingency = pd.crosstab(df[col], df[target_col])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    if contingency.shape == (2, 2) and (expected < 5).any():
        test_name = "Fisher's exact test"
        odds_ratio, p_value = stats.fisher_exact(contingency)
        statistic = odds_ratio
        effect_name = "Odds ratio"
        effect_size = odds_ratio
    else:
        test_name = "Chi-square test of independence"
        statistic = chi2
        p_value = p_chi2
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        effect_size = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
        effect_name = "Cramér's V"

    return {
        "feature": col, "test": test_name, "statistic": float(statistic),
        "p_value": float(p_value), "effect_size_name": effect_name,
        "effect_size": float(effect_size),
    }


def apply_correction(results):
    p_values = [r["p_value"] for r in results]
    _, bonferroni_p, _, _ = multipletests(p_values, alpha=ALPHA, method="bonferroni")
    _, bh_p, _, _ = multipletests(p_values, alpha=ALPHA, method="fdr_bh")

    for i, r in enumerate(results):
        r["p_value_bonferroni"] = float(bonferroni_p[i])
        r["p_value_bh_fdr"] = float(bh_p[i])
        r["significant_raw"] = bool(r["p_value"] < ALPHA)
        r["significant_bonferroni"] = bool(bonferroni_p[i] < ALPHA)
        r["significant_bh_fdr"] = bool(bh_p[i] < ALPHA)
    return results


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("=== Testing Numerical Features ===")
    numerical_results = []
    for col in NUMERICAL_COLS:
        result = test_numerical_feature(df, col)
        numerical_results.append(result)
        print(f"{col}: {result['test']}, p={result['p_value']:.6f}, "
              f"{result['effect_size_name']}={result['effect_size']:.4f}")

    print("\n=== Testing Categorical Features ===")
    categorical_results = []
    for col in CATEGORICAL_COLS:
        result = test_categorical_feature(df, col)
        categorical_results.append(result)
        print(f"{col}: {result['test']}, p={result['p_value']:.6f}, "
              f"{result['effect_size_name']}={result['effect_size']:.4f}")

    all_results = apply_correction(numerical_results + categorical_results)

    sig_raw = sum(r["significant_raw"] for r in all_results)
    sig_bonf = sum(r["significant_bonferroni"] for r in all_results)
    sig_bh = sum(r["significant_bh_fdr"] for r in all_results)
    print(f"\n=== Significance Summary ===")
    print(f"Total tests: {len(all_results)}")
    print(f"Significant at raw alpha=0.05: {sig_raw}")
    print(f"Significant after Bonferroni: {sig_bonf}")
    print(f"Significant after BH-FDR: {sig_bh}")

    with open(f"{RESULTS_DIR}/statistical_tests.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/statistical_tests.json")