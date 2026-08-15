"""
Phase 3: Statistical hypothesis testing for churn associations.

Numerical features: independent t-test (if normality holds) or
Mann-Whitney U (if not), tested via Shapiro-Wilk on a sample.
Categorical features: Chi-square test of independence.
Multiple testing: Bonferroni and Benjamini-Hochberg FDR correction applied
across all tests, since we're running many hypothesis tests simultaneously.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json

CLEAN_PATH = "data/telco_churn_clean.csv"
RESULTS_DIR = "results"
ALPHA = 0.05


def check_normality(series, sample_size=500):
    """Shapiro-Wilk on a sample (test is unreliable/slow on very large N)."""
    sample = series.sample(min(len(series), sample_size), random_state=42)
    stat, p = stats.shapiro(sample)
    return bool(p > ALPHA)  # True = looks normal


def test_numerical_feature(df, col, target_col="Churn"):
    group_yes = df.loc[df[target_col] == "Yes", col].dropna()
    group_no = df.loc[df[target_col] == "No", col].dropna()

    is_normal = check_normality(df[col].dropna())

    if is_normal:
        test_name = "Independent t-test"
        stat, p_value = stats.ttest_ind(group_yes, group_no, equal_var=False)
        # Cohen's d effect size
        pooled_std = np.sqrt((group_yes.std()**2 + group_no.std()**2) / 2)
        effect_size = (group_yes.mean() - group_no.mean()) / pooled_std
        effect_name = "Cohen's d"
    else:
        test_name = "Mann-Whitney U"
        stat, p_value = stats.mannwhitneyu(group_yes, group_no, alternative="two-sided")
        # Rank-biserial correlation as effect size for Mann-Whitney
        n1, n2 = len(group_yes), len(group_no)
        effect_size = 1 - (2 * stat) / (n1 * n2)
        effect_name = "Rank-biserial correlation"

    # 95% CI for the mean difference (via bootstrap, since assumptions vary)
    diff = group_yes.mean() - group_no.mean()
    se = np.sqrt(group_yes.var() / len(group_yes) + group_no.var() / len(group_no))
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se

    return {
        "feature": col,
        "test": test_name,
        "normality_assumed": is_normal,
        "statistic": float(stat),
        "p_value": float(p_value),
        "mean_churn_yes": float(group_yes.mean()),
        "mean_churn_no": float(group_no.mean()),
        "mean_difference": float(diff),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "effect_size_name": effect_name,
        "effect_size": float(effect_size),
    }


def test_categorical_feature(df, col, target_col="Churn"):
    contingency = pd.crosstab(df[col], df[target_col])

    # Use Fisher's exact if any expected cell count < 5 and it's a 2x2 table;
    # otherwise Chi-square
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
        # Cramér's V effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        effect_size = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
        effect_name = "Cramér's V"

    return {
        "feature": col,
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_name": effect_name,
        "effect_size": float(effect_size),
        "degrees_of_freedom": int(dof) if test_name.startswith("Chi") else None,
    }


def apply_multiple_testing_correction(results):
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
    df = pd.read_csv(CLEAN_PATH)

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]

    print("=== Testing Numerical Features ===")
    numerical_results = []
    for col in numerical_cols:
        result = test_numerical_feature(df, col)
        numerical_results.append(result)
        print(f"\n{col}: {result['test']}")
        print(f"  statistic={result['statistic']:.4f}, p={result['p_value']:.6f}")
        print(f"  mean(Churn=Yes)={result['mean_churn_yes']:.2f}, "
              f"mean(Churn=No)={result['mean_churn_no']:.2f}")
        print(f"  {result['effect_size_name']}={result['effect_size']:.4f}")

    print("\n\n=== Testing Categorical Features ===")
    categorical_results = []
    for col in categorical_cols:
        result = test_categorical_feature(df, col)
        categorical_results.append(result)
        print(f"\n{col}: {result['test']}")
        print(f"  statistic={result['statistic']:.4f}, p={result['p_value']:.6f}")
        print(f"  {result['effect_size_name']}={result['effect_size']:.4f}")

    all_results = numerical_results + categorical_results
    all_results = apply_multiple_testing_correction(all_results)

    print("\n\n=== Significance Summary (after multiple testing correction) ===")
    sig_raw = sum(r["significant_raw"] for r in all_results)
    sig_bonf = sum(r["significant_bonferroni"] for r in all_results)
    sig_bh = sum(r["significant_bh_fdr"] for r in all_results)
    print(f"Total tests run: {len(all_results)}")
    print(f"Significant at raw alpha=0.05: {sig_raw}")
    print(f"Significant after Bonferroni correction: {sig_bonf}")
    print(f"Significant after Benjamini-Hochberg FDR: {sig_bh}")

    with open(f"{RESULTS_DIR}/statistical_tests.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFull results saved to {RESULTS_DIR}/statistical_tests.json")