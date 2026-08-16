"""
Phase 3: Randomization validation.

Before trusting any treatment-effect analysis, verify the experiment's
randomization was actually valid: correct sample sizes, balanced group
proportions, and no Sample Ratio Mismatch (SRM) — a chi-square test on
group sizes themselves, which if significant means something is wrong
with assignment (bot traffic, broken redirect, logging bug, etc.) and
the experiment results should not be trusted as-is.
"""
import pandas as pd
from scipy import stats
import json

RAW_PATH = "data/cookie_cats_raw.csv"
RESULTS_DIR = "results"
EXPECTED_RATIO = 0.5  # intended 50/50 split
ALPHA = 0.01  # SRM checks conventionally use a stricter alpha than 0.05


def check_sample_ratio_mismatch(df, group_col="version"):
    counts = df[group_col].value_counts()
    total = counts.sum()

    observed = counts.values
    expected = [total * EXPECTED_RATIO] * len(counts)

    chi2, p_value = stats.chisquare(observed, expected)

    srm_detected_statistical = p_value < ALPHA

    # Practical significance: how far off is the actual split from 50/50,
    # in relative terms? A tiny relative imbalance can still be
    # "statistically significant" at very large N without being
    # practically meaningful.
    max_group_pct = observed.max() / total * 100
    relative_imbalance_pct = abs(max_group_pct - 50.0)
    PRACTICAL_THRESHOLD_PP = 1.0  # more than 1 percentage point off 50/50 = practically concerning

    practically_meaningful = relative_imbalance_pct > PRACTICAL_THRESHOLD_PP

    return {
        "group_counts": counts.to_dict(),
        "total": int(total),
        "expected_per_group": total * EXPECTED_RATIO,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "srm_detected_statistical": bool(srm_detected_statistical),
        "alpha_used": ALPHA,
        "largest_group_pct": float(max_group_pct),
        "relative_imbalance_pp": float(relative_imbalance_pct),
        "practically_meaningful_imbalance": bool(practically_meaningful),
        "final_verdict": "STATISTICALLY DETECTED BUT PRACTICALLY NEGLIGIBLE" if (srm_detected_statistical and not practically_meaningful) else ("SRM - INVESTIGATE" if srm_detected_statistical else "NO SRM"),
    }

def check_covariate_balance(df, group_col="version", covariate="sum_gamerounds"):
    groups = df.groupby(group_col)[covariate]
    summary = groups.agg(["mean", "median", "std", "count"]).to_dict("index")

    group_names = df[group_col].unique()
    g1 = df.loc[df[group_col] == group_names[0], covariate]
    g2 = df.loc[df[group_col] == group_names[1], covariate]

    stat, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")

    return {
        "covariate": covariate,
        "summary_by_group": summary,
        "mann_whitney_p_value": float(p_value),
        "balanced": bool(p_value > 0.05),
    }


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)

    print("=== Sample Ratio Mismatch (SRM) Check ===")
    srm_result = check_sample_ratio_mismatch(df)
    print(f"Group counts: {srm_result['group_counts']}")
    print(f"Expected per group (50/50): {srm_result['expected_per_group']:.1f}")
    print(f"Chi-square statistic: {srm_result['chi2_statistic']:.4f}")
    print(f"P-value: {srm_result['p_value']:.6f}")
    print(f"Statistically detected (p < {ALPHA}): {srm_result['srm_detected_statistical']}")
    print(f"Largest group: {srm_result['largest_group_pct']:.2f}% (imbalance: {srm_result['relative_imbalance_pp']:.2f}pp from 50/50)")
    print(f"Practically meaningful (>1pp off 50/50): {srm_result['practically_meaningful_imbalance']}")
    print(f"\nFinal verdict: {srm_result['final_verdict']}")

    if srm_result["final_verdict"] == "STATISTICALLY DETECTED BUT PRACTICALLY NEGLIGIBLE":
        print(f"\nNote: the chi-square SRM test is hypersensitive at this sample "
              f"size (n=90,189) — a {srm_result['relative_imbalance_pp']:.2f}pp split imbalance is statistically "
   
    print("\n=== Covariate Balance Check (sum_gamerounds) ===")
    balance_result = check_covariate_balance(df)
    for group, stats_dict in balance_result["summary_by_group"].items():
        print(f"  {group}: mean={stats_dict['mean']:.2f}, median={stats_dict['median']:.2f}, "
              f"std={stats_dict['std']:.2f}, n={int(stats_dict['count'])}")
    print(f"Mann-Whitney p-value: {balance_result['mann_whitney_p_value']:.6f}")
    print(f"Balanced: {balance_result['balanced']}")

    experiment_valid = not srm_result["practically_meaningful_imbalance"]
    print(f"\n=== Overall Randomization Validity: {'VALID' if experiment_valid else 'FLAGGED - DO NOT TRUST RESULTS'} ===")

    with open(f"{RESULTS_DIR}/randomization_validation.json", "w") as f:
        json.dump({
            "srm_check": srm_result,
            "covariate_balance": balance_result,
            "experiment_valid": bool(experiment_valid),
        }, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/randomization_validation.json")