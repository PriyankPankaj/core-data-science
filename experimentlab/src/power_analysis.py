"""
Phase 5: Power analysis.

Calculates required sample size for detecting the observed effects at
80%, 90%, and 95% power — both prospectively (what would we have needed
to plan for) and retrospectively (did we actually have enough power for
what we found), given the real baseline and effect sizes from Phase 4.
"""
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import json

RESULTS_DIR = "results"
ALPHA = 0.05
POWER_LEVELS = [0.80, 0.90, 0.95]


def required_sample_size(baseline_rate, mde_absolute, alpha=ALPHA, power=0.80):
    """MDE = minimum detectable effect (absolute). Returns required sample
    size PER GROUP."""
    p1 = baseline_rate
    p2 = baseline_rate + mde_absolute

    effect_size = proportion_effectsize(p1, p2)
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=power, ratio=1.0)
    return int(np.ceil(n))


def achieved_power(baseline_rate, observed_effect, n_per_group, alpha=ALPHA):
    """Given the actual sample size used, what power did we actually have
    to detect the effect we observed?"""
    p1 = baseline_rate
    p2 = baseline_rate + observed_effect
    effect_size = proportion_effectsize(p1, p2)
    analysis = NormalIndPower()
    power = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha,
                                   nobs1=n_per_group, ratio=1.0)
    return power


if __name__ == "__main__":
    with open(f"{RESULTS_DIR}/hypothesis_tests.json") as f:
        hyp_results = json.load(f)

    results = {}

    for metric in ["retention_1", "retention_7"]:
        r = hyp_results[metric]
        baseline = r["rate_control"]
        observed_effect = r["absolute_lift"]
        actual_n = r["n_control"]  # roughly equal group sizes

        print(f"=== {metric} ===")
        print(f"Baseline (control) rate: {baseline*100:.2f}%")
        print(f"Observed effect: {observed_effect*100:.3f}pp")
        print(f"Actual sample size per group: {actual_n}")

        # Prospective: sample size needed to detect THIS effect size at each power level
        sample_sizes = {}
        for power in POWER_LEVELS:
            n_required = required_sample_size(baseline, abs(observed_effect), power=power)
            sample_sizes[f"n_required_{int(power*100)}pct_power"] = n_required
            print(f"  Sample size needed for {int(power*100)}% power: {n_required} per group")

        # Retrospective: given our actual sample size, what power did we have?
        power_achieved = achieved_power(baseline, observed_effect, actual_n)
        print(f"  Achieved power with actual sample size: {power_achieved*100:.1f}%")

        # Also: MDE we could reliably detect at 80% power with our actual N
        # (search for the smallest effect detectable at 80% power given actual N)
        analysis = NormalIndPower()
        detectable_effect_size = analysis.solve_power(
            effect_size=None, alpha=ALPHA, power=0.80, nobs1=actual_n, ratio=1.0
        )
        # Convert Cohen's h effect size back to an approximate absolute proportion difference
        # (approximation via derivative at baseline rate)
        mde_approx = detectable_effect_size * np.sqrt(baseline * (1 - baseline)) * 2

        results[metric] = {
            "baseline_rate": float(baseline),
            "observed_effect_absolute": float(observed_effect),
            "actual_n_per_group": int(actual_n),
            **sample_sizes,
            "achieved_power_pct": float(power_achieved * 100),
            "approx_mde_at_80pct_power": float(mde_approx),
        }
        print(f"  Approx. minimum detectable effect at 80% power with actual N: {mde_approx*100:.3f}pp\n")

    with open(f"{RESULTS_DIR}/power_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}/power_analysis.json")