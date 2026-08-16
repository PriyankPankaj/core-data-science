"""
Phase 12a: Simulation engine.

Uses REAL baseline rates and effect sizes measured from Cookie Cats
(Phases 4-5) as the ground-truth parameters for repeated simulated
experiments. This validates whether our statistical methodology
correctly recovers known effects, and measures actual false-positive
rate, power, and CI coverage — not fabricated, but computed from
1,000 simulated experiment repetitions.
"""
import numpy as np
from scipy import stats
import json

RESULTS_DIR = "results"
N_SIMULATIONS = 1000
ALPHA = 0.05
RANDOM_SEED = 42


def simulate_experiment(baseline_rate, true_effect, n_per_group, rng):
    """Simulates one experiment run given true parameters."""
    control_conversions = rng.binomial(n_per_group, baseline_rate)
    treatment_conversions = rng.binomial(n_per_group, baseline_rate + true_effect)

    p_control = control_conversions / n_per_group
    p_treatment = treatment_conversions / n_per_group

    count = np.array([treatment_conversions, control_conversions])
    nobs = np.array([n_per_group, n_per_group])

    from statsmodels.stats.proportion import proportions_ztest
    z_stat, p_value = proportions_ztest(count, nobs)

    diff = p_treatment - p_control
    se = np.sqrt(p_control*(1-p_control)/n_per_group + p_treatment*(1-p_treatment)/n_per_group)
    ci_low, ci_high = diff - 1.96*se, diff + 1.96*se

    return {
        "p_value": p_value,
        "significant": p_value < ALPHA,
        "effect_estimate": diff,
        "ci_covers_true_effect": ci_low <= true_effect <= ci_high,
    }


def run_simulation_suite(baseline_rate, true_effect, n_per_group, n_sims=N_SIMULATIONS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    results = [simulate_experiment(baseline_rate, true_effect, n_per_group, rng) for _ in range(n_sims)]

    power = np.mean([r["significant"] for r in results])
    effect_estimates = [r["effect_estimate"] for r in results]
    bias = np.mean(effect_estimates) - true_effect
    ci_coverage = np.mean([r["ci_covers_true_effect"] for r in results])

    return {
        "n_simulations": n_sims,
        "baseline_rate": float(baseline_rate),
        "true_effect": float(true_effect),
        "n_per_group": int(n_per_group),
        "power_achieved": float(power),
        "effect_estimate_mean": float(np.mean(effect_estimates)),
        "effect_estimate_bias": float(bias),
        "ci_coverage_rate": float(ci_coverage),
    }


if __name__ == "__main__":
    with open(f"{RESULTS_DIR}/hypothesis_tests.json") as f:
        hyp = json.load(f)

    # Scenario 1: simulate using the REAL retention_7 baseline and effect
    r7 = hyp["retention_7"]
    print("=== Simulation: retention_7 (real baseline + real observed effect) ===")
    sim1 = run_simulation_suite(r7["rate_control"], r7["absolute_lift"], r7["n_control"])
    print(f"Achieved power: {sim1['power_achieved']*100:.1f}%")
    print(f"Effect estimate bias: {sim1['effect_estimate_bias']:.5f} (should be ~0 if unbiased)")
    print(f"95% CI coverage: {sim1['ci_coverage_rate']*100:.1f}% (should be ~95% if well-calibrated)")

    # Scenario 2: false-positive rate check — simulate with TRUE effect = 0
    print("\n=== Simulation: False-positive rate check (true effect = 0) ===")
    sim2 = run_simulation_suite(r7["rate_control"], 0.0, r7["n_control"])
    print(f"False-positive rate: {sim2['power_achieved']*100:.1f}% (should be ~5% at alpha=0.05)")

    with open(f"{RESULTS_DIR}/simulation_results.json", "w") as f:
        json.dump({"real_effect_scenario": sim1, "null_effect_scenario": sim2}, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/simulation_results.json")