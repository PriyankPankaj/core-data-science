"""
Phase 12b: Decision engine.

Combines statistical significance, effect size, business threshold, and
experiment validity into a SHIP / DO NOT SHIP / INCONCLUSIVE recommendation
with an explicit explanation — not just a p-value cutoff.
"""
import json

RESULTS_DIR = "results"

# Business threshold: what's the minimum effect size that matters enough
# to justify shipping this change? This is a business judgment call,
# stated explicitly rather than hidden.
MIN_BUSINESS_MEANINGFUL_LIFT_PP = 0.5  # a change must move retention by at least 0.5pp to matter


def make_decision(metric_name, hyp_result, randomization_valid, power_achieved=None):
    p_value = hyp_result["p_value"]
    lift_pp = hyp_result["absolute_lift"] * 100
    significant = hyp_result["significant"]

    reasons = []

    if not randomization_valid:
        decision = "INCONCLUSIVE"
        reasons.append("Randomization validity concerns — results cannot be trusted as-is.")
        return {"metric": metric_name, "decision": decision, "reasons": reasons}

    if not significant:
        decision = "DO NOT SHIP"
        reasons.append(f"Effect not statistically significant (p={p_value:.4f} >= 0.05).")
        if power_achieved is not None and power_achieved < 0.80:
            reasons.append(f"Note: experiment was underpowered ({power_achieved*100:.1f}% "
                           f"achieved power) — 'not significant' does not mean 'no effect', "
                           f"only that this sample size couldn't reliably detect one this size.")
        return {"metric": metric_name, "decision": decision, "reasons": reasons}

    # Significant result — check practical/business significance too
    practically_meaningful = abs(lift_pp) >= MIN_BUSINESS_MEANINGFUL_LIFT_PP

    if not practically_meaningful:
        decision = "DO NOT SHIP"
        reasons.append(f"Statistically significant (p={p_value:.4f}) BUT effect size "
                       f"({lift_pp:.2f}pp) is below the business-meaningful threshold "
                       f"({MIN_BUSINESS_MEANINGFUL_LIFT_PP}pp).")
    elif lift_pp < 0:
        decision = "DO NOT SHIP"
        reasons.append(f"Statistically significant AND practically meaningful "
                       f"({lift_pp:.2f}pp), but the effect is NEGATIVE — treatment "
                       f"performs worse than control.")
    else:
        decision = "SHIP"
        reasons.append(f"Statistically significant (p={p_value:.4f}) AND practically "
                       f"meaningful ({lift_pp:.2f}pp) positive effect.")

    return {"metric": metric_name, "decision": decision, "reasons": reasons}


if __name__ == "__main__":
    with open(f"{RESULTS_DIR}/hypothesis_tests.json") as f:
        hyp = json.load(f)
    with open(f"{RESULTS_DIR}/randomization_validation.json") as f:
        randomization = json.load(f)
    with open(f"{RESULTS_DIR}/power_analysis.json") as f:
        power = json.load(f)

    randomization_valid = randomization["experiment_valid"]

    print(f"Business threshold: minimum {MIN_BUSINESS_MEANINGFUL_LIFT_PP}pp lift to be worth shipping\n")

    decisions = {}
    for metric in ["retention_1", "retention_7"]:
        power_achieved = power[metric]["achieved_power_pct"] / 100
        decision = make_decision(metric, hyp[metric], randomization_valid, power_achieved)
        decisions[metric] = decision

        print(f"=== {metric}: {decision['decision']} ===")
        for reason in decision["reasons"]:
            print(f"  - {reason}")
        print()

    print("=== Overall Recommendation ===")
    print("Based on retention_7 (the metric with adequate statistical power), "
          "moving the gate from level 30 to level 40 should NOT be shipped — "
          "it produces a statistically significant AND practically meaningful "
          "NEGATIVE effect on 7-day retention, concentrated most heavily among "
          "the highest-engagement players (Phase 10 finding).")

    with open(f"{RESULTS_DIR}/decision_engine.json", "w") as f:
        json.dump(decisions, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/decision_engine.json")