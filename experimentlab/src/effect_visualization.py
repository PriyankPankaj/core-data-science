"""
Phases 7-8: Consolidates effect sizes and confidence intervals from
Phases 4 and 6 into a single comparison visualization.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"

if __name__ == "__main__":
    with open(f"{RESULTS_DIR}/hypothesis_tests.json") as f:
        hyp = json.load(f)
    with open(f"{RESULTS_DIR}/continuous_metrics.json") as f:
        cont = json.load(f)

    metrics = ["retention_1", "retention_7", "sum_gamerounds"]
    lifts = [
        hyp["retention_1"]["absolute_lift"] * 100,
        hyp["retention_7"]["absolute_lift"] * 100,
        cont["mean_difference"],
    ]
    ci_lows = [
        hyp["retention_1"]["ci_95_low"] * 100,
        hyp["retention_7"]["ci_95_low"] * 100,
        cont["ci_95_low"],
    ]
    ci_highs = [
        hyp["retention_1"]["ci_95_high"] * 100,
        hyp["retention_7"]["ci_95_high"] * 100,
        cont["ci_95_high"],
    ]
    significant = [
        hyp["retention_1"]["significant"],
        hyp["retention_7"]["significant"],
        cont["significant"],
    ]

    errors = [[lifts[i] - ci_lows[i] for i in range(3)], [ci_highs[i] - lifts[i] for i in range(3)]]
    colors = ["#C44E52" if not s else "#55A868" for s in significant]

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(metrics))
    plt.errorbar(lifts, y_pos, xerr=errors, fmt="o", color="black", capsize=5, markersize=0)
    for i, (lift, color) in enumerate(zip(lifts, colors)):
        plt.scatter(lift, i, color=color, s=150, zorder=3)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.yticks(y_pos, ["retention_1 (pp)", "retention_7 (pp)", "sum_gamerounds (raw diff)"])
    plt.xlabel("Treatment Effect (95% CI)")
    plt.title("Treatment Effect and 95% CI by Metric\n(green=significant, red=not significant)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/effect_sizes_ci.png", dpi=150)
    plt.close()

    print("Saved effect_sizes_ci.png")
    print("\n=== Summary ===")
    for m, lift, sig in zip(metrics, lifts, significant):
        print(f"{m}: effect={lift:.3f}, significant={sig}")