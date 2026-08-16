"""
Phase 9: Multiple testing correction.

We ran 3 hypothesis tests (retention_1, retention_7, sum_gamerounds)
simultaneously — without correction, the chance of at least one false
positive purely by chance rises above the nominal 5% alpha.
"""
import json
from statsmodels.stats.multitest import multipletests

RESULTS_DIR = "results"
ALPHA = 0.05

if __name__ == "__main__":
    with open(f"{RESULTS_DIR}/hypothesis_tests.json") as f:
        hyp = json.load(f)
    with open(f"{RESULTS_DIR}/continuous_metrics.json") as f:
        cont = json.load(f)

    tests = [
        ("retention_1", hyp["retention_1"]["p_value"]),
        ("retention_7", hyp["retention_7"]["p_value"]),
        ("sum_gamerounds", cont["p_value"]),
    ]

    names = [t[0] for t in tests]
    p_values = [t[1] for t in tests]

    _, bonferroni_p, _, _ = multipletests(p_values, alpha=ALPHA, method="bonferroni")
    _, bh_p, _, _ = multipletests(p_values, alpha=ALPHA, method="fdr_bh")

    print(f"{'Metric':<20} {'Raw p':<12} {'Bonferroni p':<15} {'BH-FDR p':<12} {'Sig (raw)':<12} {'Sig (Bonf)':<12} {'Sig (BH)':<10}")
    print("-" * 95)

    results = []
    for i, name in enumerate(names):
        sig_raw = p_values[i] < ALPHA
        sig_bonf = bonferroni_p[i] < ALPHA
        sig_bh = bh_p[i] < ALPHA
        print(f"{name:<20} {p_values[i]:<12.6f} {bonferroni_p[i]:<15.6f} {bh_p[i]:<12.6f} "
              f"{str(sig_raw):<12} {str(sig_bonf):<12} {str(sig_bh):<10}")
        results.append({
            "metric": name, "p_raw": p_values[i], "p_bonferroni": float(bonferroni_p[i]),
            "p_bh_fdr": float(bh_p[i]), "significant_raw": bool(sig_raw),
            "significant_bonferroni": bool(sig_bonf), "significant_bh_fdr": bool(sig_bh),
        })

    print(f"\nWhy correction matters: testing 3 metrics at raw alpha=0.05 gives "
          f"roughly a {(1 - (1-ALPHA)**3)*100:.1f}% chance of at least one false "
          f"positive purely by chance, even if nothing real is happening.")

    with open(f"{RESULTS_DIR}/multiple_testing.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/multiple_testing.json")