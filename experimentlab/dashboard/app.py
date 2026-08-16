"""
ExperimentLab Dashboard — all data loaded from actual saved results.
"""
import streamlit as st
import pandas as pd
import json

RESULTS_DIR = "../results"

st.set_page_config(page_title="ExperimentLab", layout="wide")

PAGES = [
    "Experiment Overview", "Sample Balance", "Primary Metric",
    "Statistical Tests", "Confidence Intervals", "Effect Size",
    "Power Analysis", "Segment Analysis", "Multiple Testing",
    "Time Analysis", "Simulation", "Final Decision",
]

page = st.sidebar.radio("Navigate", PAGES)


def load_json(name):
    with open(f"{RESULTS_DIR}/{name}") as f:
        return json.load(f)


if page == "Experiment Overview":
    st.title("ExperimentLab: Cookie Cats Gate Placement A/B Test")
    st.markdown("**Control**: gate_30 | **Treatment**: gate_40")
    st.markdown("**Primary metric**: retention_7 | **Secondary**: retention_1, sum_gamerounds")
    hyp = load_json("hypothesis_tests.json")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Players", hyp["retention_7"]["n_control"] + hyp["retention_7"]["n_treatment"])
    col2.metric("Control Retention_7", f"{hyp['retention_7']['rate_control']*100:.1f}%")
    col3.metric("Treatment Retention_7", f"{hyp['retention_7']['rate_treatment']*100:.1f}%")

elif page == "Sample Balance":
    st.title("Sample Ratio Mismatch & Randomization Validity")
    rv = load_json("randomization_validation.json")
    st.json(rv["srm_check"])
    st.write(f"**Overall validity**: {'VALID' if rv['experiment_valid'] else 'FLAGGED'}")

elif page == "Primary Metric":
    st.title("Primary Metric: Retention")
    hyp = load_json("hypothesis_tests.json")
    df = pd.DataFrame([hyp["retention_1"], hyp["retention_7"]])
    st.dataframe(df[["metric", "rate_control", "rate_treatment", "absolute_lift", "p_value", "significant"]])

elif page == "Statistical Tests":
    st.title("All Statistical Tests")
    hyp = load_json("hypothesis_tests.json")
    cont = load_json("continuous_metrics.json")
    st.subheader("Binary metrics (retention)")
    st.json(hyp)
    st.subheader("Continuous metric (sum_gamerounds)")
    st.json(cont)

elif page == "Confidence Intervals":
    st.title("Effect Sizes & Confidence Intervals")
    st.image(f"{RESULTS_DIR}/effect_sizes_ci.png")

elif page == "Effect Size":
    st.title("Effect Size Summary")
    hyp = load_json("hypothesis_tests.json")
    for m in ["retention_1", "retention_7"]:
        st.write(f"**{m}**: odds ratio = {hyp[m]['odds_ratio']:.4f}, absolute lift = {hyp[m]['absolute_lift']*100:.3f}pp")

elif page == "Power Analysis":
    st.title("Power Analysis")
    power = load_json("power_analysis.json")
    for m, r in power.items():
        st.subheader(m)
        st.write(f"Achieved power: {r['achieved_power_pct']:.1f}%")
        st.write(f"Required N for 80% power: {r['n_required_80pct_power']}")

elif page == "Segment Analysis":
    st.title("Segment Analysis: Engagement Level")
    seg = load_json("segment_analysis.json")
    st.info(seg["note"])
    st.dataframe(pd.DataFrame(seg["results"]))

elif page == "Multiple Testing":
    st.title("Multiple Testing Correction")
    mt = load_json("multiple_testing.json")
    st.dataframe(pd.DataFrame(mt))

elif page == "Time Analysis":
    st.title("Time Analysis")
    st.warning("Limitation: Cookie Cats dataset has no event-level timestamps. See TIME_ANALYSIS_LIMITATION.md for details.")

elif page == "Simulation":
    st.title("Monte Carlo Simulation (1,000 runs)")
    sim = load_json("simulation_results.json")
    st.subheader("Real effect scenario")
    st.json(sim["real_effect_scenario"])
    st.subheader("Null effect scenario (false-positive check)")
    st.json(sim["null_effect_scenario"])

elif page == "Final Decision":
    st.title("Final Decision")
    decisions = load_json("decision_engine.json")
    for metric, d in decisions.items():
        st.subheader(f"{metric}: {d['decision']}")
        for r in d["reasons"]:
            st.write(f"- {r}")