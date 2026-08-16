"""
Phase 3: Exploratory Data Analysis for the Student Performance dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

DATA_PATH = "data/student_performance_with_target.csv"
RESULTS_DIR = "results"
LEAKAGE_COLUMNS = ["G1", "G2", "G3"]

sns.set_style("whitegrid")

NUMERICAL_COLS = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                   "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
CATEGORICAL_COLS = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
                     "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
                     "nursery", "higher", "internet", "romantic"]


def numerical_summary(df):
    summary = {}
    for col in NUMERICAL_COLS:
        summary[col] = {
            "mean": float(df[col].mean()), "median": float(df[col].median()),
            "std": float(df[col].std()), "q1": float(df[col].quantile(0.25)),
            "q3": float(df[col].quantile(0.75)), "skewness": float(df[col].skew()),
            "min": float(df[col].min()), "max": float(df[col].max()),
        }
    return summary


def categorical_summary(df):
    summary = {}
    for col in CATEGORICAL_COLS:
        freq = df[col].value_counts().to_dict()
        risk_rate = df.groupby(col)["at_risk"].mean() * 100
        summary[col] = {"frequency": freq, "risk_rate_pct": risk_rate.round(2).to_dict()}
    return summary


def plot_target_distribution(df, save_path):
    plt.figure(figsize=(6, 5))
    counts = df["at_risk"].value_counts()
    labels = ["Not At Risk", "At Risk"]
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#4C72B0", "#C44E52"])
    plt.title("At-Risk Distribution")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_numerical_distributions(df, save_path):
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    axes = axes.flatten()
    for i, col in enumerate(NUMERICAL_COLS):
        sns.histplot(data=df, x=col, hue="at_risk", kde=True, ax=axes[i], element="step")
        axes[i].set_title(col)
    for j in range(len(NUMERICAL_COLS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_boxplots(df, save_path):
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    axes = axes.flatten()
    for i, col in enumerate(NUMERICAL_COLS):
        sns.boxplot(data=df, x="at_risk", y=col, ax=axes[i])
        axes[i].set_title(col)
    for j in range(len(NUMERICAL_COLS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_categorical_risk_rates(df, save_path):
    n_cols = 4
    n_rows = int(np.ceil(len(CATEGORICAL_COLS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(CATEGORICAL_COLS):
        risk_rate = df.groupby(col)["at_risk"].mean() * 100
        risk_rate = risk_rate.sort_values(ascending=False)
        axes[i].bar(risk_rate.index.astype(str), risk_rate.values, color="#DD8452")
        axes[i].set_title(f"Risk Rate by {col}")
        axes[i].tick_params(axis="x", rotation=45)
    for j in range(len(CATEGORICAL_COLS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_correlation_heatmap(df, save_path):
    corr_cols = NUMERICAL_COLS + ["at_risk"]
    corr_matrix = df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return corr_matrix["at_risk"].drop("at_risk").to_dict()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("=== Numerical Summary ===")
    num_summary = numerical_summary(df)
    for col, stats in num_summary.items():
        print(f"{col}: mean={stats['mean']:.2f}, skew={stats['skewness']:.2f}")

    print("\n=== Generating Visualizations ===")
    plot_target_distribution(df, f"{RESULTS_DIR}/target_distribution.png")
    print("  Saved target_distribution.png")
    plot_numerical_distributions(df, f"{RESULTS_DIR}/numerical_distributions.png")
    print("  Saved numerical_distributions.png")
    plot_boxplots(df, f"{RESULTS_DIR}/boxplots.png")
    print("  Saved boxplots.png")
    plot_categorical_risk_rates(df, f"{RESULTS_DIR}/categorical_risk_rates.png")
    print("  Saved categorical_risk_rates.png")
    correlations = plot_correlation_heatmap(df, f"{RESULTS_DIR}/correlation_heatmap.png")
    print("  Saved correlation_heatmap.png")

    print("\n=== Correlation with at_risk ===")
    for feat, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat}: {corr:.4f}")

    cat_summary = categorical_summary(df)

    with open(f"{RESULTS_DIR}/eda_findings.json", "w") as f:
        json.dump({
            "numerical_summary": num_summary,
            "categorical_summary": cat_summary,
            "correlations_with_at_risk": correlations,
        }, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/eda_findings.json")