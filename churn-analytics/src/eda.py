"""
Phase 2: Exploratory Data Analysis for the Telco Customer Churn dataset.
All statistics are computed from the actual cleaned dataset — no
fabricated numbers.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

CLEAN_PATH = "data/telco_churn_clean.csv"
RESULTS_DIR = "results"

sns.set_style("whitegrid")


def numerical_summary(df, numerical_cols):
    summary = {}
    for col in numerical_cols:
        summary[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "q1": float(df[col].quantile(0.25)),
            "q3": float(df[col].quantile(0.75)),
            "skewness": float(df[col].skew()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
    return summary


def categorical_summary(df, categorical_cols, target_col="Churn"):
    summary = {}
    for col in categorical_cols:
        freq = df[col].value_counts().to_dict()
        churn_rate = df.groupby(col)[target_col].apply(
            lambda x: (x == "Yes").mean() * 100
        ).to_dict()
        summary[col] = {
            "frequency": freq,
            "churn_rate_pct": {k: round(v, 2) for k, v in churn_rate.items()},
        }
    return summary


def plot_churn_distribution(df):
    plt.figure(figsize=(6, 5))
    counts = df["Churn"].value_counts()
    plt.bar(counts.index, counts.values, color=["#4C72B0", "#C44E52"])
    plt.title("Overall Churn Distribution")
    plt.ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 50, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/churn_distribution.png", dpi=150)
    plt.close()


def plot_numerical_distributions(df, numerical_cols):
    fig, axes = plt.subplots(1, len(numerical_cols), figsize=(5 * len(numerical_cols), 4))
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, hue="Churn", kde=True, ax=axes[i], element="step")
        axes[i].set_title(f"{col} Distribution by Churn")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/numerical_distributions.png", dpi=150)
    plt.close()


def plot_boxplots(df, numerical_cols):
    fig, axes = plt.subplots(1, len(numerical_cols), figsize=(5 * len(numerical_cols), 5))
    for i, col in enumerate(numerical_cols):
        sns.boxplot(data=df, x="Churn", y=col, ax=axes[i])
        axes[i].set_title(f"{col} by Churn")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/boxplots.png", dpi=150)
    plt.close()


def plot_categorical_churn_rates(df, categorical_cols):
    n_cols = 3
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        churn_rate = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
        churn_rate = churn_rate.sort_values(ascending=False)
        axes[i].bar(churn_rate.index.astype(str), churn_rate.values, color="#DD8452")
        axes[i].set_title(f"Churn Rate by {col}")
        axes[i].set_ylabel("Churn Rate (%)")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/categorical_churn_rates.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df, numerical_cols):
    df_corr = df.copy()
    df_corr["Churn_numeric"] = (df_corr["Churn"] == "Yes").astype(int)
    df_corr["SeniorCitizen"] = df_corr["SeniorCitizen"].astype(int)

    corr_cols = numerical_cols + ["SeniorCitizen", "Churn_numeric"]
    corr_matrix = df_corr[corr_cols].corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numerical Features + Churn)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()

    return corr_matrix["Churn_numeric"].drop("Churn_numeric").to_dict()


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ]

    print("=== Numerical Summary ===")
    num_summary = numerical_summary(df, numerical_cols)
    for col, stats in num_summary.items():
        print(f"\n{col}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}")

    print("\n=== Generating Visualizations ===")
    plot_churn_distribution(df)
    print("  Saved churn_distribution.png")

    plot_numerical_distributions(df, numerical_cols)
    print("  Saved numerical_distributions.png")

    plot_boxplots(df, numerical_cols)
    print("  Saved boxplots.png")

    plot_categorical_churn_rates(df, categorical_cols)
    print("  Saved categorical_churn_rates.png")

    correlations = plot_correlation_heatmap(df, numerical_cols)
    print("  Saved correlation_heatmap.png")

    print("\n=== Correlation with Churn (numerical features) ===")
    for feat, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat}: {corr:.4f}")

    cat_summary = categorical_summary(df, categorical_cols)

    # Save all findings as JSON for use in the final report
    with open(f"{RESULTS_DIR}/eda_findings.json", "w") as f:
        json.dump({
            "numerical_summary": num_summary,
            "categorical_summary": cat_summary,
            "numerical_correlations_with_churn": correlations,
        }, f, indent=2)

    print(f"\nAll findings saved to {RESULTS_DIR}/eda_findings.json")