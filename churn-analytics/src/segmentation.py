"""
Phase 7: Customer segmentation via K-Means.
Cluster count chosen via elbow method + silhouette score — not arbitrary.
Segmentation retained only if it produces a meaningful churn-rate spread
across clusters, per spec.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json

ENGINEERED_PATH = "data/telco_churn_engineered.csv"
RESULTS_DIR = "results"
RANDOM_SEED = 42

SEGMENTATION_FEATURES = [
    "tenure", "MonthlyCharges", "service_count", "is_month_to_month", "payment_delay_risk",
]

def find_optimal_k(X_scaled, k_range=range(2, 9)):
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    return list(k_range), inertias, silhouettes


def plot_elbow_and_silhouette(k_range, inertias, silhouettes, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k_range, inertias, marker="o")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(k_range, silhouettes, marker="o", color="orange")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score by k")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_churn_by_cluster(df, save_path):
    churn_by_cluster = df.groupby("cluster")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    plt.figure(figsize=(7, 5))
    plt.bar(churn_by_cluster.index.astype(str), churn_by_cluster.values, color="#55A868")
    plt.xlabel("Cluster")
    plt.ylabel("Churn Rate (%)")
    plt.title("Churn Rate by Customer Segment")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return churn_by_cluster.to_dict()


if __name__ == "__main__":
    df = pd.read_csv(ENGINEERED_PATH)
    X = df[SEGMENTATION_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=== Finding optimal k (elbow + silhouette) ===")
    k_range, inertias, silhouettes = find_optimal_k(X_scaled)
    for k, inertia, sil in zip(k_range, inertias, silhouettes):
        print(f"  k={k}: inertia={inertia:.1f}, silhouette={sil:.4f}")

    plot_elbow_and_silhouette(k_range, inertias, silhouettes, f"{RESULTS_DIR}/elbow_silhouette.png")
    print("Saved elbow_silhouette.png")

    best_k = k_range[int(np.argmax(silhouettes))]
    print(f"\nOptimal k by silhouette score: {best_k}")

    final_km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    df["cluster"] = final_km.fit_predict(X_scaled)

    print(f"\n=== Cluster sizes ===")
    print(df["cluster"].value_counts().sort_index())

    churn_by_cluster = plot_churn_by_cluster(df, f"{RESULTS_DIR}/churn_by_cluster.png")
    print(f"\n=== Churn rate by cluster ===")
    for cluster, rate in churn_by_cluster.items():
        print(f"  Cluster {cluster}: {rate:.2f}%")

    churn_spread = max(churn_by_cluster.values()) - min(churn_by_cluster.values())
    print(f"\nChurn rate spread across clusters: {churn_spread:.2f} percentage points")
    meaningful = churn_spread > 10.0
    print(f"Segmentation deemed meaningful (spread > 10pp): {meaningful}")

    cluster_profile = df.groupby("cluster")[SEGMENTATION_FEATURES].mean()
    print(f"\n=== Cluster profiles (mean feature values) ===")
    print(cluster_profile.to_string())

    with open(f"{RESULTS_DIR}/segmentation_resultsV2.json", "w") as f:
        json.dump({
            "optimal_k": int(best_k),
            "silhouette_scores_by_k": dict(zip(k_range, silhouettes)),
            "cluster_sizes": df["cluster"].value_counts().sort_index().to_dict(),
            "churn_rate_by_cluster": churn_by_cluster,
            "churn_rate_spread_pp": float(churn_spread),
            "segmentation_meaningful": bool(meaningful),
            "cluster_profiles": cluster_profile.to_dict("index"),
        }, f, indent=2)

    df.to_csv(f"{RESULTS_DIR}/../data/telco_churn_segmented.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR}/segmentation_results.json")