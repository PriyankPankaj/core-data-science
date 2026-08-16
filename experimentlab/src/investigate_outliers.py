"""
Investigates the extreme sum_gamerounds outlier(s) per spec rule: never
silently remove data points without investigating and documenting why.
"""
import pandas as pd

RAW_PATH = "data/cookie_cats_raw.csv"

if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)

    print("=== Top 10 sum_gamerounds values ===")
    print(df.nlargest(10, "sum_gamerounds")[["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]])

    print(f"\n=== Distribution context ===")
    print(f"Mean: {df['sum_gamerounds'].mean():.2f}")
    print(f"Median: {df['sum_gamerounds'].median():.2f}")
    print(f"99th percentile: {df['sum_gamerounds'].quantile(0.99):.2f}")
    print(f"Max: {df['sum_gamerounds'].max()}")

    extreme_count = (df["sum_gamerounds"] > 1000).sum()
    print(f"\nPlayers with >1000 rounds: {extreme_count} out of {len(df)} ({extreme_count/len(df)*100:.3f}%)")