"""
Loads the real Cookie Cats A/B testing dataset via Kaggle (kagglehub),
routing around the GitHub raw-content network issues hit on this machine.
"""
import pandas as pd
import kagglehub
import os
import shutil

LOCAL_PATH = "data/cookie_cats_raw.csv"


def load_raw_data():
    dataset_path = kagglehub.dataset_download("yufengsui/mobile-games-ab-testing")
    print(f"Downloaded to: {dataset_path}")

    # Find the CSV file in the downloaded folder
    csv_file = None
    for f in os.listdir(dataset_path):
        if f.endswith(".csv"):
            csv_file = os.path.join(dataset_path, f)
            break

    if csv_file is None:
        raise FileNotFoundError("No CSV found in downloaded Kaggle dataset")

    shutil.copy(csv_file, LOCAL_PATH)
    return pd.read_csv(LOCAL_PATH)


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nGroup sizes:\n{df['version'].value_counts()}")
    print(f"\nRetention_1 rate by group:\n{df.groupby('version')['retention_1'].mean()}")
    print(f"\nRetention_7 rate by group:\n{df.groupby('version')['retention_7'].mean()}")
    print(f"\nSaved to {LOCAL_PATH}")