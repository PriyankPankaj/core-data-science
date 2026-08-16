"""
Phase 11: Loads a real, timestamped A/B test dataset to enable genuine
time-series analysis (daily conversion, cumulative trends) — something
the static Cookie Cats snapshot cannot support honestly.

Source: Udacity's A/B testing course dataset (real, user_id/timestamp/
group/landing_page/converted), commonly mirrored on Kaggle.
"""
import pandas as pd
import kagglehub
import os
import shutil

LOCAL_PATH = "data/ab_test_timeseries_raw.csv"


def load_raw_data():
    dataset_path = kagglehub.dataset_download("zhangluyuan/ab-testing")
    print(f"Downloaded to: {dataset_path}")

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
    print(df.head())