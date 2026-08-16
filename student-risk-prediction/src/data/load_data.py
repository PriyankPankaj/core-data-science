"""
Phase 1: Loads the real UCI Student Performance dataset (Math course).

Source: UCI Machine Learning Repository — Student Performance Data Set
(Cortez & Silva, 2008), commonly mirrored on Kaggle/GitHub for direct
access. Real, publicly documented dataset — not synthetic.
"""
import pandas as pd
import kagglehub
import os
import shutil

LOCAL_PATH = "data/student_performance_raw.csv"


def load_raw_data():
    dataset_path = kagglehub.dataset_download("larsen0966/student-performance-data-set")
    print(f"Downloaded to: {dataset_path}")

    print(f"Files found: {os.listdir(dataset_path)}")

    csv_file = None
    for f in os.listdir(dataset_path):
        if "mat" in f.lower() and f.endswith(".csv"):
            csv_file = os.path.join(dataset_path, f)
            break
    if csv_file is None:
        for f in os.listdir(dataset_path):
            if f.endswith(".csv"):
                csv_file = os.path.join(dataset_path, f)
                break

    if csv_file is None:
        raise FileNotFoundError("No CSV found in downloaded dataset")

    print(f"Using file: {csv_file}")

    # Auto-detect delimiter: try comma first, check if we got real columns
    df = pd.read_csv(csv_file)
    if df.shape[1] == 1:
        # single-column result means wrong delimiter, retry with semicolon
        df = pd.read_csv(csv_file, sep=";")

    df.to_csv(LOCAL_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nG3 (final grade) distribution:\n{df['G3'].describe()}")
    print(f"\nSaved to {LOCAL_PATH}")