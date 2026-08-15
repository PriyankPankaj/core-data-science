"""
Loads the IBM Telco Customer Churn dataset — a real, public dataset
(not synthetic), as required by the project spec.

Source: IBM Sample Data Sets, mirrored on GitHub for direct CSV access.
"""
import pandas as pd
import requests
import io
import time

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_PATH = "data/telco_churn_raw.csv"


def load_raw_data(max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(DATA_URL, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df.to_csv(LOCAL_PATH, index=False)
            return df
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to load dataset after {max_retries} attempts: {last_error}")


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nTarget distribution (Churn):\n{df['Churn'].value_counts()}")
    print(f"\nSaved to: {LOCAL_PATH}")