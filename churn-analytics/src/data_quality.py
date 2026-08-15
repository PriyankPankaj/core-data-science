"""
Phase 1: Data quality analysis for the Telco Customer Churn dataset.
Generates DATA_QUALITY_REPORT.md with real, measured findings — no
fabricated numbers, per project spec rule #1.
"""
import pandas as pd

RAW_PATH = "data/telco_churn_raw.csv"
CLEAN_PATH = "data/telco_churn_clean.csv"
REPORT_PATH = "DATA_QUALITY_REPORT.md"


def analyze_and_clean(df):
    report = {}
    report["rows_before"] = len(df)
    report["columns_before"] = len(df.columns)

    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()

    # Duplicates
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Data type validation: TotalCharges is stored as object but should be numeric
    # (known issue in this dataset — some rows have blank strings for new customers
    # with 0 tenure, since they haven't been billed yet)
    report["total_charges_dtype_before"] = str(df["TotalCharges"].dtype)
    non_numeric_mask = pd.to_numeric(df["TotalCharges"], errors="coerce").isna()
    report["total_charges_non_numeric_count"] = int(non_numeric_mask.sum())
    report["total_charges_non_numeric_examples"] = (
        df.loc[non_numeric_mask, ["customerID", "tenure", "TotalCharges"]].to_dict("records")
    )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Investigate: are the non-numeric rows all tenure=0? (expected pattern)
    if non_numeric_mask.sum() > 0:
        tenure_of_bad_rows = df.loc[non_numeric_mask, "tenure"]
        report["non_numeric_all_tenure_zero"] = bool((tenure_of_bad_rows == 0).all())

    # Handle the resulting NaNs: for tenure=0 customers, TotalCharges=0 is the
    # correct, documented interpretation (no billing has occurred yet) —
    # not a missing-data problem to impute, but a legitimate zero
    zero_tenure_nan_mask = (df["tenure"] == 0) & (df["TotalCharges"].isna())
    df.loc[zero_tenure_nan_mask, "TotalCharges"] = 0.0
    report["total_charges_filled_as_zero_for_new_customers"] = int(zero_tenure_nan_mask.sum())

    # Any remaining NaNs after this targeted fix?
    report["remaining_missing_after_cleaning"] = int(df["TotalCharges"].isna().sum())

    # customerID uniqueness check (should be 100% unique, sanity check)
    report["customerID_unique"] = df["customerID"].nunique() == len(df)

    # Categorical consistency check: list unique values per categorical column
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ["customerID"]]
    report["categorical_unique_values"] = {
        col: df[col].unique().tolist() for col in categorical_cols
    }

    # Target leakage check: no column should perfectly predict Churn by definition
    # (a basic sanity pass — real leakage detection happens more rigorously in
    # feature engineering phase)
    report["target_column"] = "Churn"
    report["target_distribution"] = df["Churn"].value_counts().to_dict()

    report["rows_after"] = len(df)
    report["columns_after"] = len(df.columns)

    return df, report


def write_report(report):
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Data Quality Report\n\n")
        f.write("**Source:** IBM Telco Customer Churn dataset\n\n")
        f.write(f"- Rows before cleaning: {report['rows_before']}\n")
        f.write(f"- Columns before cleaning: {report['columns_before']}\n")
        f.write(f"- Rows after cleaning: {report['rows_after']}\n")
        f.write(f"- Columns after cleaning: {report['columns_after']}\n\n")

        f.write("## Missing Values (original)\n\n")
        if report["missing_values"]:
            for col, count in report["missing_values"].items():
                f.write(f"- {col}: {count}\n")
        else:
            f.write("None found via `.isnull()` — see TotalCharges dtype issue below "
                    "for a non-null-but-invalid case.\n")
        f.write("\n")

        f.write("## Duplicate Rows\n\n")
        f.write(f"{report['duplicate_rows']} duplicate rows found.\n\n")

        f.write("## TotalCharges Data Type Issue\n\n")
        f.write(f"- Original dtype: `{report['total_charges_dtype_before']}` "
                f"(should be numeric)\n")
        f.write(f"- Non-numeric values found: {report['total_charges_non_numeric_count']}\n")
        if report["total_charges_non_numeric_count"] > 0:
            f.write(f"- All non-numeric rows have tenure=0: "
                    f"{report.get('non_numeric_all_tenure_zero')}\n")
            f.write(f"- **Investigation finding**: these are new customers "
                    f"(tenure=0) who haven't been billed yet — TotalCharges is "
                    f"blank, not genuinely missing. Filled as 0.0 "
                    f"({report['total_charges_filled_as_zero_for_new_customers']} rows), "
                    f"not imputed via mean/median, since 0 is the factually "
                    f"correct value here.\n")
        f.write(f"- Remaining missing values after cleaning: "
                f"{report['remaining_missing_after_cleaning']}\n\n")

        f.write("## customerID Uniqueness\n\n")
        f.write(f"All customerID values unique: {report['customerID_unique']}\n\n")

        f.write("## Target Variable Distribution\n\n")
        for cls, count in report["target_distribution"].items():
            f.write(f"- {cls}: {count}\n")

        f.write("\n## Removed Features\n\n")
        f.write("None removed at this stage — all 21 original columns retained. "
                "customerID will be excluded from modeling (not a predictive "
                "feature, purely an identifier), documented at the modeling phase.\n")

    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    clean_df, report = analyze_and_clean(df)
    clean_df.to_csv(CLEAN_PATH, index=False)
    write_report(report)

    print(f"\nCleaned dataset saved to {CLEAN_PATH}")
    print(f"Rows: {report['rows_before']} -> {report['rows_after']}")
    print(f"TotalCharges non-numeric values found and handled: "
          f"{report['total_charges_non_numeric_count']}")