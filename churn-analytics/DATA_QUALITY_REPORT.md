# Data Quality Report

**Source:** IBM Telco Customer Churn dataset

- Rows before cleaning: 7043
- Columns before cleaning: 21
- Rows after cleaning: 7043
- Columns after cleaning: 21

## Missing Values (original)

None found via `.isnull()` — see TotalCharges dtype issue below for a non-null-but-invalid case.

## Duplicate Rows

0 duplicate rows found.

## TotalCharges Data Type Issue

- Original dtype: `str` (should be numeric)
- Non-numeric values found: 11
- All non-numeric rows have tenure=0: True
- **Investigation finding**: these are new customers (tenure=0) who haven't been billed yet — TotalCharges is blank, not genuinely missing. Filled as 0.0 (11 rows), not imputed via mean/median, since 0 is the factually correct value here.
- Remaining missing values after cleaning: 0

## customerID Uniqueness

All customerID values unique: True

## Target Variable Distribution

- No: 5174
- Yes: 1869

## Removed Features

None removed at this stage — all 21 original columns retained. customerID will be excluded from modeling (not a predictive feature, purely an identifier), documented at the modeling phase.
