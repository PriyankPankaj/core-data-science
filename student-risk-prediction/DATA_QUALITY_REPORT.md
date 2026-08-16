# Data Quality Report

**Source:** UCI Student Performance dataset (Portuguese course)

- Rows before cleaning: 649
- Columns before cleaning: 34
- Rows after cleaning: 649
- Feature columns for modeling (excluding G1/G2/G3 and target): 31

## Missing Values

None found - dataset is complete.

## Duplicate Rows

0 duplicate rows found.

## Data Type Issues

None found.

## Invalid Values (out of documented range)

None found - all values within documented UCI dataset ranges.

## Outlier Investigation: absences

Mean=3.66, Median=2.00, Max=32, 99th percentile=21.00

## Target Leakage Prevention

Excluded from modeling features: ['G1', 'G2', 'G3'] (prior/target grade values that would leak the outcome)

## Class Imbalance

- Not At Risk: 549
- At Risk: 100

This is a real, meaningfully imbalanced classification problem (15.41% positive class), addressed explicitly in later phases (class weighting, threshold tuning).
