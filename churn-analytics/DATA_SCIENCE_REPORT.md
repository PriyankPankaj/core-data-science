# Customer Churn Analytics — Final Report

## 1. Problem
Predict which customers are likely to churn and identify actionable drivers.

## 2. Dataset
IBM Telco Customer Churn — 7,043 real customers, publicly sourced.

## 3. Methodology
EDA -> Statistical Testing -> Feature Engineering -> ML (3 models, stratified 5-fold CV) -> Interpretability (SHAP) -> Segmentation.

## 4. Results
Best model: **RandomForest_balanced (improved)**, Test ROC-AUC=0.8399. See RESUME_METRICS.md for full metrics.

## 5. Limitations
Single static dataset (no time-series churn trends); segmentation and models trained on the same population without external validation set.

## 6. Future Improvements
Time-series churn modeling, external validation on a second cohort, cost-sensitive threshold tuning using real business cost estimates.
