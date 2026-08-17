# Student Performance & At-Risk Prediction — Final Report

## 1. Problem Definition
Predict academic at-risk status early enough for intervention.

## 2. Dataset
UCI Student Performance dataset (Portuguese course), 649 real students.

## 3. Target Construction
G3 < 10 (standard 0-20 scale pass/fail boundary). G1/G2/G3 excluded from features (leakage prevention).

## 4-7. Data Quality, EDA, Statistics, Feature Engineering
See DATA_QUALITY_REPORT.md and FEATURE_ENGINEERING_REPORT.md.

## 8-13. Model Results
              model             kernel_or_variant  test_f1  test_recall  test_roc_auc
      Random Forest                             - 0.511628         0.55      0.822273
                SVM                       SVM_RBF 0.500000         0.60      0.812727
            XGBoost                             - 0.500000         0.65      0.747500
Logistic Regression   LogisticRegression_balanced 0.468085         0.55      0.759545
                SVM                    SVM_Linear 0.434783         0.50      0.786818
Logistic Regression LogisticRegression_unweighted 0.214286         0.15      0.773636

## 14. Explainability
failures, total_alcohol_consumption, and higher-ed aspiration are the strongest predictors across permutation importance and SHAP. Associations only, not causal claims.

## 18. Conclusion
XGBoost maximizes recall (65%) for catching at-risk students; Random Forest offers the best overall balance. Model choice should reflect the real cost trade-off between missed at-risk students and false alarms.
