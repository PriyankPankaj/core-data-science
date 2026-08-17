# Resume-Ready Metrics (all measured, none fabricated)

- Dataset: 649 real students (UCI Student Performance, Portuguese course)
- At-risk rate: 15.4% (real class imbalance)
- Statistical tests: 30 (t-test/Mann-Whitney, Chi-square), Bonferroni + BH-FDR corrected
- Significant predictors (Bonferroni): 7
- Models compared: 5 (Logistic Regression, Linear SVM, RBF SVM, Random Forest, XGBoost)
- SVM kernels evaluated: 2 (Linear, RBF)
- Feature selection methods: 5 (All, F-test, Mutual Information, RFE, SVM-RFE)
- Cross-validation folds: 5 (stratified)
- Best F1: Random Forest (-) — 0.5116
- Best Recall: XGBoost (-) — 0.6500
- Best feature selection (F1): Statistical (F-test) (0.5306)
