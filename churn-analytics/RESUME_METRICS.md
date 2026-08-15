# Resume-Ready Metrics (all measured, none fabricated)

- Dataset: 7,043 customers, 21 original + 7 engineered features
- Statistical tests: 19 hypothesis tests run (Mann-Whitney U, Chi-square), Bonferroni + Benjamini-Hochberg FDR correction applied
- Best model: RandomForest_balanced (improved)
- Test ROC-AUC: 0.8399
- Test F1 (default threshold): 0.6305
- Test F1 (optimized threshold 0.5): 0.6305
- Precision: 0.5377
- Recall: 0.7620
- Customer segments identified: 6 (via elbow method + silhouette score)
- Churn rate spread across segments: 54.1 percentage points
