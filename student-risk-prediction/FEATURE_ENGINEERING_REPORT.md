# Feature Engineering Report

- Original feature count: 34
- Engineered feature count: 6
- Final feature count: 40

## Engineered Features & Rationale

### `failure_risk_flag`

Binary flag: True if failures > 0. failures was the strongest single predictor (rank-biserial=-0.40, p<0.0001); a binary flag isolates 'has failed before' from the count magnitude, which may be a cleaner signal for models sensitive to the resulting skewed distribution (skewness=3.09).

### `parental_education_avg`

Average of Medu and Fedu. Both individually showed significant, similarly-directed negative correlation with risk in Phase 3/4 (Medu: -0.14, Fedu: -0.15); combining captures overall parental educational background as a single signal rather than two correlated ones.

### `social_engagement`

goout + freetime combined. Both showed weaker but present correlations with risk; combined as a general social/leisure engagement proxy.

### `study_efficiency`

studytime / (failures + 1), avoiding division by zero. Captures whether study time is translating into avoided failures — a student with high studytime AND high failures may indicate different circumstances than raw studytime alone suggests.

### `total_alcohol_consumption`

Dalc + Walc combined. Both individually significant (p<0.005 each) with similar direction; combining into a single weekly consumption proxy reduces redundant correlated features.

### `wants_higher_no_support`

Binary flag: True if higher=='yes' AND schoolsup=='no'. higher was the strongest categorical predictor (Cramér's V=0.30); this flag identifies students who are motivated (want higher education) but lack formal school support — a potentially actionable intervention target.

