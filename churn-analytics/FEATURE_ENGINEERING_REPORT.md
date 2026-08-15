# Feature Engineering Report

- Original feature count: 21
- Engineered feature count: 7
- Final feature count: 28

## Engineered Features & Rationale

### `avg_monthly_spend`

TotalCharges / tenure (with tenure=0 handled as MonthlyCharges itself, since new customers have no historical average yet). Disentangles a customer's typical spend level from how long they've been a customer, since raw TotalCharges is confounded with tenure (Phase 2 finding).

### `charges_per_service`

MonthlyCharges / (service_count + 1), avoiding division by zero for customers with 0 add-on services. Captures whether a customer is paying a premium relative to how many services they actually use — a proxy for perceived value, which isn't directly present in any single original column.

### `has_internet_addons`

Binary flag: True if the customer has any of OnlineSecurity, OnlineBackup, DeviceProtection, or TechSupport. These four services only apply to customers with InternetService != 'No', so this captures a meaningful engagement signal specifically within the internet-subscriber segment.

### `is_month_to_month`

Binary flag for Contract == 'Month-to-month'. Contract type had the single strongest categorical association with churn in Phase 3 (Cramér's V = 0.41); isolating the highest-risk contract type as its own binary feature makes this the most direct possible signal for models that benefit from explicit flags over multi-category one-hot columns.

### `payment_delay_risk`

Binary flag: True if PaymentMethod is 'Electronic check' (historically the highest-churn payment method in this dataset per common domain knowledge, confirmed by Phase 3's significant PaymentMethod association, Cramér's V = 0.30).

### `service_count`

Count of add-on services subscribed (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, MultipleLines) that are 'Yes'. Phase 3 showed several individual services significantly associated with churn (Cramér's V 0.23-0.35); a combined count may capture overall customer engagement/lock-in more robustly than any single service.

### `tenure_bucket`

Groups raw tenure (0-72 months) into interpretable bands (New/Growing/Established/Loyal). Raw tenure showed a strong negative correlation with churn (-0.35 in Phase 3); bucketing helps capture non-linear risk thresholds a linear model alone might miss, and gives directly interpretable segments for business stakeholders.

