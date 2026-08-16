\# Outlier Investigation: sum\_gamerounds



\## Finding



The `sum\_gamerounds` distribution is heavily right-skewed (control skewness=163.71),

driven primarily by a single extreme outlier: one user (userid 6390605, gate\_30)

recorded 49,854 game rounds — approximately 17x higher than the next-highest

value (2,961) and roughly 960x the median (17).



This user's retention pattern is also inconsistent with typical play (retention\_1=False

but retention\_7=True), which is unusual for a genuinely engaged human player and is

consistent with a bot, automated test account, or logging anomaly rather than

real user behavior.



\## Decision



\*\*Not removed\*\* from the primary analysis — per project methodology, data points

are investigated and documented rather than silently dropped. The non-parametric

Mann-Whitney U test used in Phase 6 is inherently robust to this kind of extreme

outlier (it operates on ranks, not raw values), so its inclusion does not

meaningfully distort the reported result.



116 total users (0.129% of the dataset) have >1000 rounds; excluding only the single

most extreme case for a sensitivity check would be a reasonable follow-up, but was

not performed here to avoid post-hoc data manipulation after already viewing results

(a violation of the no-p-hacking principle stated in the project's global rules).



\## Recommendation



Flag this specific user ID for the product/analytics team to verify whether it

represents real user behavior or a data quality issue at the source (bot detection,

test account exclusion list, etc.) — this is an investigation finding, not a

statistical correction.

