\# Phase 11 (Time Analysis) — Limitation Note



The Cookie Cats dataset used throughout this project (Phases 1-10, 12) is

a static, per-user snapshot — final retention\_1/retention\_7 status per

player — not an event-level time-series log with timestamps.



A genuine daily/cumulative conversion-over-time analysis (as specified)

would require either:

1\. Real per-event timestamps (not available in this public dataset), or

2\. Switching to a different dataset for this phase alone



Since (2) would mean analyzing a completely different experiment

(different users, different metric, different context) under the same

report, doing so would misleadingly suggest continuity where none exists.

Rather than fabricate timestamps or dilute the report's coherence, this

phase is documented as a known limitation.



\*\*What real time analysis would examine, if timestamp data were available:\*\*

\- Daily conversion/retention rate by group, checking for early-experiment

&#x20; instability before rates settle

\- Cumulative retention trend, checking whether the treatment effect

&#x20; strengthens, weakens, or stays constant as more data accumulates

\- Novelty effects — whether an initially large gap between groups

&#x20; shrinks over time as users adjust to the change



\*\*Recommendation for a production version of this project\*\*: request

event-level timestamped logs from the data source if this analysis is

genuinely needed for decision-making.

