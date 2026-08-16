# ExperimentLab — Cookie Cats Gate Placement A/B Test

## 1. Problem
Does moving a progression gate from level 30 to 40 change player retention?

## 2. Dataset
Real, public Cookie Cats dataset (90,189 players), via Kaggle.

## 3. Methodology
Randomization validation -> two-proportion z-tests -> power analysis -> continuous metric testing -> multiple testing correction -> segment analysis -> Monte Carlo simulation -> decision engine.

## 4. Results
retention_7 shows a significant, robust negative effect (-0.82pp, p=0.0016), concentrated in the highest-engagement player segment.

## 5. Decision
**DO NOT SHIP**

## 6. Limitations
No event-level timestamps in source data (Phase 11 time-series analysis limited); segments derived from engagement quartiles rather than demographic data, which isn't present in this dataset.

## 7. Future Improvements
Request timestamped event logs; test additional gate positions; run a follow-up experiment targeting high-engagement players specifically.
