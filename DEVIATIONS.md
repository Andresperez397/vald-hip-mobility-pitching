# Deviations from the plan

## Changes to how results are computed

1. **Winsorizing added after the first run.** The first run (ridge with imputation and scaling only) showed the static tests making predictions *worse* than guessing the mean, which is not plausible for a shrinkage model with a tuned penalty:

   | First run, no winsorizing | Release speed R² | Elbow torque R² |
   |---|---|---|
   | Baseline | 0.113 | 0.278 |
   | + hip tests | −1.33 | −0.26 |
   | + hip and shoulder tests | −1.00 | −0.38 |

   The cause was one pitcher's `asymmetry_hip_add` of 619% (every other pitcher is at 43% or less), a percentage with a near-zero denominator. After standardizing, that single value dominated the fit. Predictors are now clipped to the training fold's 2.5th and 97.5th percentiles, fit inside each fold. All reported numbers use the fixed version. The conclusion did not flip from "tests help" to "tests don't": the broken run only made a null result look like a loss.
2. **Question 2, first run.** Before winsorizing, no target beat the shuffled null (best gain 0.077, null 95th percentile 0.091, p = 0.085). After winsorizing, one target does (0.154 against 0.140, p = 0.025). The difference between the two runs is why that target is called borderline and not a finding.

## Added after seeing the null result

3. **Power analysis (`scripts/06_power.py`).** Added after the first conclusions were drafted. It showed the design detects a planted gain of 0.20 R² only 44% (speed) and 22% (torque) of the time, so the wording changed from "static tests add nothing" to "no detectable gain; large effects ruled out, small ones not" in the README, summary and figures.

6. **Sample-size planning (`scripts/07_sample_size.py`).** Added after the power analysis, to turn "the design was too small" into a number for a follow-up. A first pass with 60 simulations per cell was too noisy (power not monotone in n) and was replaced by 150 per cell; only the 150-simulation run is reported. Sample sizes 100 to 450 and planted gains 0.05, 0.10, 0.15 were fixed before running.

## Not in the plan

4. **Spearman correlations** of each static test with each outcome are in `reports/tables/results.json`. They are shown for context and were not used to select predictors.
5. **The raw-table comparison** (`scripts/03_uncleaned_comparison.py`) was added to quantify what the audit prevents.
