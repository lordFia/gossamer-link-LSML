# Breakdown Case — Diversity Collapse

## Summary
All nodes converge to nearly identical Essence vectors, eliminating behavioral diversity.

## Trigger Conditions
- initial distribution = identical or clustered
- mutation_intensity = low
- feedback = skewed_positive or uniform
- trust convergence < 0.2

## Symptoms
- diversity < 0.1 for 50+ steps
- mutation events drop to near zero
- trust values converge tightly

## Interpretation
System becomes overly stable and loses adaptive capacity.

## Expected Output Signals
- diversity_over_time.csv shows monotonic decline
- essence_drift.csv shows minimal movement
