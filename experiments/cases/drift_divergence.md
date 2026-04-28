# Breakdown Case — Drift Divergence

## Summary
Essence vectors drift endlessly without converging, preventing stable behavior.

## Trigger Conditions
- mutation_intensity = medium or high
- feedback = bursty
- trust variance remains high

## Symptoms
- drift magnitude increases linearly or exponentially
- diversity oscillates instead of stabilizing
- trust never converges

## Interpretation
System cannot settle into a stable evolutionary basin.

## Expected Output Signals
- essence_drift.csv shows continuous movement
- diversity_over_time.csv shows repeated oscillation
