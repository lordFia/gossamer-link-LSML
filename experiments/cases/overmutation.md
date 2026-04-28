# Breakdown Case — Overmutation Runaway

## Summary
Mutation strength escalates uncontrollably, causing chaotic Essence drift.

## Trigger Conditions
- mutation_intensity = high
- feedback = bursty or adversarial
- trust variance > 0.7

## Symptoms
- mutation_strength variance > 0.5
- drift magnitude grows without bound
- trust fails to stabilize

## Interpretation
System becomes unstable and cannot maintain coherent behavior.

## Expected Output Signals
- mutation_events.csv shows repeated high-strength events
- drift values spike sharply
