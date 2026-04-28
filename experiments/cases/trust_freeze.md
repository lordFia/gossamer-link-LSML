# Breakdown Case — Trust Freeze

## Summary
Trust values stop updating, causing nodes to become unresponsive to feedback.

## Trigger Conditions
- trust update dampening too strong
- feedback distribution = skewed_negative
- mutation_intensity = low

## Symptoms
- trust_before == trust_after for 100+ steps
- trust_convergence ≈ 0
- diversity remains static

## Interpretation
System loses ability to adapt because trust becomes inert.

## Expected Output Signals
- trust_history.csv shows flat lines
- mutation_events.csv shows minimal activity
