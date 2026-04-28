# Breakdown Case — Sybil Takeover

## Summary
Sybil nodes dominate trust ranking and distort system behavior.

## Trigger Conditions
- sybil_ratio ≥ 0.2
- feedback = adversarial
- trust update sensitivity high

## Symptoms
- sybil trust_share > 0.4
- non-sybil nodes collapse in trust
- diversity becomes polarized

## Interpretation
System becomes compromised by malicious nodes.

## Expected Output Signals
- sybil_impact.json shows trust_share > 0.4
- trust_history.csv shows divergence between sybil and non-sybil
