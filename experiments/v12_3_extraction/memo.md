# v12.3.0 Extraction

## Core
Introduced conditional attraction and repulsion during extraction.

## Mechanism
If similarity < 0.6 → attraction (0.12)
If similarity ≥ 0.6 → repulsion (0.06)

## Structural Role
Prevents collapse while preserving selective extraction.

## Result
mean_variance within [0.02, 0.06] across all runs
trust spread ≥ 0.2
no structural collapse

## Conclusion
Extraction stabilized with diversity preservation.
