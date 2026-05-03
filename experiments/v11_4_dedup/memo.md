# v11.4.1 Deduplication

## Summary
Global deduplication stabilized by replacing random refill with neighbor-based reconstruction.

## Mechanism
- Remove high-similarity vectors (>0.9) globally
- Refill using existing vectors + small noise
- Preserve slot count without introducing random spikes

## Result
- Variance stabilized (~0.031)
- Trust distribution maintained
- No collapse across 3 runs

## Conclusion
Deduplication is now stable and compatible with propagation and compression.
