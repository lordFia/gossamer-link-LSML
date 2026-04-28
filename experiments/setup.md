# Synaptic Mesh — Experiment Setup (v1.4)

## 1. Environment

Experiments require:
- deterministic random seed
- fixed Essence dimension (n = 16)
- identical mutation operators across runs
- identical trust update parameters

No external models or inference engines are used.

---

## 2. Reproducibility Rules

1. All random seeds must be logged  
2. All feedback events must be recorded  
3. All Essence vectors must be saved per step  
4. Trust updates must be logged before and after  
5. Mutation events must include type and magnitude  

---

## 3. Logging Format

### 3.1 essence_drift.csv
step, node_id, e1, e2, ..., e16

### 3.2 trust_history.csv
step, node_id, trust_before, trust_after

### 3.3 mutation_events.csv
step, node_id, mutation_type, strength

### 3.4 diversity_over_time.csv
step, diversity

### 3.5 sybil_impact.json
{
  "sybil_ratio": 0.2,
  "trust_share": 0.31
}

---

## 4. Run Length

- Minimum: 100 steps  
- Recommended: 300–500 steps  
- Long-run tests: 1000+ steps (v1.5)

---

## 5. Notes

This setup ensures that all v1.4 experiments are reproducible and comparable.  
v1.5 will introduce automated analysis scripts.
