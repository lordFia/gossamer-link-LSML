import numpy as np
import pandas as pd
import json

def mutate(vec, intensity):
    noise = np.random.normal(0, intensity, size=len(vec))
    return vec + noise

def update_trust(before, feedback, rate):
    return before + rate * (feedback - before)

def behavior_feedback(trust):
    if trust > 0.7:
        return 1.0
    elif trust < 0.3:
        return 0.0
    else:
        return np.random.rand()

def main():
    with open("config.json") as f:
        cfg = json.load(f)

    steps = cfg["steps"]
    nodes = cfg["nodes"]
    intensity = cfg["mutation_intensity"]
    trust_rate = cfg["trust_update_rate"]

    essence = np.random.rand(nodes, 8)
    trust = np.random.rand(nodes)

    drift_log = []
    trust_log = []
    mutation_log = []

    for step in range(steps):
        feedback = np.array([behavior_feedback(t) for t in trust])

        for i in range(nodes):
            before = trust[i]
            trust[i] = update_trust(before, feedback[i], trust_rate)

            # --- 正しい drift 計測 ---
            old_vec = essence[i]
            new_vec = mutate(old_vec, intensity)
            drift = np.linalg.norm(new_vec - old_vec)
            essence[i] = new_vec
            # --------------------------

            drift_log.append([step, i, drift])
            trust_log.append([step, i, before, trust[i]])
            mutation_log.append([step, i, intensity])

    pd.DataFrame(drift_log, columns=["step", "node", "drift"]).to_csv("essence_drift.csv", index=False)
    pd.DataFrame(trust_log, columns=["step", "node", "trust_before", "trust_after"]).to_csv("trust_history.csv", index=False)
    pd.DataFrame(mutation_log, columns=["step", "node", "strength"]).to_csv("mutation_events.csv", index=False)

if __name__ == "__main__":
    main()
