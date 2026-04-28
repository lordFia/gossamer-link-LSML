import pandas as pd
import numpy as np
import json

def compute_convergence(df):
    tail = df.tail(int(len(df) * 0.2))
    return float(np.var(tail["trust_after"]))

def detect_freeze(df):
    return bool((df["trust_before"] == df["trust_after"]).rolling(100).sum().max() >= 100)

def main():
    df = pd.read_csv("trust_history.csv")
    convergence = compute_convergence(df)
    freeze = detect_freeze(df)

    result = {
        "trust_convergence": convergence,
        "trust_freeze": freeze
    }

    with open("summary.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
