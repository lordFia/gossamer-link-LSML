import pandas as pd
import numpy as np
import json

def compute_stability(df):
    return float(1 - np.var(df["strength"]))

def detect_runaway(df):
    return float(np.var(df["strength"])) > 0.5

def main():
    df = pd.read_csv("mutation_events.csv")
    stability = compute_stability(df)
    runaway = detect_runaway(df)

    result = {
        "mutation_stability": stability,
        "runaway": runaway
    }

    with open("summary.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
