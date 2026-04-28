import pandas as pd
import numpy as np
import json

def compute_diversity(df):
    vectors = df[[c for c in df.columns if c.startswith("e")]].values
    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dists.append(np.linalg.norm(vectors[i] - vectors[j]))
    return float(np.mean(dists))

def detect_collapse(diversity_series):
    return float(np.mean(diversity_series[-50:])) < 0.1

def main():
    df = pd.read_csv("diversity_over_time.csv")
    diversity = compute_diversity(df)
    collapsed = detect_collapse(df["diversity"].values)

    result = {
        "diversity": diversity,
        "collapsed": collapsed
    }

    with open("summary.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
