# experiments/v17_6_7_reusable_lineage_continuity_stabilization/code.py

import numpy as np
import random
import copy

# =========================================================
# constants
# =========================================================

BASE_SEED = 42

NODE_COUNT = 12
STEP_COUNT = 700

DIM = 8
ESSENCE_SLOTS = 3

MAX_CONNECTIONS = 4
SEARCH_K = 5

MIN_TRUST = 0.05
MAX_TRUST = 0.95

TRUST_DECAY = 0.02
UPDATE_SCALE = 0.05

TARGET_VAR_MIN = 0.02
TARGET_VAR_MAX = 0.06

ANOMALY_Z = 1.2

LINEAGE_MEMORY_LIMIT = 120
HYSTERESIS_DECAY = 0.985

# =========================================================
# node
# =========================================================

class Node:

    def __init__(self, node_id):

        self.id = node_id

        self.essences = np.random.randn(
            ESSENCE_SLOTS,
            DIM
        )

        self.essences /= (
            np.linalg.norm(
                self.essences,
                axis=1,
                keepdims=True
            ) + 1e-8
        )

        self.trust = random.uniform(0.3, 0.7)

        self.connections = set()

        self.persistence_counter = 0

        self.anomaly_score = 0.0
        self.is_anomaly = False

# =========================================================
# utility
# =========================================================

def similarity(a, b):

    return np.max(np.dot(a, b.T))

def compute_signature(nodes):

    return tuple(sorted([
        tuple(sorted(list(n.connections)))
        for n in nodes
    ]))

def topology_overlap(sig_a, sig_b):

    if sig_a is None or sig_b is None:
        return 0.0

    overlap = 0
    total = 0

    for a, b in zip(sig_a, sig_b):

        sa = set(a)
        sb = set(b)

        overlap += len(sa & sb)
        total += len(sa | sb)

    return overlap / (total + 1e-8)

def compute_variance(nodes):

    vectors = np.array([
        e
        for n in nodes
        for e in n.essences
    ])

    return float(
        np.mean(
            np.var(vectors, axis=0)
        )
    )

def compute_density(nodes):

    total_edges = sum(
        len(n.connections)
        for n in nodes
    )

    return total_edges / (
        NODE_COUNT * MAX_CONNECTIONS + 1e-8
    )

def compute_persistence(nodes):

    return float(np.mean([
        n.persistence_counter
        for n in nodes
    ]))

def compute_trust_range(nodes):

    values = [n.trust for n in nodes]

    return max(values) - min(values)

def compute_anomaly_ratio(nodes):

    return (
        sum(n.is_anomaly for n in nodes)
        / NODE_COUNT
    )

# =========================================================
# anomaly detection
# =========================================================

def detect_anomaly(nodes):

    trusts = np.array([
        n.trust for n in nodes
    ])

    mean = np.mean(trusts)
    std = np.std(trusts) + 1e-8

    for i, n in enumerate(nodes):

        z = abs(
            (trusts[i] - mean) / std
        )

        n.anomaly_score = z
        n.is_anomaly = z > ANOMALY_Z

# =========================================================
# trust update
# =========================================================

def update_trust(nodes):

    scores = []

    for n in nodes:

        neighbors = [
            nodes[i]
            for i in n.connections
        ]

        if neighbors:

            sims = [
                similarity(
                    n.essences,
                    nb.essences
                )
                for nb in neighbors
            ]

            score = np.mean(sims)

        else:
            score = 0.5

        scores.append(score)

    scores = np.array(scores)

    order = np.argsort(scores)

    ranks = np.empty_like(order)

    ranks[order] = np.arange(len(nodes))

    targets = (
        MIN_TRUST +
        (MAX_TRUST - MIN_TRUST)
        * (
            ranks /
            (NODE_COUNT - 1 + 1e-8)
        )
    )

    for i, n in enumerate(nodes):

        anomaly_penalty = (
            1.0 -
            0.05 *
            (
                n.anomaly_score /
                (1.0 + n.anomaly_score)
            )
        )

        updated = (
            (1 - TRUST_DECAY) * n.trust
            + 0.50 * (targets[i] - n.trust)
        )

        n.trust = float(
            np.clip(
                updated * anomaly_penalty,
                MIN_TRUST,
                MAX_TRUST
            )
        )

# =========================================================
# propagation
# =========================================================

def propagate_essences(nodes):

    for n in nodes:

        neighbors = [
            nodes[i]
            for i in n.connections
        ]

        if not neighbors:
            continue

        for i in range(ESSENCE_SLOTS):

            src_node = random.choice(neighbors)

            src = src_node.essences[
                random.randrange(ESSENCE_SLOTS)
            ]

            sim = np.dot(
                n.essences[i],
                src
            )

            if sim < 0.70:

                n.essences[i] += (
                    UPDATE_SCALE *
                    (src - n.essences[i])
                )

            elif sim > 0.90:

                n.essences[i] -= (
                    UPDATE_SCALE * 0.20 *
                    (src - n.essences[i])
                )

            n.essences[i] /= (
                np.linalg.norm(
                    n.essences[i]
                ) + 1e-8
            )

# =========================================================
# rewiring
# =========================================================

def rewire(nodes):

    previous = {
        n.id: copy.deepcopy(n.connections)
        for n in nodes
    }

    for n in nodes:

        candidates = [
            c for c in nodes
            if c.id != n.id
        ]

        scored = sorted(
            [
                (
                    0.60 * c.trust
                    + 0.40 * similarity(
                        n.essences,
                        c.essences
                    ),
                    c
                )
                for c in candidates
            ],
            reverse=True,
            key=lambda x: x[0]
        )

        selected = []

        for _, c in scored:

            if len(selected) >= SEARCH_K:
                break

            if all(
                similarity(
                    c.essences,
                    s.essences
                ) < 0.84
                for s in selected
            ):
                selected.append(c)

        while len(selected) < SEARCH_K:

            c = random.choice(candidates)

            if c not in selected:
                selected.append(c)

        n.connections = set(
            c.id
            for c in selected[:MAX_CONNECTIONS]
        )

        overlap = len(
            previous[n.id] &
            n.connections
        )

        if overlap >= 2:

            n.persistence_counter += 1

            if n.persistence_counter > 80:
                n.persistence_counter = 80

# =========================================================
# variance regulation
# =========================================================

def regulate_variance(nodes):

    variance = compute_variance(nodes)

    if variance > TARGET_VAR_MAX:

        scale = np.sqrt(
            TARGET_VAR_MAX /
            (variance + 1e-8)
        )

        for n in nodes:
            n.essences *= scale

    elif variance < TARGET_VAR_MIN:

        for n in nodes:

            n.essences += (
                np.random.randn(
                    *n.essences.shape
                ) * 0.01
            )

            n.essences /= (
                np.linalg.norm(
                    n.essences,
                    axis=1,
                    keepdims=True
                ) + 1e-8
            )

# =========================================================
# lineage system
# =========================================================

def update_reuse_lineage_memory(
    lineage_memory,
    signature
):

    matched = False

    for lineage in lineage_memory:

        overlap = topology_overlap(
            signature,
            lineage["anchor"]
        )

        if 0.55 <= overlap <= 0.92:

            lineage["history"].append(signature)

            lineage["continuity"] = (
                0.90 * lineage["continuity"]
                + 0.10 * overlap
            )

            lineage["reuse_density"] += 1

            lineage["last_overlap"] = overlap

            matched = True
            break

    if not matched:

        lineage_memory.append({

            "anchor": signature,

            "history": [signature],

            "continuity": 0.60,

            "reuse_density": 1,

            "last_overlap": 0.60
        })

    if len(lineage_memory) > LINEAGE_MEMORY_LIMIT:
        lineage_memory.pop(0)

# =========================================================
# simulation
# =========================================================

def run_simulation(seed):

    random.seed(seed)
    np.random.seed(seed)

    nodes = [
        Node(i)
        for i in range(NODE_COUNT)
    ]

    reuse_lineage_memory = []

    continuity_hysteresis_memory = 0.65

    stable_model_fixation_events = 0
    adaptive_pattern_reuse_events = 0
    transition_suppression_events = 0
    topology_identity_preservation_events = 0

    for step in range(STEP_COUNT):

        detect_anomaly(nodes)

        update_trust(nodes)

        propagate_essences(nodes)

        rewire(nodes)

        regulate_variance(nodes)

        signature = compute_signature(nodes)

        update_reuse_lineage_memory(
            reuse_lineage_memory,
            signature
        )

        lineage_scores = []

        reuse_density_scores = []

        for lineage in reuse_lineage_memory:

            overlap = topology_overlap(
                signature,
                lineage["anchor"]
            )

            continuity_score = (

                0.50 * overlap
                + 0.30 * lineage["continuity"]
                + 0.20 * min(
                    lineage["reuse_density"] / 10.0,
                    1.0
                )

            )

            lineage_scores.append(
                continuity_score
            )

            reuse_density_scores.append(
                lineage["reuse_density"]
            )

        if lineage_scores:

            strongest_lineage_score = max(
                lineage_scores
            )

            reuse_family_score = np.mean(
                sorted(
                    lineage_scores,
                    reverse=True
                )[:5]
            )

        else:

            strongest_lineage_score = 0.60
            reuse_family_score = 0.60

        topology_continuity_score = max(

            strongest_lineage_score,

            continuity_hysteresis_memory

        )

        continuity_hysteresis_memory = (

            HYSTERESIS_DECAY *
            continuity_hysteresis_memory

            +

            (1.0 - HYSTERESIS_DECAY) *
            topology_continuity_score

        )

        adaptive_reuse_score = (

            0.45 * strongest_lineage_score
            + 0.35 * reuse_family_score
            + 0.20 * continuity_hysteresis_memory

        )

        persistence_score = min(
            compute_persistence(nodes) / 80.0,
            1.0
        )

        anomaly_score = (
            1.0 - compute_anomaly_ratio(nodes)
        )

        resilience_consistency_score = (

            0.40 * persistence_score
            + 0.35 * topology_continuity_score
            + 0.25 * anomaly_score

        )

        self_model_stability_score = (

            0.35 * persistence_score
            + 0.35 * resilience_consistency_score
            + 0.30 * topology_continuity_score

        )

        transition_suppression_score = (

            0.50 * topology_continuity_score
            + 0.30 * continuity_hysteresis_memory
            + 0.20 * persistence_score

        )

        if self_model_stability_score > 0.60:
            stable_model_fixation_events += 1

        if adaptive_reuse_score > 0.60:
            adaptive_pattern_reuse_events += 1

        if transition_suppression_score > 0.60:
            transition_suppression_events += 1

        if topology_continuity_score > 0.60:
            topology_identity_preservation_events += 1

    patterns = {

        tuple(sorted(list(n.connections)))
        for n in nodes
    }

    structural_diversity = (
        len(patterns) / NODE_COUNT
    )

    average_persistence = (
        compute_persistence(nodes)
    )

    mean_variance = (
        compute_variance(nodes)
    )

    trust_range = (
        compute_trust_range(nodes)
    )

    validation_result = all([

        adaptive_pattern_reuse_events >= 5,

        topology_identity_preservation_events >= 5,

        transition_suppression_events >= 5,

        stable_model_fixation_events >= 5,

        structural_diversity >= 0.35,

        10.0 <= average_persistence <= 80.0,

        0.02 <= mean_variance <= 0.06,

        0.15 <= trust_range <= 0.85
    ])

    print(f"\n--- RUN #{seed - BASE_SEED + 1} ---")

    print(
        "stable_model_fixation_events:",
        stable_model_fixation_events
    )

    print(
        "adaptive_pattern_reuse_events:",
        adaptive_pattern_reuse_events
    )

    print(
        "transition_suppression_events:",
        transition_suppression_events
    )

    print(
        "topology_identity_preservation_events:",
        topology_identity_preservation_events
    )

    print(
        "structural_diversity:",
        round(structural_diversity, 6)
    )

    print(
        "average_persistence:",
        round(average_persistence, 6)
    )

    print(
        "mean_variance:",
        round(mean_variance, 6)
    )

    print(
        "trust_range:",
        round(trust_range, 6)
    )

    print(
        "validation_result:",
        validation_result
    )

    return validation_result

# =========================================================
# triple execution
# =========================================================

results = []

for seed in [42, 43, 44]:

    results.append(
        run_simulation(seed)
    )

print("\nfinal_result:")

if all(results):
    print("ACHIEVED")
else:
    print("NOT ACHIEVED")
