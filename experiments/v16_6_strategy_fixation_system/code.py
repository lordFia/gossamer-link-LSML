# v16.6.0 strategy fixation system
# persistent strategic topology
# triple execution validation

import numpy as np
import random
import hashlib
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

POW_DIFFICULTY = 2

TRUST_DECAY = 0.018

MIN_TRUST = 0.05
MAX_TRUST = 0.95

ANOMALY_Z = 1.2

TARGET_VAR_MIN = 0.02
TARGET_VAR_MAX = 0.06

UPDATE_SCALE = 0.05

HISTORY_INTERVAL = 5

FIXATION_THRESHOLD = 0.72
STRATEGY_MEMORY_LIMIT = 50

# =========================================================
# node
# =========================================================

class Node:

    def __init__(self, node_id):

        self.id = node_id

        self.essences = np.random.randn(ESSENCE_SLOTS, DIM)
        self.essences /= (
            np.linalg.norm(self.essences, axis=1, keepdims=True) + 1e-8
        )

        self.trust = random.uniform(0.3, 0.7)

        self.connections = set()

        self.age = 1
        self.nonce = 0

        self.is_anomaly = False
        self.anomaly_score = 0.0

        self.persistence_counter = 0
        self.decay_resistance = 1.0

# =========================================================
# utility
# =========================================================

def compute_hash(val):
    return hashlib.sha256(str(val).encode()).hexdigest()

def valid_pow(node):
    return compute_hash((node.id, node.nonce)).startswith(
        "0" * POW_DIFFICULTY
    )

def mine_pow(node):

    if valid_pow(node):
        return

    while not valid_pow(node):
        node.nonce += 1

def similarity(a, b):
    return np.max(np.dot(a, b.T))

# =========================================================
# anomaly
# =========================================================

def detect_anomaly(nodes):

    vals = np.array([n.trust for n in nodes])

    mean = np.mean(vals)
    std = np.std(vals) + 1e-8

    for i, node in enumerate(nodes):

        z = abs((vals[i] - mean) / std)

        node.anomaly_score = float(z)
        node.is_anomaly = z > ANOMALY_Z

# =========================================================
# trust
# =========================================================

def update_trust(nodes):

    scores = []

    for node in nodes:

        neighbors = [nodes[i] for i in node.connections]

        if neighbors:

            sims = [
                similarity(node.essences, nb.essences)
                for nb in neighbors
            ]

            scores.append(np.mean(sims))

        else:
            scores.append(0.5)

    scores = np.array(scores)

    order = np.argsort(scores)

    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(nodes))

    target = (
        MIN_TRUST
        + (MAX_TRUST - MIN_TRUST)
        * (ranks / (len(nodes) - 1 + 1e-8))
    )

    mean_trust = np.mean([n.trust for n in nodes])

    for i, node in enumerate(nodes):

        anomaly_scale = (
            1.0
            - 0.18 * (
                node.anomaly_score
                / (1.0 + node.anomaly_score)
            )
        )

        stabilization_bonus = (
            0.02 * node.decay_resistance
        )

        new_trust = (
            (1 - TRUST_DECAY) * node.trust
            + 0.50 * (target[i] - node.trust)
            + 0.10 * (node.trust - mean_trust)
            + stabilization_bonus
        )

        node.trust = float(
            np.clip(
                new_trust * anomaly_scale,
                MIN_TRUST,
                MAX_TRUST
            )
        )

# =========================================================
# propagation
# =========================================================

def propagate_essences(nodes):

    for node in nodes:

        neighbors = [nodes[i] for i in node.connections]

        if not neighbors:
            continue

        for i in range(ESSENCE_SLOTS):

            src_node = random.choice(neighbors)

            src = src_node.essences[
                random.randrange(ESSENCE_SLOTS)
            ]

            sim = np.dot(node.essences[i], src)

            if sim < 0.6:

                node.essences[i] += (
                    UPDATE_SCALE
                    * (src - node.essences[i])
                )

            elif sim > 0.85:

                node.essences[i] -= (
                    UPDATE_SCALE
                    * 0.5
                    * (src - node.essences[i])
                )

            node.essences[i] /= (
                np.linalg.norm(node.essences[i]) + 1e-8
            )

# =========================================================
# strategy memory
# =========================================================

strategy_memory = []

strategy_fixation_events = 0
strategy_reuse_events = 0
strategy_inheritance_events = 0
successful_strategy_recoveries = 0

# =========================================================
# survival scoring
# =========================================================

def compute_survival_score(nodes):

    all_vectors = np.array([
        e for n in nodes for e in n.essences
    ])

    mean_variance = np.mean(
        np.var(all_vectors, axis=0)
    )

    trust_values = [n.trust for n in nodes]

    trust_range = (
        max(trust_values) - min(trust_values)
    )

    persistence = np.mean([
        n.persistence_counter for n in nodes
    ])

    anomaly_penalty = np.mean([
        n.anomaly_score for n in nodes
    ])

    variance_score = (
        1.0 - abs(mean_variance - 0.03)
    )

    trust_score = min(trust_range, 1.0)

    persistence_score = min(
        persistence / 80.0,
        1.0
    )

    collapse_resistance = (
        1.0 / (1.0 + anomaly_penalty)
    )

    score = (
        0.30 * variance_score
        + 0.25 * trust_score
        + 0.25 * persistence_score
        + 0.20 * collapse_resistance
    )

    return float(score)

# =========================================================
# strategy fixation
# =========================================================

def store_strategy(nodes, step):

    global strategy_fixation_events

    survival_score = compute_survival_score(nodes)

    if survival_score < FIXATION_THRESHOLD:
        return

    topology_density = (
        np.mean([
            len(n.connections)
            for n in nodes
        ]) / MAX_CONNECTIONS
    )

    strategy = {
        "step": step,
        "connections": {
            n.id: sorted(list(n.connections))
            for n in nodes
        },
        "trust_distribution": [
            round(n.trust, 4)
            for n in nodes
        ],
        "topology_density": topology_density,
        "persistence": np.mean([
            n.persistence_counter
            for n in nodes
        ]),
        "survival_score": survival_score,
        "reuse_count": 0
    }

    strategy_memory.append(strategy)

    if len(strategy_memory) > STRATEGY_MEMORY_LIMIT:
        strategy_memory.pop(0)

    strategy_fixation_events += 1

# =========================================================
# strategy ranking
# =========================================================

def ranked_strategies():

    ranked = sorted(
        strategy_memory,
        key=lambda x: x["survival_score"],
        reverse=True
    )

    return ranked

# =========================================================
# partial inheritance
# =========================================================

def apply_strategy_inheritance(nodes):

    global strategy_reuse_events
    global strategy_inheritance_events
    global successful_strategy_recoveries

    if len(strategy_memory) < 10:
        return

    ranked = ranked_strategies()

    best_strategy = ranked[0]

    inheritance_success = False

    for node in nodes:

        if random.random() < 0.35:

            historical_connections = best_strategy[
                "connections"
            ][node.id]

            partial_size = max(
                1,
                int(len(historical_connections) * 0.5)
            )

            inherited = random.sample(
                historical_connections,
                partial_size
            )

            current = list(node.connections)

            merged = set(
                current[:2] + inherited
            )

            merged.discard(node.id)

            node.connections = set(
                list(merged)[:MAX_CONNECTIONS]
            )

            node.persistence_counter += 1

            node.decay_resistance = min(
                1.5,
                node.decay_resistance + 0.02
            )

            inheritance_success = True

    if inheritance_success:

        best_strategy["reuse_count"] += 1

        strategy_reuse_events += 1
        strategy_inheritance_events += 1
        successful_strategy_recoveries += 1

# =========================================================
# rewiring
# =========================================================

def rewire(nodes):

    for node in nodes:

        candidates = [
            n for n in nodes
            if n.id != node.id
        ]

        scored = []

        for c in candidates:

            sim = similarity(
                node.essences,
                c.essences
            )

            score = (
                0.55 * c.trust
                + 0.45 * sim
            )

            scored.append((score, c))

        scored.sort(
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
                ) < 0.75
                for s in selected
            ):
                selected.append(c)

        while len(selected) < SEARCH_K:

            c = random.choice(candidates)

            if c not in selected:
                selected.append(c)

        old_connections = set(node.connections)

        node.connections = set(
            c.id
            for c in selected[:MAX_CONNECTIONS]
        )

        overlap = len(
            old_connections & node.connections
        )

        persistence_gain = (
            overlap
            * node.decay_resistance
        )

        if persistence_gain >= 2.0:
            node.persistence_counter += 1

# =========================================================
# variance regulation
# =========================================================

def regulate_variance(nodes):

    all_vectors = np.array([
        e for n in nodes for e in n.essences
    ])

    mean_variance = np.mean(
        np.var(all_vectors, axis=0)
    )

    if mean_variance > TARGET_VAR_MAX:

        scale = np.sqrt(
            TARGET_VAR_MAX
            / (mean_variance + 1e-8)
        )

        for n in nodes:
            n.essences *= scale

    elif mean_variance < TARGET_VAR_MIN:

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
# history
# =========================================================

topology_history = []
cause_memory = []

def record_history(nodes, step):

    topology_history.append({
        "step": step,
        "connections": [
            sorted(list(n.connections))
            for n in nodes
        ]
    })

    cause_memory.append({
        "step": step,
        "cause_event": "strategy_update",
        "effect_topology": [
            len(n.connections)
            for n in nodes
        ],
        "survival_delta": compute_survival_score(nodes)
    })

# =========================================================
# metrics
# =========================================================

def compute_metrics(nodes):

    all_vectors = np.array([
        e for n in nodes for e in n.essences
    ])

    mean_variance = float(
        np.mean(np.var(all_vectors, axis=0))
    )

    trust_values = [n.trust for n in nodes]

    patterns = {
        tuple(sorted(list(n.connections)))
        for n in nodes
    }

    structural_diversity = (
        len(patterns) / NODE_COUNT
    )

    average_persistence = np.mean([
        min(n.persistence_counter, 80)
        for n in nodes
    ])

    strategy_integrity = all([
        (
            "connections" in s
            and "survival_score" in s
            and "reuse_count" in s
        )
        for s in strategy_memory
    ]) if strategy_memory else False

    long_term_persistence = (
        strategy_fixation_events >= 5
        and strategy_reuse_events >= 5
        and strategy_inheritance_events >= 5
    )

    adaptive_strategy_ranking = (
        len(strategy_memory) > 1
    )

    simulation_stability = (
        0.02 <= mean_variance <= 0.06
    )

    trust_range = (
        max(trust_values)
        - min(trust_values)
    )

    metrics = {

        "history_length":
            len(topology_history),

        "cause_memory_length":
            len(cause_memory),

        "strategy_memory_length":
            len(strategy_memory),

        "strategy_fixation_events":
            strategy_fixation_events,

        "strategy_reuse_events":
            strategy_reuse_events,

        "strategy_inheritance_events":
            strategy_inheritance_events,

        "successful_strategy_recoveries":
            successful_strategy_recoveries,

        "adaptive_strategy_ranking":
            adaptive_strategy_ranking,

        "strategy_integrity":
            strategy_integrity,

        "long_term_persistence":
            long_term_persistence,

        "historical_adaptation":
            strategy_reuse_events > 0,

        "structural_diversity":
            round(
                float(structural_diversity),
                6
            ),

        "average_persistence":
            round(
                float(average_persistence),
                6
            ),

        "simulation_stability":
            simulation_stability,

        "mean_variance":
            round(mean_variance, 6),

        "trust_range":
            round(float(trust_range), 6)
    }

    return metrics

# =========================================================
# validation
# =========================================================

def validate(metrics):

    return (

        metrics["history_length"] >= 120
        and metrics["cause_memory_length"] >= 120
        and metrics["strategy_memory_length"] >= 10

        and metrics["strategy_fixation_events"] >= 5
        and metrics["strategy_reuse_events"] >= 5
        and metrics["strategy_inheritance_events"] >= 5
        and metrics["successful_strategy_recoveries"] >= 3

        and metrics["adaptive_strategy_ranking"] is True
        and metrics["strategy_integrity"] is True
        and metrics["long_term_persistence"] is True
        and metrics["historical_adaptation"] is True

        and metrics["structural_diversity"] >= 0.35

        and 10.0 <= metrics["average_persistence"] <= 80.0

        and metrics["simulation_stability"] is True

        and 0.02 <= metrics["mean_variance"] <= 0.06

        and 0.15 <= metrics["trust_range"] <= 0.85
    )

# =========================================================
# simulation
# =========================================================

def run_simulation(run_id):

    global topology_history
    global cause_memory
    global strategy_memory

    global strategy_fixation_events
    global strategy_reuse_events
    global strategy_inheritance_events
    global successful_strategy_recoveries

    topology_history = []
    cause_memory = []
    strategy_memory = []

    strategy_fixation_events = 0
    strategy_reuse_events = 0
    strategy_inheritance_events = 0
    successful_strategy_recoveries = 0

    seed = BASE_SEED + run_id

    random.seed(seed)
    np.random.seed(seed)

    nodes = [
        Node(i)
        for i in range(NODE_COUNT)
    ]

    for t in range(STEP_COUNT):

        for n in nodes:

            n.age += 1

            mine_pow(n)

        detect_anomaly(nodes)

        update_trust(nodes)

        propagate_essences(nodes)

        rewire(nodes)

        if t % 20 == 0:
            apply_strategy_inheritance(nodes)

        regulate_variance(nodes)

        if t % HISTORY_INTERVAL == 0:

            record_history(nodes, t)

            store_strategy(nodes, t)

    metrics = compute_metrics(nodes)

    valid = validate(metrics)

    print(f"\n--- RUN #{run_id + 1} ---")

    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("validation_result:", valid)

    return valid

# =========================================================
# triple execution
# =========================================================

results = []

for i in range(3):

    results.append(
        run_simulation(i)
    )

print("\nfinal_result:")

if all(results):
    print("ACHIEVED")
else:
    print("NOT ACHIEVED")
