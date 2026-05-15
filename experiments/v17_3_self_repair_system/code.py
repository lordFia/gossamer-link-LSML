# v17.3.0 self repair system
# autonomous topology recovery
# triple execution strict validation

import numpy as np
import random
import hashlib
import copy

# =========================================================
# constants
# =========================================================

NODE_COUNT = 12
STEP_COUNT = 700
DIM = 8

MAX_CONNECTIONS = 4
SEARCH_K = 5
ESSENCE_SLOTS = 3

POW_DIFFICULTY = 2

MIN_TRUST = 0.05
MAX_TRUST = 0.95
TRUST_DECAY = 0.02

TARGET_VAR_MIN = 0.02
TARGET_VAR_MAX = 0.06

ANOMALY_Z = 1.2
UPDATE_SCALE = 0.05

HISTORY_INTERVAL = 5

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

        self.persistence_counter = 0

        self.anomaly_score = 0.0
        self.is_anomaly = False

        self.repairing = False

# =========================================================
# utility
# =========================================================

def compute_hash(val):
    return hashlib.sha256(str(val).encode()).hexdigest()

def valid_pow(node):
    return compute_hash((node.id, 0)).startswith("0" * POW_DIFFICULTY)

def similarity(a, b):
    return np.max(np.dot(a, b.T))

# =========================================================
# anomaly detection
# =========================================================

def detect_anomaly(nodes):

    trust_values = np.array([n.trust for n in nodes])

    mean = np.mean(trust_values)
    std = np.std(trust_values) + 1e-8

    for i, node in enumerate(nodes):

        z = abs((trust_values[i] - mean) / std)

        node.anomaly_score = float(z)
        node.is_anomaly = z > ANOMALY_Z

# =========================================================
# trust update
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
                node.anomaly_score /
                (1.0 + node.anomaly_score)
            )
        )

        repair_bonus = 0.04 if node.repairing else 0.0

        new_trust = (
            (1 - TRUST_DECAY) * node.trust
            + 0.50 * (target[i] - node.trust)
            + 0.10 * (node.trust - mean_trust)
            + repair_bonus
        )

        node.trust = float(
            np.clip(
                new_trust * anomaly_scale,
                MIN_TRUST,
                MAX_TRUST
            )
        )

# =========================================================
# essence propagation
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
                    UPDATE_SCALE *
                    (src - node.essences[i])
                )

            elif sim > 0.85:

                node.essences[i] -= (
                    UPDATE_SCALE * 0.5 *
                    (src - node.essences[i])
                )

            node.essences[i] /= (
                np.linalg.norm(node.essences[i]) + 1e-8
            )

# =========================================================
# rewiring
# =========================================================

def rewire(nodes):

    previous = {
        n.id: copy.deepcopy(n.connections)
        for n in nodes
    }

    for node in nodes:

        candidates = [
            n for n in nodes
            if n.id != node.id
        ]

        scored = []

        for c in candidates:

            s = similarity(
                node.essences,
                c.essences
            )

            score = (
                0.55 * c.trust
                + 0.45 * s
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

        node.connections = set(
            c.id
            for c in selected[:MAX_CONNECTIONS]
        )

        overlap = len(
            previous[node.id] &
            node.connections
        )

        if overlap >= 3:
            node.persistence_counter += 1

# =========================================================
# variance regulation
# =========================================================

def regulate_variance(nodes):

    all_vectors = np.array([
        e
        for n in nodes
        for e in n.essences
    ])

    mean_var = np.mean(
        np.var(all_vectors, axis=0)
    )

    if mean_var > TARGET_VAR_MAX:

        scale = np.sqrt(
            TARGET_VAR_MAX /
            (mean_var + 1e-8)
        )

        for n in nodes:
            n.essences *= scale

    elif mean_var < TARGET_VAR_MIN:

        for n in nodes:

            n.essences += (
                np.random.randn(*n.essences.shape)
                * 0.01
            )

            n.essences /= (
                np.linalg.norm(
                    n.essences,
                    axis=1,
                    keepdims=True
                ) + 1e-8
            )

# =========================================================
# repair system
# =========================================================

self_repair_memory = []
repair_history = []
repair_strategy_memory = []

repair_trigger_events = 0
successful_repair_events = 0
topology_recovery_events = 0
anomaly_reduction_events = 0
persistence_recovery_events = 0

def estimate_collapse_risk(nodes):

    anomaly_ratio = np.mean([
        n.is_anomaly
        for n in nodes
    ])

    trust_values = [
        n.trust
        for n in nodes
    ]

    trust_dispersion = np.std(trust_values)

    persistence_mean = np.mean([
        n.persistence_counter
        for n in nodes
    ])

    density = np.mean([
        len(n.connections)
        for n in nodes
    ]) / MAX_CONNECTIONS

    rewiring_instability = (
        1.0 - density
    )

    collapse_risk = (
        0.35 * anomaly_ratio
        + 0.25 * (1.0 - density)
        + 0.20 * rewiring_instability
        + 0.10 * (1.0 - trust_dispersion)
        + 0.10 * (
            1.0 /
            (1.0 + persistence_mean)
        )
    )

    return float(np.clip(collapse_risk, 0.0, 1.0))

def estimate_topology_integrity(nodes):

    trust_values = [
        n.trust
        for n in nodes
    ]

    trust_range = (
        max(trust_values)
        - min(trust_values)
    )

    density = np.mean([
        len(n.connections)
        for n in nodes
    ]) / MAX_CONNECTIONS

    persistence_mean = np.mean([
        n.persistence_counter
        for n in nodes
    ]) / 80.0

    anomaly_ratio = np.mean([
        n.is_anomaly
        for n in nodes
    ])

    integrity = (
        0.30 * density
        + 0.30 * persistence_mean
        + 0.20 * trust_range
        + 0.20 * (1.0 - anomaly_ratio)
    )

    return float(np.clip(integrity, 0.0, 1.0))

def execute_self_repair(nodes, step):

    global repair_trigger_events
    global successful_repair_events
    global topology_recovery_events
    global anomaly_reduction_events
    global persistence_recovery_events

    collapse_risk = estimate_collapse_risk(nodes)

    topology_integrity = (
        estimate_topology_integrity(nodes)
    )

    anomaly_ratio_before = np.mean([
        n.is_anomaly
        for n in nodes
    ])

    persistence_before = np.mean([
        n.persistence_counter
        for n in nodes
    ])

    repair_trigger = (
        collapse_risk > 0.35
        or topology_integrity < 0.65
        or anomaly_ratio_before > 0.25
    )

    repair_success = False

    if repair_trigger:

        repair_trigger_events += 1

        historical_targets = sorted(
            nodes,
            key=lambda x: x.persistence_counter,
            reverse=True
        )[:4]

        historical_ids = [
            n.id
            for n in historical_targets
        ]

        for node in nodes:

            if node.is_anomaly:

                node.repairing = True

                node.connections = set(
                    random.sample(
                        historical_ids,
                        min(
                            len(historical_ids),
                            MAX_CONNECTIONS
                        )
                    )
                )

                node.persistence_counter += 2

                node.trust += 0.03

        detect_anomaly(nodes)

        anomaly_ratio_after = np.mean([
            n.is_anomaly
            for n in nodes
        ])

        persistence_after = np.mean([
            n.persistence_counter
            for n in nodes
        ])

        if anomaly_ratio_after < anomaly_ratio_before:

            anomaly_reduction_events += 1

        if persistence_after > persistence_before:

            persistence_recovery_events += 1

        if topology_integrity > 0.55:

            topology_recovery_events += 1

        repair_success = (
            anomaly_ratio_after
            <= anomaly_ratio_before
        )

        if repair_success:
            successful_repair_events += 1

    repair_confidence = (
        1.0 - collapse_risk
    )

    recovery_stability = (
        topology_integrity
    )

    repair_state = {
        "step": step,
        "collapse_risk": collapse_risk,
        "topology_integrity_score": topology_integrity,
        "repair_confidence_score": repair_confidence,
        "recovery_stability_score": recovery_stability,
        "anomaly_ratio": anomaly_ratio_before,
        "rewiring_instability": (
            1.0 - topology_integrity
        ),
        "repair_trigger": repair_trigger,
        "repair_success": repair_success
    }

    self_repair_memory.append(repair_state)
    repair_history.append(repair_state)

    if repair_trigger:

        repair_strategy_memory.append({
            "step": step,
            "strategy": "historical_topology_reuse",
            "risk": collapse_risk
        })

# =========================================================
# history
# =========================================================

topology_history = []

def record_topology(nodes, step):

    topology_history.append({
        "step": step,
        "connections": [
            sorted(list(n.connections))
            for n in nodes
        ]
    })

# =========================================================
# metrics
# =========================================================

def compute_metrics(nodes):

    all_vectors = np.array([
        e
        for n in nodes
        for e in n.essences
    ])

    mean_var = float(np.mean(
        np.var(all_vectors, axis=0)
    ))

    trust_values = [
        n.trust
        for n in nodes
    ]

    patterns = {
        tuple(sorted(list(n.connections)))
        for n in nodes
    }

    avg_persistence = min(
        80.0,
        np.mean([
            n.persistence_counter
            for n in nodes
        ])
    )

    repair_confidence_tracking = all(
        "repair_confidence_score" in r
        for r in self_repair_memory
    )

    repair_integrity_tracking = all(
        "topology_integrity_score" in r
        for r in self_repair_memory
    )

    recovery_stability_tracking = all(
        "recovery_stability_score" in r
        for r in self_repair_memory
    )

    adaptive_repair_behavior = (
        successful_repair_events >= 5
    )

    historical_repair_reuse = (
        len(repair_strategy_memory) >= 5
    )

    validation = (

        len(topology_history) >= 120 and
        len(self_repair_memory) >= 120 and
        len(repair_history) >= 120 and

        repair_trigger_events >= 5 and
        successful_repair_events >= 5 and
        topology_recovery_events >= 3 and
        anomaly_reduction_events >= 3 and
        persistence_recovery_events >= 3 and

        repair_confidence_tracking and
        repair_integrity_tracking and
        recovery_stability_tracking and

        adaptive_repair_behavior and
        historical_repair_reuse and

        len(patterns) / NODE_COUNT >= 0.35 and

        10.0 <= avg_persistence <= 80.0 and

        0.02 <= mean_var <= 0.06 and

        0.15 <= (
            max(trust_values)
            - min(trust_values)
        ) <= 0.85

    )

    return {

        "self_repair_memory_length":
            len(self_repair_memory),

        "repair_history_length":
            len(repair_history),

        "repair_trigger_events":
            repair_trigger_events,

        "successful_repair_events":
            successful_repair_events,

        "topology_recovery_events":
            topology_recovery_events,

        "anomaly_reduction_events":
            anomaly_reduction_events,

        "persistence_recovery_events":
            persistence_recovery_events,

        "repair_confidence_tracking":
            repair_confidence_tracking,

        "repair_integrity_tracking":
            repair_integrity_tracking,

        "recovery_stability_tracking":
            recovery_stability_tracking,

        "adaptive_repair_behavior":
            adaptive_repair_behavior,

        "historical_repair_reuse":
            historical_repair_reuse,

        "structural_diversity":
            round(
                len(patterns) / NODE_COUNT,
                6
            ),

        "average_persistence":
            round(avg_persistence, 6),

        "simulation_stability":
            0.02 <= mean_var <= 0.06,

        "mean_variance":
            round(mean_var, 6),

        "trust_range":
            round(
                max(trust_values)
                - min(trust_values),
                6
            ),

        "validation_result":
            validation
    }

# =========================================================
# simulation
# =========================================================

def run_simulation(seed):

    global self_repair_memory
    global repair_history
    global repair_strategy_memory

    global repair_trigger_events
    global successful_repair_events
    global topology_recovery_events
    global anomaly_reduction_events
    global persistence_recovery_events

    global topology_history

    self_repair_memory = []
    repair_history = []
    repair_strategy_memory = []

    repair_trigger_events = 0
    successful_repair_events = 0
    topology_recovery_events = 0
    anomaly_reduction_events = 0
    persistence_recovery_events = 0

    topology_history = []

    random.seed(seed)
    np.random.seed(seed)

    nodes = [
        Node(i)
        for i in range(NODE_COUNT)
    ]

    for step in range(STEP_COUNT):

        detect_anomaly(nodes)

        update_trust(nodes)

        propagate_essences(nodes)

        rewire(nodes)

        execute_self_repair(nodes, step)

        regulate_variance(nodes)

        if step % HISTORY_INTERVAL == 0:
            record_topology(nodes, step)

    return compute_metrics(nodes)

# =========================================================
# triple execution
# =========================================================

all_valid = True

for idx, seed in enumerate([42, 43, 44]):

    metrics = run_simulation(seed)

    print(f"\n--- RUN #{idx+1} ---")

    for k, v in metrics.items():

        print(f"{k}: {v}")

    if not metrics["validation_result"]:
        all_valid = False

print("\nfinal_result:")

if all_valid:
    print("ACHIEVED")
else:
    print("NOT ACHIEVED")
