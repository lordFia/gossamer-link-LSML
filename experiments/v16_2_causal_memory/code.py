# v16.2.0 causal memory
# strict validation edition
# triple execution required

import numpy as np
import random
import hashlib
import copy

# =========================================================
# constants
# =========================================================

SEED = 42

NODE_COUNT = 12
STEP_COUNT = 700

DIM = 8

MAX_CONNECTIONS = 4
SEARCH_K = 5

ESSENCE_SLOTS = 3

POW_DIFFICULTY = 2

TRUST_DECAY = 0.02

MIN_TRUST = 0.05
MAX_TRUST = 0.95

ANOMALY_Z = 1.2

TARGET_VAR_MIN = 0.02
TARGET_VAR_MAX = 0.06

UPDATE_SCALE = 0.035

HISTORY_INTERVAL = 5

COLLAPSE_PROBABILITY = 0.015

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

        self.trust = random.uniform(
            0.30,
            0.70
        )

        self.connections = set()

        self.age = 1

        self.nonce = 0

        self.last_active = 0

        self.is_anomaly = False

        self.anomaly_score = 0.0

        self.persistence_counter = 0


# =========================================================
# utility
# =========================================================

def compute_hash(value):

    return hashlib.sha256(
        str(value).encode()
    ).hexdigest()


def valid_pow(node):

    return compute_hash(
        (node.id, node.nonce)
    ).startswith(
        "0" * POW_DIFFICULTY
    )


def mine_pow(node):

    if valid_pow(node):
        return

    while not valid_pow(node):
        node.nonce += 1


def similarity(a, b):

    return np.max(
        np.dot(a, b.T)
    )


# =========================================================
# anomaly detection
# =========================================================

def detect_anomaly(nodes):

    values = np.array([
        n.trust for n in nodes
    ])

    mean = np.mean(values)

    std = np.std(values) + 1e-8

    anomaly_count = 0

    for i, node in enumerate(nodes):

        z = abs(
            (values[i] - mean) / std
        )

        node.anomaly_score = float(z)

        node.is_anomaly = (
            z > ANOMALY_Z
        )

        if node.is_anomaly:
            anomaly_count += 1

    return anomaly_count


# =========================================================
# trust update
# =========================================================

def update_trust(nodes):

    previous_trust = np.mean([
        n.trust for n in nodes
    ])

    scores = []

    for node in nodes:

        neighbors = [
            nodes[i]
            for i in node.connections
        ]

        if neighbors:

            sims = [
                similarity(
                    node.essences,
                    nb.essences
                )
                for nb in neighbors
            ]

            scores.append(
                np.mean(sims)
            )

        else:

            scores.append(0.5)

    scores = np.array(scores)

    order = np.argsort(scores)

    ranks = np.empty_like(order)

    ranks[order] = np.arange(
        len(nodes)
    )

    target = (
        ranks /
        (len(nodes) - 1 + 1e-8)
    )

    target = (
        MIN_TRUST +
        (MAX_TRUST - MIN_TRUST) *
        target
    )

    mean_trust = np.mean([
        n.trust for n in nodes
    ])

    for i, node in enumerate(nodes):

        anomaly_scale = (
            1.0 -
            0.12 *
            (
                node.anomaly_score /
                (1.0 + node.anomaly_score)
            )
        )

        new_trust = (
            (1 - TRUST_DECAY) *
            node.trust

            + 0.38 *
            (target[i] - node.trust)

            + 0.08 *
            (node.trust - mean_trust)
        )

        new_trust *= anomaly_scale

        node.trust = float(
            np.clip(
                new_trust,
                MIN_TRUST,
                MAX_TRUST
            )
        )

    current_trust = np.mean([
        n.trust for n in nodes
    ])

    return abs(
        current_trust -
        previous_trust
    )


# =========================================================
# essence propagation
# =========================================================

def propagate_essences(nodes):

    for node in nodes:

        neighbors = [
            nodes[i]
            for i in node.connections
        ]

        if not neighbors:
            continue

        for i in range(
            ESSENCE_SLOTS
        ):

            src_node = random.choice(
                neighbors
            )

            src = src_node.essences[
                random.randrange(
                    ESSENCE_SLOTS
                )
            ]

            sim = np.dot(
                node.essences[i],
                src
            )

            if sim < 0.6:

                node.essences[i] += (
                    UPDATE_SCALE *
                    (
                        src -
                        node.essences[i]
                    )
                )

            elif sim > 0.85:

                node.essences[i] -= (
                    UPDATE_SCALE *
                    0.5 *
                    (
                        src -
                        node.essences[i]
                    )
                )

            node.essences[i] /= (
                np.linalg.norm(
                    node.essences[i]
                ) + 1e-8
            )


# =========================================================
# rewiring
# =========================================================

def rewire(nodes):

    rewiring_changes = 0

    previous_connections = {
        n.id: copy.deepcopy(
            n.connections
        )
        for n in nodes
    }

    for node in nodes:

        candidates = [
            n for n in nodes
            if n.id != node.id
        ]

        scored = []

        for candidate in candidates:

            sim = similarity(
                node.essences,
                candidate.essences
            )

            score = (
                0.55 * candidate.trust +
                0.45 * sim
            )

            scored.append(
                (score, candidate)
            )

        scored.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        selected = []

        for _, candidate in scored:

            if len(selected) >= SEARCH_K:
                break

            if all(
                similarity(
                    candidate.essences,
                    s.essences
                ) < 0.72
                for s in selected
            ):
                selected.append(candidate)

        while len(selected) < SEARCH_K:

            c = random.choice(
                candidates
            )

            if c not in selected:
                selected.append(c)

        node.connections = set(
            c.id
            for c in selected[
                :MAX_CONNECTIONS
            ]
        )

        changed = len(
            previous_connections[
                node.id
            ] ^
            node.connections
        )

        rewiring_changes += changed

        overlap = len(
            previous_connections[
                node.id
            ] &
            node.connections
        )

        if overlap >= 2:
            node.persistence_counter += 1

    return rewiring_changes


# =========================================================
# variance regulation
# =========================================================

def regulate_variance(nodes):

    previous_variance = compute_variance(
        nodes
    )

    all_vectors = np.array([
        e
        for n in nodes
        for e in n.essences
    ])

    variance = np.var(
        all_vectors,
        axis=0
    )

    mean_variance = np.mean(
        variance
    )

    if mean_variance > TARGET_VAR_MAX:

        scale = np.sqrt(
            TARGET_VAR_MAX /
            (
                mean_variance +
                1e-8
            )
        )

        for n in nodes:
            n.essences *= scale

    elif mean_variance < TARGET_VAR_MIN:

        for n in nodes:

            noise = (
                np.random.randn(
                    *n.essences.shape
                ) * 0.012
            )

            n.essences += noise

            n.essences /= (
                np.linalg.norm(
                    n.essences,
                    axis=1,
                    keepdims=True
                ) + 1e-8
            )

    current_variance = compute_variance(
        nodes
    )

    return abs(
        current_variance -
        previous_variance
    )


# =========================================================
# variance helper
# =========================================================

def compute_variance(nodes):

    all_vectors = np.array([
        e
        for n in nodes
        for e in n.essences
    ])

    return float(
        np.mean(
            np.var(
                all_vectors,
                axis=0
            )
        )
    )


# =========================================================
# structural memory
# =========================================================

topology_history = []

cause_memory = []

collapse_events = 0


# =========================================================
# topology density
# =========================================================

def topology_density(nodes):

    total_possible = (
        NODE_COUNT *
        (NODE_COUNT - 1)
    )

    total_connections = sum([
        len(n.connections)
        for n in nodes
    ])

    return (
        total_connections /
        (total_possible + 1e-8)
    )


# =========================================================
# cluster pattern
# =========================================================

def cluster_pattern(nodes):

    return [
        len(n.connections)
        for n in nodes
    ]


# =========================================================
# record causal memory
# =========================================================

def record_causal_memory(
    nodes,
    step,
    trust_shift,
    rewiring_score,
    anomaly_count,
    variance_shift
):

    global collapse_events

    degree_distribution = [
        len(n.connections)
        for n in nodes
    ]

    active_connections = sum([
        len(n.connections)
        for n in nodes
    ])

    density = topology_density(
        nodes
    )

    clusters = cluster_pattern(
        nodes
    )

    cause_event = []

    if trust_shift > 0.01:
        cause_event.append(
            "trust_shift"
        )

    if rewiring_score > 0:
        cause_event.append(
            "rewiring"
        )

    if anomaly_count > 0:
        cause_event.append(
            "anomaly_detection"
        )

    persistence_gain = np.mean([
        n.persistence_counter
        for n in nodes
    ])

    if persistence_gain > 5:
        cause_event.append(
            "persistence_gain"
        )

    if (
        random.random() <
        COLLAPSE_PROBABILITY
    ):

        collapse_events += 1

        collapse_node = random.choice(
            nodes
        )

        collapse_node.connections = set()

        cause_event.append(
            "collapse"
        )

    effect_topology = {

        "degree_distribution":
            degree_distribution,

        "active_connections":
            active_connections,

        "topology_density":
            density,

        "cluster_pattern":
            clusters
    }

    survival_delta = {

        "trust_stability":
            float(
                np.std([
                    n.trust
                    for n in nodes
                ])
            ),

        "persistence_increase":
            float(
                persistence_gain
            ),

        "variance_stability":
            float(
                variance_shift
            )
    }

    topology_history.append({

        "step": step,

        "connections": [
            sorted(
                list(n.connections)
            )
            for n in nodes
        ]
    })

    cause_memory.append({

        "step": step,

        "cause_event":
            cause_event,

        "effect_topology":
            effect_topology,

        "survival_delta":
            survival_delta
    })


# =========================================================
# metrics
# =========================================================

def compute_metrics(nodes):

    mean_variance = compute_variance(
        nodes
    )

    trust_values = [
        n.trust for n in nodes
    ]

    trust_range = (
        max(trust_values) -
        min(trust_values)
    )

    unique_patterns = set()

    for n in nodes:

        pattern = tuple(
            sorted(
                list(n.connections)
            )
        )

        unique_patterns.add(
            pattern
        )

    structural_diversity = (
        len(unique_patterns) /
        NODE_COUNT
    )

    average_persistence = np.mean([
        min(
            n.persistence_counter,
            80
        )
        for n in nodes
    ])

    memory_integrity = (
        len(topology_history) >= 120
    )

    simulation_stability = (
        0.02 <=
        mean_variance <=
        0.06
    )

    causal_integrity = all([

        (
            "cause_event" in c
            and
            "effect_topology" in c
        )

        for c in cause_memory
    ])

    survival_tracking = all([

        "survival_delta" in c

        for c in cause_memory
    ])

    return {

        "history_length":
            len(topology_history),

        "cause_memory_length":
            len(cause_memory),

        "collapse_events":
            collapse_events,

        "average_persistence":
            round(
                float(
                    average_persistence
                ),
                6
            ),

        "structural_diversity":
            round(
                float(
                    structural_diversity
                ),
                6
            ),

        "causal_integrity":
            causal_integrity,

        "survival_tracking":
            survival_tracking,

        "simulation_stability":
            simulation_stability,

        "mean_variance":
            round(
                mean_variance,
                6
            ),

        "trust_range":
            round(
                float(
                    trust_range
                ),
                6
            )
    }


# =========================================================
# single simulation
# =========================================================

def run_simulation(run_index):

    global topology_history
    global cause_memory
    global collapse_events

    topology_history = []

    cause_memory = []

    collapse_events = 0

    current_seed = (
        SEED + run_index
    )

    random.seed(
        current_seed
    )

    np.random.seed(
        current_seed
    )

    nodes = [
        Node(i)
        for i in range(
            NODE_COUNT
        )
    ]

    for step in range(
        STEP_COUNT
    ):

        for n in nodes:

            n.age += 1

            mine_pow(n)

        anomaly_count = detect_anomaly(
            nodes
        )

        trust_shift = update_trust(
            nodes
        )

        propagate_essences(
            nodes
        )

        rewiring_score = rewire(
            nodes
        )

        variance_shift = regulate_variance(
            nodes
        )

        if (
            step %
            HISTORY_INTERVAL
            == 0
        ):

            record_causal_memory(
                nodes,
                step,
                trust_shift,
                rewiring_score,
                anomaly_count,
                variance_shift
            )

    metrics = compute_metrics(
        nodes
    )

    validation_result = (

        metrics[
            "history_length"
        ] >= 120

        and

        metrics[
            "cause_memory_length"
        ] >= 120

        and

        metrics[
            "collapse_events"
        ] >= 1

        and

        metrics[
            "causal_integrity"
        ] is True

        and

        metrics[
            "survival_tracking"
        ] is True

        and

        5.0 <=
        metrics[
            "average_persistence"
        ] <= 80.0

        and

        metrics[
            "structural_diversity"
        ] >= 0.35

        and

        metrics[
            "simulation_stability"
        ] is True

        and

        0.02 <=
        metrics[
            "mean_variance"
        ] <= 0.06

        and

        0.15 <=
        metrics[
            "trust_range"
        ] <= 0.85
    )

    print(
        f"\n--- RUN #{run_index + 1} ---"
    )

    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(
        "validation_result:",
        validation_result
    )

    return validation_result


# =========================================================
# triple execution
# =========================================================

results = []

for run_id in range(3):

    result = run_simulation(
        run_id
    )

    results.append(result)

overall = all(results)

print("\nfinal_result:")

if overall:
    print("ACHIEVED")
else:
    print("NOT ACHIEVED")
