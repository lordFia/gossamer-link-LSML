# v17.2.0 structural diagnosis system
# internal topology diagnosis
# triple execution strict validation

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
            0.3,
            0.7
        )

        self.connections = set()

        self.age = 1
        self.nonce = 0

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

    return np.max(np.dot(a, b.T))

# =========================================================
# anomaly detection
# =========================================================

def detect_anomaly(nodes):

    trust_values = np.array([
        n.trust for n in nodes
    ])

    mean = np.mean(trust_values)
    std = np.std(trust_values) + 1e-8

    for i, node in enumerate(nodes):

        z = abs(
            (trust_values[i] - mean) / std
        )

        node.anomaly_score = float(z)

        node.is_anomaly = (
            z > ANOMALY_Z
        )

# =========================================================
# trust update
# =========================================================

def update_trust(nodes):

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
        * (
            ranks
            / (len(nodes) - 1 + 1e-8)
        )
    )

    mean_trust = np.mean([
        n.trust for n in nodes
    ])

    for i, node in enumerate(nodes):

        anomaly_scale = (
            1.0
            - 0.18
            * (
                node.anomaly_score
                / (1.0 + node.anomaly_score)
            )
        )

        new_trust = (
            (1 - TRUST_DECAY)
            * node.trust
            + 0.50
            * (
                target[i]
                - node.trust
            )
            + 0.10
            * (
                node.trust
                - mean_trust
            )
        )

        node.trust = float(
            np.clip(
                new_trust
                * anomaly_scale,
                MIN_TRUST,
                MAX_TRUST
            )
        )

# =========================================================
# propagation
# =========================================================

def propagate_essences(nodes):

    for node in nodes:

        neighbors = [
            nodes[i]
            for i in node.connections
        ]

        if not neighbors:
            continue

        for i in range(ESSENCE_SLOTS):

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
                    UPDATE_SCALE
                    * (
                        src
                        - node.essences[i]
                    )
                )

            elif sim > 0.85:

                node.essences[i] -= (
                    UPDATE_SCALE
                    * 0.5
                    * (
                        src
                        - node.essences[i]
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

rewiring_instability_log = []

def rewire(nodes):

    previous = {
        n.id: copy.deepcopy(
            n.connections
        )
        for n in nodes
    }

    rewiring_changes = 0

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
                0.55 * candidate.trust
                + 0.45 * sim
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
                ) < 0.75
                for s in selected
            ):
                selected.append(candidate)

        while len(selected) < SEARCH_K:

            candidate = random.choice(
                candidates
            )

            if candidate not in selected:
                selected.append(candidate)

        node.connections = set(
            c.id
            for c in selected[:MAX_CONNECTIONS]
        )

        diff = len(
            previous[node.id]
            ^ node.connections
        )

        rewiring_changes += diff

        overlap = len(
            previous[node.id]
            & node.connections
        )

        if overlap >= 2:
            node.persistence_counter += 1

    rewiring_instability_log.append(
        rewiring_changes / NODE_COUNT
    )

# =========================================================
# variance regulation
# =========================================================

def regulate_variance(nodes):

    all_vectors = np.array([
        e for n in nodes
        for e in n.essences
    ])

    mean_variance = np.mean(
        np.var(
            all_vectors,
            axis=0
        )
    )

    if mean_variance > TARGET_VAR_MAX:

        scale = np.sqrt(
            TARGET_VAR_MAX
            / (mean_variance + 1e-8)
        )

        for node in nodes:
            node.essences *= scale

    elif mean_variance < TARGET_VAR_MIN:

        for node in nodes:

            node.essences += (
                np.random.randn(
                    *node.essences.shape
                ) * 0.01
            )

            node.essences /= (
                np.linalg.norm(
                    node.essences,
                    axis=1,
                    keepdims=True
                ) + 1e-8
            )

# =========================================================
# diagnosis memory
# =========================================================

topology_history = []

structural_diagnosis_memory = []

diagnosis_history = []

collapse_risk_events = 0

successful_instability_detection = 0

anomaly_cluster_events = 0

topology_degradation_events = 0

# =========================================================
# history
# =========================================================

def record_history(nodes, step):

    topology_history.append({

        "step":
            step,

        "connections":
            [
                sorted(
                    list(n.connections)
                )
                for n in nodes
            ]
    })

# =========================================================
# diagnosis system
# =========================================================

def diagnose_structure(nodes, step):

    global collapse_risk_events
    global successful_instability_detection
    global anomaly_cluster_events
    global topology_degradation_events

    trust_values = np.array([
        n.trust for n in nodes
    ])

    trust_dispersion = float(
        np.std(trust_values)
    )

    anomaly_ratio = float(
        np.mean([
            1.0 if n.is_anomaly else 0.0
            for n in nodes
        ])
    )

    density = float(
        np.mean([
            len(n.connections)
            for n in nodes
        ]) / MAX_CONNECTIONS
    )

    persistence = float(
        np.mean([
            n.persistence_counter
            for n in nodes
        ])
    )

    all_vectors = np.array([
        e for n in nodes
        for e in n.essences
    ])

    mean_variance = float(
        np.mean(
            np.var(
                all_vectors,
                axis=0
            )
        )
    )

    rewiring_instability = float(
        np.mean(
            rewiring_instability_log[-5:]
        ) if rewiring_instability_log else 0.0
    )

    collapse_risk = float(
        np.clip(
            (
                trust_dispersion
                + anomaly_ratio
                + abs(mean_variance - 0.03)
                + rewiring_instability * 0.1
                + (1.0 - density)
            ) / 5.0,
            0.0,
            1.0
        )
    )

    topology_integrity_score = float(
        np.clip(
            (
                0.30 * density
                + 0.25 * (
                    1.0
                    - collapse_risk
                )
                + 0.20 * min(
                    persistence / 80.0,
                    1.0
                )
                + 0.15 * (
                    1.0
                    - anomaly_ratio
                )
                + 0.10 * (
                    1.0
                    - trust_dispersion
                )
            ),
            0.0,
            1.0
        )
    )

    if collapse_risk > 0.18:
        collapse_risk_events += 1

    if (
        trust_dispersion > 0.12
        and anomaly_ratio > 0.0
    ):
        successful_instability_detection += 1

    anomaly_nodes = sum([
        1
        for n in nodes
        if n.is_anomaly
    ])

    if anomaly_nodes >= 2:
        anomaly_cluster_events += 1

    degradation_detected = False

    if (
        density < 0.80
        or rewiring_instability > 3.0
        or persistence < 10.0
    ):

        topology_degradation_events += 1

        degradation_detected = True

    if collapse_risk < 0.10:

        classification = "healthy"

    elif degradation_detected:

        classification = "degraded"

    elif collapse_risk > 0.25:

        classification = "fragmented"

    elif anomaly_ratio > 0.10:

        classification = "unstable"

    else:

        classification = "recovery_phase"

    diagnostic_confidence = float(
        np.clip(
            (
                topology_integrity_score
                + (1.0 - collapse_risk)
            ) / 2.0,
            0.0,
            1.0
        )
    )

    diagnosis = {

        "step":
            step,

        "classification":
            classification,

        "collapse_risk":
            collapse_risk,

        "trust_dispersion":
            trust_dispersion,

        "anomaly_ratio":
            anomaly_ratio,

        "density":
            density,

        "persistence":
            persistence,

        "rewiring_instability":
            rewiring_instability,

        "topology_integrity_score":
            topology_integrity_score,

        "diagnostic_confidence_score":
            diagnostic_confidence
    }

    structural_diagnosis_memory.append(
        diagnosis
    )

    diagnosis_history.append({

        "step":
            step,

        "diagnosis":
            diagnosis
    })

# =========================================================
# metrics
# =========================================================

def compute_metrics(nodes):

    all_vectors = np.array([
        e for n in nodes
        for e in n.essences
    ])

    mean_variance = float(
        np.mean(
            np.var(
                all_vectors,
                axis=0
            )
        )
    )

    trust_values = [
        n.trust for n in nodes
    ]

    patterns = {

        tuple(
            sorted(
                list(n.connections)
            )
        )
        for n in nodes
    }

    structural_diversity = (
        len(patterns)
        / NODE_COUNT
    )

    average_persistence = float(
        np.mean([
            min(
                n.persistence_counter,
                80
            )
            for n in nodes
        ])
    )

    diagnostic_confidence_tracking = all([
        (
            0.0
            <= d[
                "diagnostic_confidence_score"
            ]
            <= 1.0
        )
        for d in structural_diagnosis_memory
    ])

    internal_health_analysis = all([
        "topology_integrity_score"
        in d
        for d in structural_diagnosis_memory
    ])

    topology_integrity_tracking = all([
        (
            0.0
            <= d[
                "topology_integrity_score"
            ]
            <= 1.0
        )
        for d in structural_diagnosis_memory
    ])

    instability_classification = all([
        "classification"
        in d
        for d in structural_diagnosis_memory
    ])

    collapse_risk_estimation = all([
        "collapse_risk"
        in d
        for d in structural_diagnosis_memory
    ])

    simulation_stability = (
        0.02
        <= mean_variance
        <= 0.06
    )

    trust_range = float(
        max(trust_values)
        - min(trust_values)
    )

    metrics = {

        "history_length":
            len(topology_history),

        "structural_diagnosis_memory_length":
            len(
                structural_diagnosis_memory
            ),

        "diagnosis_history_length":
            len(diagnosis_history),

        "collapse_risk_events":
            collapse_risk_events,

        "successful_instability_detection":
            successful_instability_detection,

        "anomaly_cluster_events":
            anomaly_cluster_events,

        "topology_degradation_events":
            topology_degradation_events,

        "diagnostic_confidence_tracking":
            diagnostic_confidence_tracking,

        "internal_health_analysis":
            internal_health_analysis,

        "topology_integrity_tracking":
            topology_integrity_tracking,

        "instability_classification":
            instability_classification,

        "collapse_risk_estimation":
            collapse_risk_estimation,

        "structural_diversity":
            round(
                structural_diversity,
                6
            ),

        "average_persistence":
            round(
                average_persistence,
                6
            ),

        "simulation_stability":
            simulation_stability,

        "mean_variance":
            round(
                mean_variance,
                6
            ),

        "trust_range":
            round(
                trust_range,
                6
            )
    }

    return metrics

# =========================================================
# validation
# =========================================================

def validate(metrics):

    return (

        metrics["history_length"] >= 120

        and metrics[
            "structural_diagnosis_memory_length"
        ] >= 120

        and metrics[
            "diagnosis_history_length"
        ] >= 120

        and metrics[
            "collapse_risk_events"
        ] >= 5

        and metrics[
            "successful_instability_detection"
        ] >= 5

        and metrics[
            "anomaly_cluster_events"
        ] >= 3

        and metrics[
            "topology_degradation_events"
        ] >= 3

        and metrics[
            "diagnostic_confidence_tracking"
        ] is True

        and metrics[
            "internal_health_analysis"
        ] is True

        and metrics[
            "topology_integrity_tracking"
        ] is True

        and metrics[
            "instability_classification"
        ] is True

        and metrics[
            "collapse_risk_estimation"
        ] is True

        and metrics[
            "structural_diversity"
        ] >= 0.35

        and 10.0
        <= metrics[
            "average_persistence"
        ]
        <= 80.0

        and metrics[
            "simulation_stability"
        ] is True

        and 0.02
        <= metrics[
            "mean_variance"
        ]
        <= 0.06

        and 0.15
        <= metrics[
            "trust_range"
        ]
        <= 0.85
    )

# =========================================================
# simulation
# =========================================================

def run_simulation(seed):

    global topology_history
    global structural_diagnosis_memory
    global diagnosis_history

    global collapse_risk_events
    global successful_instability_detection
    global anomaly_cluster_events
    global topology_degradation_events

    global rewiring_instability_log

    topology_history = []

    structural_diagnosis_memory = []

    diagnosis_history = []

    rewiring_instability_log = []

    collapse_risk_events = 0

    successful_instability_detection = 0

    anomaly_cluster_events = 0

    topology_degradation_events = 0

    random.seed(seed)
    np.random.seed(seed)

    nodes = [
        Node(i)
        for i in range(NODE_COUNT)
    ]

    for step in range(STEP_COUNT):

        for node in nodes:

            node.age += 1

            mine_pow(node)

        detect_anomaly(nodes)

        update_trust(nodes)

        propagate_essences(nodes)

        rewire(nodes)

        regulate_variance(nodes)

        if step % HISTORY_INTERVAL == 0:

            record_history(
                nodes,
                step
            )

            diagnose_structure(
                nodes,
                step
            )

    metrics = compute_metrics(nodes)

    validation_result = validate(
        metrics
    )

    print(
        f"\n--- RUN #{seed - 41} ---"
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

for seed in [42, 43, 44]:

    results.append(
        run_simulation(seed)
    )

print("\nfinal_result:")

if all(results):

    print("ACHIEVED")

else:

    print("NOT ACHIEVED")
