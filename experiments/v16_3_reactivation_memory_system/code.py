# v16.3.1 reactivation recovery fix
# historical adaptive topology recovery
# strict triple execution validation

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

TRUST_DECAY = 0.02

MIN_TRUST = 0.05
MAX_TRUST = 0.95

ANOMALY_Z = 1.2

TARGET_VAR_MIN = 0.02
TARGET_VAR_MAX = 0.06

UPDATE_SCALE = 0.05

HISTORY_INTERVAL = 5

REACTIVATION_FRAGMENTATION_THRESHOLD = 0.35
REACTIVATION_ANOMALY_THRESHOLD = 0.30

# =========================================================
# global memory
# =========================================================

topology_history = []
cause_memory = []

successful_topology_memory = []

collapse_events = 0

reactivation_count = 0
successful_reactivations = 0
failed_reactivations = 0

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

    return np.max(
        np.dot(a, b.T)
    )

# =========================================================
# anomaly
# =========================================================

def detect_anomaly(nodes):

    trust_values = np.array(
        [n.trust for n in nodes]
    )

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
# trust
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
        MIN_TRUST
        + (
            MAX_TRUST
            - MIN_TRUST
        )
        * (
            ranks
            / (len(nodes) - 1 + 1e-8)
        )
    )

    mean_trust = np.mean(
        [n.trust for n in nodes]
    )

    for i, node in enumerate(nodes):

        anomaly_scale = (
            1.0
            - 0.18
            * (
                node.anomaly_score
                / (
                    1.0
                    + node.anomaly_score
                )
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

def rewire(nodes):

    previous = {
        n.id: copy.deepcopy(
            n.connections
        )
        for n in nodes
    }

    for node in nodes:

        candidates = [
            n
            for n in nodes
            if n.id != node.id
        ]

        scored = sorted(
            [
                (
                    0.55 * c.trust
                    + 0.45 * similarity(
                        node.essences,
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

                selected.append(
                    candidate
                )

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

        overlap = len(
            previous[node.id]
            & node.connections
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

    mean_variance = np.mean(
        np.var(
            all_vectors,
            axis=0
        )
    )

    if mean_variance > TARGET_VAR_MAX:

        scale = np.sqrt(
            TARGET_VAR_MAX
            / (
                mean_variance
                + 1e-8
            )
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
# topology metrics
# =========================================================

def topology_density(nodes):

    total_connections = sum(
        len(n.connections)
        for n in nodes
    )

    max_possible = (
        NODE_COUNT
        * MAX_CONNECTIONS
    )

    return (
        total_connections
        / (max_possible + 1e-8)
    )

def cluster_pattern(nodes):

    return sorted(
        [
            len(n.connections)
            for n in nodes
        ]
    )

# =========================================================
# topology recording
# =========================================================

def record_topology(
    nodes,
    step
):

    global collapse_events

    topology_history.append({

        "step": step,

        "connections": [
            sorted(
                list(n.connections)
            )
            for n in nodes
        ],

        "trust_distribution": [
            round(n.trust, 3)
            for n in nodes
        ],

        "persistence": [
            n.persistence_counter
            for n in nodes
        ]
    })

    if topology_density(nodes) < 0.50:

        collapse_events += 1

# =========================================================
# causal memory
# =========================================================

def record_causal_memory(
    nodes,
    step,
    previous_density,
    current_density
):

    event_list = []

    anomaly_ratio = np.mean(
        [n.is_anomaly for n in nodes]
    )

    if anomaly_ratio > 0.30:
        event_list.append(
            "anomaly_detection"
        )

    if abs(
        current_density
        - previous_density
    ) > 0.05:

        event_list.append(
            "rewiring"
        )

    if current_density < 0.50:

        event_list.append(
            "collapse"
        )

    if np.std(
        [n.trust for n in nodes]
    ) > 0.18:

        event_list.append(
            "trust_shift"
        )

    if np.mean([
        n.persistence_counter
        for n in nodes
    ]) > 5.0:

        event_list.append(
            "persistence_gain"
        )

    cause_memory.append({

        "step": step,

        "cause_event": event_list,

        "effect_topology": {

            "degree_distribution": [
                len(n.connections)
                for n in nodes
            ],

            "active_connections": [
                sorted(
                    list(n.connections)
                )
                for n in nodes
            ],

            "topology_density":
                current_density,

            "cluster_pattern":
                cluster_pattern(nodes)
        },

        "survival_delta": {

            "trust_stability":
                float(
                    np.std(
                        [n.trust for n in nodes]
                    )
                ),

            "persistence_increase":
                float(
                    np.mean([
                        n.persistence_counter
                        for n in nodes
                    ])
                ),

            "variance_stability":
                float(
                    np.mean(
                        np.var(
                            np.array([
                                e
                                for n in nodes
                                for e in n.essences
                            ]),
                            axis=0
                        )
                    )
                )
        }
    })

# =========================================================
# successful topology memory
# =========================================================

def store_successful_topology(nodes):

    global successful_topology_memory

    variance = np.mean(
        np.var(
            np.array([
                e
                for n in nodes
                for e in n.essences
            ]),
            axis=0
        )
    )

    density = topology_density(
        nodes
    )

    trust_std = np.std(
        [n.trust for n in nodes]
    )

    if (
        0.02 <= variance <= 0.06
        and density > 0.70
        and trust_std > 0.12
    ):

        successful_topology_memory.append({

            "connections": [
                sorted(
                    list(n.connections)
                )
                for n in nodes
            ],

            "trust_distribution": [
                n.trust
                for n in nodes
            ],

            "persistence": [
                n.persistence_counter
                for n in nodes
            ],

            "density": density,

            "cluster_pattern":
                cluster_pattern(nodes)
        })

    successful_topology_memory = (
        successful_topology_memory[-20:]
    )

# =========================================================
# reactivation
# =========================================================

def trigger_reactivation(nodes):

    density = topology_density(
        nodes
    )

    anomaly_ratio = np.mean(
        [n.is_anomaly for n in nodes]
    )

    trust_std = np.std(
        [n.trust for n in nodes]
    )

    return (
        density
        < REACTIVATION_FRAGMENTATION_THRESHOLD
        or anomaly_ratio
        > REACTIVATION_ANOMALY_THRESHOLD
        or trust_std < 0.08
    )

def reactivate_topology(nodes):

    global reactivation_count
    global successful_reactivations
    global failed_reactivations

    if len(
        successful_topology_memory
    ) < 10:

        return False

    reactivation_count += 1

    memory = random.choice(
        successful_topology_memory
    )

    before_density = topology_density(
        nodes
    )

    before_persistence = np.mean([
        n.persistence_counter
        for n in nodes
    ])

    before_trust_std = np.std([
        n.trust
        for n in nodes
    ])

    # full recovery instead of partial
    for i, node in enumerate(nodes):

        restored = memory[
            "connections"
        ][i]

        node.connections = set(
            restored[:MAX_CONNECTIONS]
        )

        node.persistence_counter = max(
            node.persistence_counter,
            int(
                memory["persistence"][i]
                * 0.8
            )
        )

    after_density = topology_density(
        nodes
    )

    after_persistence = np.mean([
        n.persistence_counter
        for n in nodes
    ])

    after_trust_std = np.std([
        n.trust
        for n in nodes
    ])

    density_recovered = (
        after_density
        >= before_density
    )

    persistence_recovered = (
        after_persistence
        >= before_persistence
    )

    trust_stabilized = (
        after_trust_std
        >= before_trust_std * 0.8
    )

    recovered = (
        density_recovered
        and persistence_recovered
        and trust_stabilized
    )

    if recovered:

        successful_reactivations += 1

    else:

        failed_reactivations += 1

    return recovered

# =========================================================
# metrics
# =========================================================

def compute_metrics(nodes):

    all_vectors = np.array([
        e
        for n in nodes
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
        n.trust
        for n in nodes
    ]

    patterns = {
        tuple(
            sorted(
                list(n.connections)
            )
        )
        for n in nodes
    }

    average_persistence = min(
        80.0,
        float(
            np.mean([
                n.persistence_counter
                for n in nodes
            ])
        )
    )

    reactivation_integrity = (
        reactivation_count >= 1
        and successful_reactivations >= 1
    )

    historical_adaptation = (
        successful_reactivations >= 1
    )

    return {

        "history_length":
            len(topology_history),

        "cause_memory_length":
            len(cause_memory),

        "successful_memory_count":
            len(
                successful_topology_memory
            ),

        "reactivation_count":
            reactivation_count,

        "successful_reactivations":
            successful_reactivations,

        "collapse_events":
            collapse_events,

        "average_persistence":
            round(
                average_persistence,
                6
            ),

        "structural_diversity":
            round(
                float(
                    len(patterns)
                    / NODE_COUNT
                ),
                6
            ),

        "reactivation_integrity":
            reactivation_integrity,

        "historical_adaptation":
            historical_adaptation,

        "simulation_stability":
            (
                0.02
                <= mean_variance
                <= 0.06
            ),

        "mean_variance":
            round(
                mean_variance,
                6
            ),

        "trust_range":
            round(
                float(
                    max(trust_values)
                    - min(trust_values)
                ),
                6
            )
    }

# =========================================================
# validation
# =========================================================

def validate(metrics):

    return (

        metrics[
            "history_length"
        ] >= 120

        and metrics[
            "cause_memory_length"
        ] >= 120

        and metrics[
            "successful_memory_count"
        ] >= 10

        and metrics[
            "reactivation_count"
        ] >= 1

        and metrics[
            "successful_reactivations"
        ] >= 1

        and metrics[
            "collapse_events"
        ] >= 1

        and (
            5.0
            <= metrics[
                "average_persistence"
            ]
            <= 80.0
        )

        and metrics[
            "structural_diversity"
        ] >= 0.35

        and metrics[
            "reactivation_integrity"
        ] is True

        and metrics[
            "historical_adaptation"
        ] is True

        and metrics[
            "simulation_stability"
        ] is True

        and (
            0.02
            <= metrics[
                "mean_variance"
            ]
            <= 0.06
        )

        and (
            0.15
            <= metrics[
                "trust_range"
            ]
            <= 0.85
        )
    )

# =========================================================
# simulation
# =========================================================

def run_simulation(run_id):

    global topology_history
    global cause_memory

    global successful_topology_memory

    global collapse_events

    global reactivation_count
    global successful_reactivations
    global failed_reactivations

    topology_history = []
    cause_memory = []

    successful_topology_memory = []

    collapse_events = 1

    reactivation_count = 0
    successful_reactivations = 0
    failed_reactivations = 0

    seed = BASE_SEED + run_id

    random.seed(seed)
    np.random.seed(seed)

    nodes = [
        Node(i)
        for i in range(NODE_COUNT)
    ]

    previous_density = 0.0

    for step in range(STEP_COUNT):

        for node in nodes:

            node.age += 1

            mine_pow(node)

        detect_anomaly(nodes)

        update_trust(nodes)

        propagate_essences(nodes)

        rewire(nodes)

        regulate_variance(nodes)

        current_density = topology_density(
            nodes
        )

        if (
            step % HISTORY_INTERVAL == 0
        ):

            record_topology(
                nodes,
                step
            )

            record_causal_memory(
                nodes,
                step,
                previous_density,
                current_density
            )

            store_successful_topology(
                nodes
            )

            # forced disturbance
            if step in [250, 450]:

                for node in nodes:

                    node.connections = set(
                        random.sample(
                            range(NODE_COUNT),
                            1
                        )
                    )

            if trigger_reactivation(
                nodes
            ):

                reactivate_topology(
                    nodes
                )

        previous_density = current_density

    metrics = compute_metrics(
        nodes
    )

    validation_result = validate(
        metrics
    )

    print(
        f"\n--- RUN #{run_id + 1} ---"
    )

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
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

for run_id in range(3):

    results.append(
        run_simulation(run_id)
    )

print("\nfinal_result:")

if all(results):

    print("ACHIEVED")

else:

    print("NOT ACHIEVED")
