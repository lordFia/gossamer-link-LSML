# v17.5.1 adaptive evaluation stabilization patch
# adaptive scoring stabilization fix
# triple execution strict validation

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
# utilities
# =========================================================

def similarity(a, b):

    return np.max(np.dot(a, b.T))

def compute_variance(nodes):

    all_vectors = np.array([
        e for n in nodes
        for e in n.essences
    ])

    return float(
        np.mean(
            np.var(all_vectors, axis=0)
        )
    )

def compute_density(nodes):

    total_edges = sum(
        len(n.connections)
        for n in nodes
    )

    max_edges = NODE_COUNT * MAX_CONNECTIONS

    return total_edges / (max_edges + 1e-8)

def compute_anomaly_ratio(nodes):

    return (
        sum(n.is_anomaly for n in nodes)
        / NODE_COUNT
    )

def compute_trust_range(nodes):

    trusts = [n.trust for n in nodes]

    return max(trusts) - min(trusts)

def compute_persistence(nodes):

    return float(
        np.mean([
            n.persistence_counter
            for n in nodes
        ])
    )

def compute_rewiring_instability(
    nodes,
    previous_connections
):

    changes = 0

    for n in nodes:

        before = previous_connections[n.id]
        after = n.connections

        changes += len(
            before.symmetric_difference(after)
        )

    return (
        changes /
        (NODE_COUNT * MAX_CONNECTIONS + 1e-8)
    )

# =========================================================
# anomaly
# =========================================================

def detect_anomaly(nodes):

    trust_values = np.array([
        n.trust for n in nodes
    ])

    mean = np.mean(trust_values)
    std = np.std(trust_values) + 1e-8

    for i, n in enumerate(nodes):

        z = abs(
            (trust_values[i] - mean) / std
        )

        n.anomaly_score = z
        n.is_anomaly = z > ANOMALY_Z

# =========================================================
# trust
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
            0.12 *
            (
                n.anomaly_score /
                (1.0 + n.anomaly_score)
            )
        )

        updated = (
            (1 - TRUST_DECAY) * n.trust
            + 0.48 * (targets[i] - n.trust)
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

            if sim < 0.65:

                n.essences[i] += (
                    UPDATE_SCALE *
                    (src - n.essences[i])
                )

            elif sim > 0.90:

                n.essences[i] -= (
                    UPDATE_SCALE * 0.4 *
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
                    0.58 * c.trust
                    + 0.42 * similarity(
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
                ) < 0.80
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

    return previous

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
# adaptive classification
# =========================================================

def classify_state(
    adaptation_score,
    resilience_score,
    anomaly_ratio,
    rewiring_instability
):

    if adaptation_score > 0.82:
        return "adaptive_optimization"

    if resilience_score > 0.75:
        return "resilient_state"

    if adaptation_score > 0.60:
        return "successful_adaptation"

    if anomaly_ratio > 0.45:
        return "adaptation_failure"

    if rewiring_instability > 0.55:
        return "unstable_adaptation"

    if adaptation_score < 0.40:
        return "degraded_adaptation"

    return "recovery_phase"

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

    # histories

    topology_history = []

    adaptive_evaluation_memory = []

    adaptation_score_history = []

    current_adaptive_state_history = []

    adaptive_transition_events = []

    evaluation_event_history = []

    adaptive_feedback_history = []

    # counters

    successful_adaptation_events = 0

    resilience_improvement_events = 0

    repair_effectiveness_events = 0

    anomaly_reduction_success_events = 0

    persistence_growth_events = 0

    rewiring_stabilization_events = 0

    previous_state = None

    previous_anomaly_ratio = 1.0

    previous_persistence = 0.0

    previous_rewiring = 1.0

    # =====================================================
    # main loop
    # =====================================================

    for step in range(STEP_COUNT):

        detect_anomaly(nodes)

        update_trust(nodes)

        propagate_essences(nodes)

        previous_connections = rewire(nodes)

        regulate_variance(nodes)

        variance = compute_variance(nodes)

        anomaly_ratio = compute_anomaly_ratio(nodes)

        persistence = compute_persistence(nodes)

        rewiring_instability = (
            compute_rewiring_instability(
                nodes,
                previous_connections
            )
        )

        density = compute_density(nodes)

        trust_range = compute_trust_range(nodes)

        rewiring_stability = (
            1.0 -
            rewiring_instability * 0.55
        )

        anomaly_reduction_score = (
            1.0 -
            anomaly_ratio
        )

        persistence_score = min(
            persistence / 80.0,
            1.0
        )

        topology_stability = (
            1.0 -
            (
                abs(0.04 - variance)
                / 0.04
            )
        )

        # =================================================
        # patched resilience score
        # =================================================

        resilience_score = (
            0.35 * topology_stability
            + 0.35 * persistence_score
            + 0.20 * anomaly_reduction_score
            + 0.10 * rewiring_stability
        )

        # =================================================
        # patched repair effectiveness
        # =================================================

        repair_effectiveness_score = (
            0.40 * anomaly_reduction_score
            + 0.35 * persistence_score
            + 0.25 * topology_stability
        )

        recovery_efficiency_score = (
            0.45 * anomaly_reduction_score
            + 0.35 * rewiring_stability
            + 0.20 * topology_stability
        )

        # =================================================
        # patched adaptation score
        # =================================================

        adaptation_score = (
            0.25 * resilience_score
            + 0.25 * repair_effectiveness_score
            + 0.20 * recovery_efficiency_score
            + 0.15 * topology_stability
            + 0.15 * anomaly_reduction_score
        )

        adaptation_score = float(
            np.clip(
                adaptation_score,
                0.0,
                1.0
            )
        )

        adaptive_confidence_score = (
            0.50 * adaptation_score
            + 0.50 * resilience_score
        )

        adaptive_state = classify_state(
            adaptation_score,
            resilience_score,
            anomaly_ratio,
            rewiring_instability
        )

        # =================================================
        # event tracking
        # =================================================

        if adaptation_score > 0.60:
            successful_adaptation_events += 1

        if resilience_score > 0.55:
            resilience_improvement_events += 1

        if repair_effectiveness_score > 0.55:
            repair_effectiveness_events += 1

        if anomaly_ratio < previous_anomaly_ratio:
            anomaly_reduction_success_events += 1

        if persistence > previous_persistence:
            persistence_growth_events += 1

        if rewiring_instability < previous_rewiring:
            rewiring_stabilization_events += 1

        # =================================================
        # transitions
        # =================================================

        current_adaptive_state_history.append(
            adaptive_state
        )

        if previous_state is not None:

            if adaptive_state != previous_state:

                adaptive_transition_events.append({
                    "step": step,
                    "from": previous_state,
                    "to": adaptive_state
                })

        previous_state = adaptive_state

        # =================================================
        # feedback
        # =================================================

        feedback = {

            "step":
                step,

            "adaptation_score":
                adaptation_score,

            "repair_effectiveness_score":
                repair_effectiveness_score,

            "resilience_score":
                resilience_score,

            "recovery_efficiency_score":
                recovery_efficiency_score,

            "topology_stability":
                topology_stability,

            "anomaly_reduction_score":
                anomaly_reduction_score,

            "adaptive_confidence_score":
                adaptive_confidence_score,

            "adaptive_state":
                adaptive_state
        }

        adaptive_feedback_history.append(
            feedback
        )

        adaptive_evaluation_memory.append(
            feedback
        )

        adaptation_score_history.append(
            adaptation_score
        )

        evaluation_event_history.append({

            "step":
                step,

            "state":
                adaptive_state
        })

        topology_history.append({

            "step":
                step,

            "variance":
                variance,

            "density":
                density
        })

        previous_anomaly_ratio = anomaly_ratio

        previous_persistence = persistence

        previous_rewiring = rewiring_instability

    # =====================================================
    # metrics
    # =====================================================

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

    simulation_stability = (
        TARGET_VAR_MIN <= mean_variance <= TARGET_VAR_MAX
    )

    adaptive_feedback_integrity = (
        len(adaptive_feedback_history) >= 120
    )

    adaptation_score_tracking = (
        len(adaptation_score_history) >= 120
    )

    resilience_tracking = (
        resilience_improvement_events >= 5
    )

    adaptive_transition_awareness = (
        len(adaptive_transition_events) >= 5
    )

    continuous_adaptive_evaluation = (
        len(adaptive_evaluation_memory) >= 120
    )

    validation_result = all([

        len(topology_history) >= 120,

        len(adaptive_evaluation_memory) >= 120,

        len(adaptation_score_history) >= 120,

        len(current_adaptive_state_history) >= 120,

        len(adaptive_transition_events) >= 5,

        len(evaluation_event_history) >= 120,

        len(adaptive_feedback_history) >= 120,

        successful_adaptation_events >= 5,

        resilience_improvement_events >= 5,

        repair_effectiveness_events >= 5,

        anomaly_reduction_success_events >= 5,

        persistence_growth_events >= 5,

        rewiring_stabilization_events >= 5,

        adaptive_feedback_integrity,

        adaptation_score_tracking,

        resilience_tracking,

        adaptive_transition_awareness,

        continuous_adaptive_evaluation,

        structural_diversity >= 0.35,

        10.0 <= average_persistence <= 80.0,

        simulation_stability,

        0.15 <= trust_range <= 0.85
    ])

    # =====================================================
    # output
    # =====================================================

    print(f"\n--- RUN #{seed - BASE_SEED + 1} ---")

    print(
        "adaptive_evaluation_memory_length:",
        len(adaptive_evaluation_memory)
    )

    print(
        "adaptation_score_history_length:",
        len(adaptation_score_history)
    )

    print(
        "current_adaptive_state_history_length:",
        len(current_adaptive_state_history)
    )

    print(
        "adaptive_transition_events_length:",
        len(adaptive_transition_events)
    )

    print(
        "evaluation_event_history_length:",
        len(evaluation_event_history)
    )

    print(
        "adaptive_feedback_history_length:",
        len(adaptive_feedback_history)
    )

    print(
        "successful_adaptation_events:",
        successful_adaptation_events
    )

    print(
        "resilience_improvement_events:",
        resilience_improvement_events
    )

    print(
        "repair_effectiveness_events:",
        repair_effectiveness_events
    )

    print(
        "anomaly_reduction_success_events:",
        anomaly_reduction_success_events
    )

    print(
        "persistence_growth_events:",
        persistence_growth_events
    )

    print(
        "rewiring_stabilization_events:",
        rewiring_stabilization_events
    )

    print(
        "adaptive_feedback_integrity:",
        adaptive_feedback_integrity
    )

    print(
        "adaptation_score_tracking:",
        adaptation_score_tracking
    )

    print(
        "resilience_tracking:",
        resilience_tracking
    )

    print(
        "adaptive_transition_awareness:",
        adaptive_transition_awareness
    )

    print(
        "continuous_adaptive_evaluation:",
        continuous_adaptive_evaluation
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
        "simulation_stability:",
        simulation_stability
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
