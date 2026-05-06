# v14.3.1 evaluation integration stabilization
# critical fix:
# preserve structural diversity
# prevent variance collapse
# maintain weak human pressure

import numpy as np
import hashlib
import random

# ============================================
# constants
# ============================================

NUM_ROLES = 3
MAX_CONNECTIONS = 4
SEARCH_K = 5
ESSENCE_SLOTS = 3

POW_DIFFICULTY = 2

TRUST_DECAY = 0.02

MIN_TRUST = 0.05
MAX_TRUST = 0.95

ANOMALY_Z = 1.1
ANOMALY_WEIGHT = 0.22

TARGET_VAR_MIN = 0.02
TARGET_VAR_MAX = 0.06

EPS = 1e-8

INPUT_DIM = 8

# evaluation
EMA_ALPHA = 0.10
EVAL_DECAY = 0.992

# weaker than previous version
HUMAN_FEEDBACK_WEIGHT = 0.018

# diversity preservation
DIVERSITY_PUSH = 0.035

# ============================================
# preprocessor
# ============================================

class HumanPreprocessor:

    def preprocess(self, raw):

        x = np.array(raw, dtype=np.float64)

        if x.shape != (INPUT_DIM,):

            padded = np.zeros(INPUT_DIM)

            usable = min(INPUT_DIM, len(x))

            padded[:usable] = x[:usable]

            x = padded

        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        norm = np.linalg.norm(x)

        if norm < EPS:

            x = np.zeros(INPUT_DIM)
            x[0] = 1.0

            norm = np.linalg.norm(x)

        x = x / (norm + EPS)

        return x


# ============================================
# evaluation memory
# ============================================

class EvaluationMemory:

    def __init__(self):

        self.state = np.zeros(INPUT_DIM)

    def update(self, processed):

        self.state = (
            EVAL_DECAY * self.state
            + EMA_ALPHA * processed
        )

        norm = np.linalg.norm(self.state)

        # keep human influence weak
        if norm > 0.15:

            self.state *= (
                0.15 / (norm + EPS)
            )

        return self.state.copy()


# ============================================
# node
# ============================================

class Node:

    def __init__(self, idx, dim=8):

        self.id = idx

        self.essences = np.random.randn(
            ESSENCE_SLOTS,
            dim
        )

        self.essences /= (
            np.linalg.norm(
                self.essences,
                axis=1,
                keepdims=True
            ) + EPS
        )

        self.trust = 0.5

        self.role = np.random.randint(
            0,
            NUM_ROLES
        )

        self.connections = set()

        self.age = 1

        self.nonce = 0

        self.anomaly_score = 0.0
        self.is_anomaly = False


# ============================================
# utility
# ============================================

def compute_hash(v):

    return hashlib.sha256(
        str(v).encode()
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


def sim(a, b):

    return np.max(
        np.dot(a, b.T)
    )


# ============================================
# anomaly
# ============================================

def detect_anomaly(nodes):

    vals = np.array([
        n.trust for n in nodes
    ])

    mean = np.mean(vals)

    std = np.std(vals) + EPS

    for i, n in enumerate(nodes):

        z = abs(
            (vals[i] - mean)
            / std
        )

        n.anomaly_score = float(z)

        n.is_anomaly = (
            z > ANOMALY_Z
        )


# ============================================
# trust
# ============================================

def update_trust(nodes):

    scores = []

    for n in nodes:

        neighbors = [
            nodes[i]
            for i in n.connections
        ]

        if neighbors:

            sims = [
                sim(
                    n.essences,
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
        ranks
        / (len(nodes) - 1 + EPS)
    )

    target = (
        MIN_TRUST
        + (MAX_TRUST - MIN_TRUST)
        * target
    )

    mean_trust = np.mean([
        n.trust for n in nodes
    ])

    for i, n in enumerate(nodes):

        anomaly_scale = (
            1.0
            - ANOMALY_WEIGHT
            * (
                n.anomaly_score
                / (1.0 + n.anomaly_score)
            )
        )

        new_trust = (
            (1.0 - TRUST_DECAY)
            * n.trust

            + 0.45
            * (target[i] - n.trust)

            + 0.08
            * (n.trust - mean_trust)
        )

        new_trust *= anomaly_scale

        if new_trust <= MIN_TRUST:

            new_trust = (
                MIN_TRUST
                + 0.03 * random.random()
            )

        n.trust = float(
            np.clip(
                new_trust,
                MIN_TRUST,
                MAX_TRUST
            )
        )


# ============================================
# essence update
# critical fix:
# add diversity repulsion
# reduce over-convergence
# ============================================

def update_essences(nodes, eval_vec):

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

            sim_val = np.dot(
                node.essences[i],
                src
            )

            # softer convergence
            if sim_val < 0.6:

                node.essences[i] += (
                    0.04
                    * (
                        src
                        - node.essences[i]
                    )
                )

            elif sim_val > 0.88:

                node.essences[i] -= (
                    0.025
                    * (
                        src
                        - node.essences[i]
                    )
                )

            # weak human pressure
            node.essences[i] += (
                HUMAN_FEEDBACK_WEIGHT
                * (
                    eval_vec
                    - node.essences[i]
                )
            )

            # diversity repulsion
            repel = np.zeros(INPUT_DIM)

            for j in range(ESSENCE_SLOTS):

                if i == j:
                    continue

                s = np.dot(
                    node.essences[i],
                    node.essences[j]
                )

                if s > 0.7:

                    repel += (
                        node.essences[i]
                        - node.essences[j]
                    )

            node.essences[i] += (
                DIVERSITY_PUSH * repel
            )

            node.essences[i] /= (
                np.linalg.norm(
                    node.essences[i]
                ) + EPS
            )


# ============================================
# variance clamp
# ============================================

def regulate_variance(nodes):

    all_vecs = np.array([
        e
        for n in nodes
        for e in n.essences
    ])

    var = np.var(
        all_vecs,
        axis=0
    )

    mean_var = np.mean(var)

    # low variance recovery
    if mean_var < TARGET_VAR_MIN:

        scale = np.sqrt(
            TARGET_VAR_MIN
            / (mean_var + EPS)
        )

        for n in nodes:

            noise = (
                np.random.randn(
                    *n.essences.shape
                ) * 0.03
            )

            n.essences += noise

            n.essences *= scale

            n.essences /= (
                np.linalg.norm(
                    n.essences,
                    axis=1,
                    keepdims=True
                ) + EPS
            )

    # upper clamp
    if mean_var > TARGET_VAR_MAX:

        scale = np.sqrt(
            TARGET_VAR_MAX
            / (mean_var + EPS)
        )

        for n in nodes:

            n.essences *= scale


# ============================================
# diversity
# ============================================

def structural_diversity(nodes):

    vals = []

    for i in range(len(nodes)):

        for j in range(i + 1, len(nodes)):

            vals.append(
                sim(
                    nodes[i].essences,
                    nodes[j].essences
                )
            )

    vals = np.array(vals)

    return float(
        1.0 - np.mean(vals)
    )


# ============================================
# dominance
# ============================================

def dominance_ratio(nodes):

    vals = np.array([
        n.trust for n in nodes
    ])

    return float(
        np.max(vals)
        / (np.sum(vals) + EPS)
    )


# ============================================
# step
# ============================================

def step(nodes, eval_vec):

    for n in nodes:

        n.age += 1

        mine_pow(n)

    detect_anomaly(nodes)

    for node in nodes:

        candidates = [
            n for n in nodes
            if n.id != node.id
        ]

        scored = []

        for c in candidates:

            s = sim(
                node.essences,
                c.essences
            )

            score = (
                0.6 * c.trust
                + 0.4 * s
            )

            scored.append(
                (score, c)
            )

        scored.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        selected = []

        for _, c in scored:

            if len(selected) >= SEARCH_K:
                break

            if all(
                sim(
                    c.essences,
                    s.essences
                ) < 0.68
                for s in selected
            ):
                selected.append(c)

        while len(selected) < SEARCH_K:

            c = random.choice(
                candidates
            )

            if c not in selected:
                selected.append(c)

        node.connections = {
            c.id
            for c in selected[
                :MAX_CONNECTIONS
            ]
        }

    update_trust(nodes)

    update_essences(
        nodes,
        eval_vec
    )

    regulate_variance(nodes)


# ============================================
# validation run
# ============================================

pre = HumanPreprocessor()

memory = EvaluationMemory()

nodes = [
    Node(i)
    for i in range(10)
]

eval_norms = []

for t in range(400):

    persistent = np.array([
        0.7,
        -0.5,
        0.8,
        0.2,
        -0.3,
        0.6,
        0.1,
        -0.4
    ])

    spike = np.zeros(INPUT_DIM)

    if t % 35 == 0:

        spike += (
            np.random.randn(INPUT_DIM)
            * 2.5
        )

    noise = (
        np.random.randn(INPUT_DIM)
        * 0.08
    )

    raw = (
        persistent
        + spike
        + noise
    )

    processed = pre.preprocess(raw)

    eval_vec = memory.update(
        processed
    )

    eval_norms.append(
        np.linalg.norm(eval_vec)
    )

    step(
        nodes,
        eval_vec
    )

# ============================================
# metrics
# ============================================

trusts = np.array([
    n.trust for n in nodes
])

trust_range = float(
    np.max(trusts)
    - np.min(trusts)
)

all_vecs = np.array([
    e
    for n in nodes
    for e in n.essences
])

variances = np.var(
    all_vecs,
    axis=0
)

mean_variance = float(
    np.mean(variances)
)

diversity = float(
    structural_diversity(nodes)
)

dom_ratio = float(
    dominance_ratio(nodes)
)

eval_memory_norm = float(
    np.mean(eval_norms)
)

trust_saturation = bool(
    np.any(trusts >= 0.999)
)

validation_result = True

if not (
    0.02 <= mean_variance <= 0.06
):
    validation_result = False

if trust_range < 0.2:
    validation_result = False

if trust_saturation:
    validation_result = False

if diversity <= 0.08:
    validation_result = False

if dom_ratio >= 0.5:
    validation_result = False

# ============================================
# output
# ============================================

print("metrics")

print(
    "mean_variance:",
    round(mean_variance, 6)
)

print(
    "trust_range:",
    round(trust_range, 6)
)

print(
    "human_feedback_weight:",
    HUMAN_FEEDBACK_WEIGHT
)

print(
    "ema_smoothing_active:",
    True
)

print(
    "evaluation_decay_active:",
    True
)

print(
    "evaluation_memory_norm:",
    round(eval_memory_norm, 6)
)

print(
    "structural_diversity:",
    round(diversity, 6)
)

print(
    "dominance_ratio:",
    round(dom_ratio, 6)
)

print(
    "trust_saturation:",
    trust_saturation
)

print(
    "validation_result:",
    validation_result
)

print(
    "final_result:",
    "achieved"
    if validation_result
    else "not achieved"
)
