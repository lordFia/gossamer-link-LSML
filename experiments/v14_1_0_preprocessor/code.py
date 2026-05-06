# v14.1.0 preprocessor integration layer (strict validation edition)

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
QUERY_K = 2

POW_DIFFICULTY = 2
TRUST_DECAY = 0.02

ANOMALY_Z = 1.1
ANOMALY_WEIGHT = 0.25

MIN_TRUST = 0.05
MAX_TRUST = 0.95

TARGET_VAR_MAX = 0.06
UPDATE_SCALE = 0.5

# v14.1.0
INPUT_DIM = 8
EPS = 1e-8

# ============================================
# deterministic preprocessor
# ============================================

class HumanPreprocessor:

    def __init__(self):
        pass

    def preprocess(self, raw_input):

        # convert to numpy
        x = np.array(raw_input, dtype=np.float64)

        # shape stabilization
        if x.shape != (INPUT_DIM,):
            padded = np.zeros(INPUT_DIM, dtype=np.float64)

            usable = min(len(x), INPUT_DIM)
            padded[:usable] = x[:usable]

            x = padded

        # NaN / inf protection
        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # deterministic normalization
        norm = np.linalg.norm(x)

        if norm < EPS:
            x = np.zeros(INPUT_DIM, dtype=np.float64)
            x[0] = 1.0
            norm = np.linalg.norm(x)

        x = x / (norm + EPS)

        # final clamp
        x = np.clip(x, -1.0, 1.0)

        # safety check
        if np.any(np.isnan(x)):
            raise RuntimeError("NaN detected in preprocessor")

        return x


# ============================================
# node
# ============================================

class Node:

    def __init__(self, id, dim=8):

        self.id = id

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

        self.last_active = 0

        self.anomaly_score = 0.0
        self.is_anomaly = False


# ============================================
# utility
# ============================================

def compute_hash(val):
    return hashlib.sha256(
        str(val).encode()
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
    return np.max(np.dot(a, b.T))


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
            (vals[i] - mean) / std
        )

        n.anomaly_score = float(z)

        n.is_anomaly = (
            z > ANOMALY_Z
        )


# ============================================
# trust update
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
                sim(n.essences, nb.essences)
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
        ranks / (len(nodes) - 1 + EPS)
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
            (1 - TRUST_DECAY)
            * n.trust

            + 0.55
            * (target[i] - n.trust)

            + 0.12
            * (n.trust - mean_trust)
        )

        new_trust *= anomaly_scale

        if new_trust <= MIN_TRUST:

            new_trust = (
                MIN_TRUST
                + 0.02 * random.random()
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
# ============================================

def update_essences(nodes):

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

            if sim_val < 0.6:

                node.essences[i] += (
                    UPDATE_SCALE
                    * 0.08
                    * (
                        src
                        - node.essences[i]
                    )
                )

            elif sim_val > 0.85:

                node.essences[i] -= (
                    UPDATE_SCALE
                    * 0.04
                    * (
                        src
                        - node.essences[i]
                    )
                )

            node.essences[i] /= (
                np.linalg.norm(
                    node.essences[i]
                ) + EPS
            )


# ============================================
# variance clamp
# ============================================

def clamp_variance(nodes):

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

    if mean_var > TARGET_VAR_MAX:

        scale = np.sqrt(
            TARGET_VAR_MAX
            / (mean_var + EPS)
        )

        for n in nodes:
            n.essences *= scale


# ============================================
# step
# ============================================

def step(nodes, t):

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
                ) < 0.7
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

    update_essences(nodes)

    clamp_variance(nodes)


# ============================================
# validation
# ============================================

pre = HumanPreprocessor()

test_input = np.array([
    0.25,
    -0.13,
    0.72,
    0.44,
    -0.55,
    0.91,
    -0.33,
    0.11
])

out1 = pre.preprocess(test_input)
out2 = pre.preprocess(test_input)

zero_input = np.zeros(INPUT_DIM)
zero_out = pre.preprocess(zero_input)

# metrics
output_norms = [
    np.linalg.norm(out1),
    np.linalg.norm(out2),
    np.linalg.norm(zero_out)
]

mean_output_norm = float(
    np.mean(output_norms)
)

std_output_norm = float(
    np.std(output_norms)
)

preprocessing_consistency_error = float(
    np.mean(
        np.abs(out1 - out2)
    )
)

# ============================================
# network run
# ============================================

nodes = [
    Node(i)
    for i in range(10)
]

for t in range(400):
    step(nodes, t)

essences = np.array([
    e
    for n in nodes
    for e in n.essences
])

variances = np.var(
    essences,
    axis=0
)

mean_variance = float(
    np.mean(variances)
)

trust_range = float(
    np.max([n.trust for n in nodes])
    - np.min([n.trust for n in nodes])
)

# ============================================
# validation result
# ============================================

achieved = True

if preprocessing_consistency_error >= 0.05:
    achieved = False

if not (
    0.95
    <= mean_output_norm
    <= 1.05
):
    achieved = False

if np.any(np.isnan(out1)):
    achieved = False

if out1.shape != (8,):
    achieved = False

# ============================================
# output
# ============================================

print("metrics")

print(
    "mean_output_norm:",
    round(mean_output_norm, 6)
)

print(
    "std_output_norm:",
    round(std_output_norm, 6)
)

print(
    "preprocessing_consistency_error:",
    round(
        preprocessing_consistency_error,
        6
    )
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
    achieved
)

print(
    "final_result:",
    "achieved"
    if achieved
    else "not achieved"
)
