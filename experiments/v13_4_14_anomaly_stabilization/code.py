# v13.4.14 anomaly stabilization + variance clamp (critical fix)

import numpy as np
import hashlib
import random

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

# variance control
TARGET_VAR_MAX = 0.06
UPDATE_SCALE = 0.5  # update damping


class Node:
    def __init__(self, id, dim=8):
        self.id = id
        self.essences = np.random.randn(ESSENCE_SLOTS, dim)
        self.essences /= (np.linalg.norm(self.essences, axis=1, keepdims=True) + 1e-8)
        self.trust = 0.5
        self.role = np.random.randint(0, NUM_ROLES)
        self.connections = set()
        self.age = 1
        self.nonce = 0
        self.last_active = 0
        self.anomaly_score = 0.0
        self.is_anomaly = False


def compute_hash(val):
    return hashlib.sha256(str(val).encode()).hexdigest()


def valid_pow(node):
    return compute_hash((node.id, node.nonce)).startswith("0" * POW_DIFFICULTY)


def mine_pow(node):
    if valid_pow(node):
        return
    while not valid_pow(node):
        node.nonce += 1


def sim(a, b):
    return np.max(np.dot(a, b.T))


# ----------------------
# anomaly detection
# ----------------------
def detect_anomaly(nodes):
    vals = np.array([n.trust for n in nodes])
    mean = np.mean(vals)
    std = np.std(vals) + 1e-8

    for i, n in enumerate(nodes):
        z = abs((vals[i] - mean) / std)
        n.anomaly_score = float(z)
        n.is_anomaly = z > ANOMALY_Z


# ----------------------
# trust update
# ----------------------
def update_trust(nodes):

    scores = []
    for n in nodes:
        neighbors = [nodes[i] for i in n.connections]
        if neighbors:
            sims = [sim(n.essences, nb.essences) for nb in neighbors]
            scores.append(np.mean(sims))
        else:
            scores.append(0.5)

    scores = np.array(scores)

    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(nodes))

    target = ranks / (len(nodes) - 1 + 1e-8)
    target = MIN_TRUST + (MAX_TRUST - MIN_TRUST) * target

    mean_trust = np.mean([n.trust for n in nodes])

    for i, n in enumerate(nodes):

        anomaly_scale = 1.0 - ANOMALY_WEIGHT * (n.anomaly_score / (1 + n.anomaly_score))

        new_trust = (
            (1 - TRUST_DECAY) * n.trust
            + 0.55 * (target[i] - n.trust)
            + 0.12 * (n.trust - mean_trust)
        )

        new_trust *= anomaly_scale

        if new_trust <= MIN_TRUST:
            new_trust = MIN_TRUST + 0.02 * random.random()

        n.trust = float(np.clip(new_trust, MIN_TRUST, MAX_TRUST))


# ----------------------
# essence update
# ----------------------
def update_essences(nodes):

    for node in nodes:
        neighbors = [nodes[i] for i in node.connections]
        if not neighbors:
            continue

        for i in range(ESSENCE_SLOTS):
            src = random.choice(neighbors).essences[random.randrange(ESSENCE_SLOTS)]

            sim_val = np.dot(node.essences[i], src)

            # damped update
            if sim_val < 0.6:
                node.essences[i] += UPDATE_SCALE * 0.08 * (src - node.essences[i])
            elif sim_val > 0.85:
                node.essences[i] -= UPDATE_SCALE * 0.04 * (src - node.essences[i])

            node.essences[i] /= (np.linalg.norm(node.essences[i]) + 1e-8)


# ----------------------
# variance clamp
# ----------------------
def clamp_variance(nodes):

    all_vecs = np.array([e for n in nodes for e in n.essences])
    var = np.var(all_vecs, axis=0)
    mean_var = np.mean(var)

    if mean_var > TARGET_VAR_MAX:
        scale = np.sqrt(TARGET_VAR_MAX / (mean_var + 1e-8))
        for n in nodes:
            n.essences *= scale


# ----------------------
# step
# ----------------------
def step(nodes, t):

    for n in nodes:
        n.age += 1
        mine_pow(n)

    detect_anomaly(nodes)

    for node in nodes:
        candidates = [n for n in nodes if n.id != node.id]

        scored = []
        for c in candidates:
            s = sim(node.essences, c.essences)
            score = 0.6 * c.trust + 0.4 * s
            scored.append((score, c))

        scored.sort(reverse=True, key=lambda x: x[0])

        selected = []
        for _, c in scored:
            if len(selected) >= SEARCH_K:
                break
            if all(sim(c.essences, s.essences) < 0.7 for s in selected):
                selected.append(c)

        while len(selected) < SEARCH_K:
            c = random.choice(candidates)
            if c not in selected:
                selected.append(c)

        node.connections = {c.id for c in selected[:MAX_CONNECTIONS]}

    update_trust(nodes)
    update_essences(nodes)
    clamp_variance(nodes)


# ----------------------
# run
# ----------------------
nodes = [Node(i) for i in range(10)]

for t in range(400):
    step(nodes, t)

essences = np.array([e for n in nodes for e in n.essences])
variances = np.var(essences, axis=0)

print("mean_variance:", float(np.mean(variances)))
print("trust:", [round(n.trust, 3) for n in nodes])
print("anomaly_score:", [round(n.anomaly_score, 3) for n in nodes])
print("connections:", [list(n.connections) for n in nodes])
