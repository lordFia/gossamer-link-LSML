import numpy as np
import random

NUM_ROLES = 3
MAX_CONNECTIONS = 4

class Node:
    def __init__(self, id, dim=8):
        self.id = id
        self.essence = self._normalize(np.random.randn(dim))
        self.trust = 0.5
        self.role = np.random.randint(0, NUM_ROLES)
        self.usage = 0.0
        self.connections = set()

    def _normalize(self, x):
        return x / (np.linalg.norm(x) + 1e-8)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def update_connections(nodes):
    incoming_count = {n.id: 0 for n in nodes}

    for node in nodes:
        candidates = [n for n in nodes if n.id != node.id]

        scored = []
        for n in candidates:
            sim = cosine_similarity(node.essence, n.essence)
            trust = n.trust
            score = 0.7 * trust + 0.3 * sim
            scored.append((score, n))

        scored.sort(reverse=True, key=lambda x: x[0])
        node.connections = set([n.id for _, n in scored[:MAX_CONNECTIONS]])

        for cid in node.connections:
            incoming_count[cid] += 1

    for node in nodes:
        adjusted = set()
        for cid in node.connections:
            penalty = incoming_count[cid] / len(nodes)
            if random.random() > penalty:
                adjusted.add(cid)

        while len(adjusted) < MAX_CONNECTIONS:
            candidates = [n.id for n in nodes if n.id != node.id and n.id not in adjusted]
            if not candidates:
                break
            adjusted.add(random.choice(candidates))

        node.connections = adjusted

def get_neighbors(node, nodes):
    return [nodes[i] for i in node.connections]

def repulsion(a, b, threshold=0.75, strength=0.5):
    sim = cosine_similarity(a, b)
    if sim > threshold:
        diff = a - b
        return strength * diff / (np.linalg.norm(diff) + 1e-8)
    return np.zeros_like(a)

def attraction(a, b, sim, role_same, eta=0.04):
    if sim < 0.6:
        if role_same:
            return eta * (b - a)
        else:
            return -eta * (b - a) * 0.3
    return np.zeros_like(a)

def add_noise(vec, scale=0.02):
    noise = np.random.randn(*vec.shape) * scale
    vec = vec + noise
    return vec / (np.linalg.norm(vec) + 1e-8)

def update_role(node, nodes, neighbors):
    role_counts = {r: 0 for r in range(NUM_ROLES)}
    for n in nodes:
        role_counts[n.role] += 1

    scores = []
    for r in range(NUM_ROLES):
        sims = [cosine_similarity(node.essence, n.essence) for n in neighbors if n.role == r]
        sim_score = np.mean(sims) if sims else 0.0
        balance_bonus = 1.0 / (role_counts[r] + 1e-6)
        scores.append(sim_score + 0.5 * balance_bonus)

    node.role = int(np.argmax(scores))

def compute_quality(node, neighbors):
    if not neighbors:
        return 0.5
    sims = np.array([cosine_similarity(node.essence, n.essence) for n in neighbors])
    diversity = 1.0 - np.mean(sims)
    trust_neighbors = np.mean([n.trust for n in neighbors])
    return 0.7 * diversity + 0.3 * trust_neighbors

def compute_reputation(node, neighbors):
    if not neighbors:
        return 0.5
    sims = np.array([cosine_similarity(node.essence, n.essence) for n in neighbors])
    weights = np.array([n.trust for n in neighbors])
    return float(np.mean(weights * sims) - np.mean(sims))

def update_usage(nodes, neighbors_dict):
    usages = []
    for node in nodes:
        neighbors = neighbors_dict[node.id]
        sims = [cosine_similarity(node.essence, n.essence) for n in neighbors]
        usage = np.mean(sims) if sims else 0.0
        usages.append(usage)

    usages = np.array(usages)
    usages -= np.mean(usages)

    for i, node in enumerate(nodes):
        node.usage = usages[i]

def optimize_coefficients(nodes):
    trusts = np.array([n.trust for n in nodes])
    var = np.var(trusts)

    alpha = np.clip(0.3 + var * 2.0, 0.3, 1.0)
    beta  = np.clip(0.2 + var, 0.2, 0.6)
    gamma = np.clip(0.2 + var, 0.2, 0.6)

    return alpha, beta, gamma

def update_trust(nodes, neighbors_dict,
                 lam=0.05, recovery=0.05):

    alpha, beta, gamma = optimize_coefficients(nodes)

    deltas = []

    for node in nodes:
        neighbors = neighbors_dict[node.id]

        Q = compute_quality(node, neighbors)
        R = compute_reputation(node, neighbors)
        U = node.usage

        Q_neighbors = np.array([compute_quality(n, neighbors) for n in neighbors]) if neighbors else np.array([0.5])
        Q_mean = np.mean(Q_neighbors)

        delta = (Q - Q_mean) + beta * R + gamma * U
        deltas.append(delta)

    deltas = np.array(deltas)
    deltas -= np.mean(deltas)

    for i, node in enumerate(nodes):
        new_T = (1 - lam) * node.trust + alpha * deltas[i]

        if new_T < 0.1:
            new_T += recovery

        node.trust = float(np.clip(new_T, 0.05, 1.0))

def propagate(node, neighbors):
    if not neighbors:
        return

    weights = np.array([n.trust for n in neighbors])
    weights = weights ** 2
    weights = weights / (np.sum(weights) + 1e-8)

    for n, w in zip(neighbors, weights):
        sim = cosine_similarity(node.essence, n.essence)
        same_role = (node.role == n.role)

        n.essence += w * attraction(n.essence, node.essence, sim, same_role)
        n.essence += w * repulsion(n.essence, node.essence)
        n.essence = add_noise(n.essence)

def step(nodes):
    update_connections(nodes)

    neighbors_dict = {}
    for node in nodes:
        neighbors_dict[node.id] = get_neighbors(node, nodes)

    for node in nodes:
        update_role(node, nodes, neighbors_dict[node.id])

    update_usage(nodes, neighbors_dict)

    update_trust(nodes, neighbors_dict)

    for node in nodes:
        propagate(node, neighbors_dict[node.id])

nodes = [Node(i) for i in range(10)]

for t in range(400):
    step(nodes)

essences = np.array([n.essence for n in nodes])
variances = np.var(essences, axis=0)

print("平均分散:", float(np.mean(variances)))
print("信頼スコア:", [round(float(n.trust), 3) for n in nodes])
print("役割:", [n.role for n in nodes])
print("接続:", [list(n.connections) for n in nodes])
