# v13.2.0 PoW (proof of work for influence control)

import numpy as np
import random
import hashlib

NUM_ROLES = 3
MAX_CONNECTIONS = 4
SEARCH_K = 5
ESSENCE_SLOTS = 3
QUERY_K = 2  # number of query targets

POW_DIFFICULTY = 2  # number of leading zeros required

class Node:
    def __init__(self, id, dim=8):
        self.id = id
        self.essences = [self._normalize(np.random.randn(dim)) for _ in range(ESSENCE_SLOTS)]
        self.trust = 0.5
        self.role = np.random.randint(0, NUM_ROLES)
        self.connections = set()
        self.in_deg = 0
        self.age = 1
        self.nonce = 0  # PoW nonce

    def _normalize(self, x):
        return x / (np.linalg.norm(x) + 1e-8)

def compute_hash(val):
    return hashlib.sha256(str(val).encode()).hexdigest()

def valid_pow(node):
    h = compute_hash((node.id, node.nonce))
    return h.startswith("0" * POW_DIFFICULTY)

def mine_pow(node):
    while not valid_pow(node):
        node.nonce += 1

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def essence_similarity(node_a, node_b):
    return max(
        cosine_similarity(ea, eb)
        for ea in node_a.essences
        for eb in node_b.essences
    )

def query_nodes(node, nodes):
    candidates = [n for n in nodes if n.id != node.id]

    scored = []
    for n in candidates:
        sim = essence_similarity(node, n)
        novelty = 1.0 - sim

        age_factor = np.log(n.age + 1)
        pow_factor = 2.0 if valid_pow(n) else 0.5

        score = (0.5 * n.trust + 0.5 * novelty) * age_factor * pow_factor
        scored.append((score, n))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [n for _, n in scored[:QUERY_K]]

def extract(node, sources):
    if not sources:
        return

    for i in range(len(node.essences)):
        best_vec = None
        best_score = -1.0

        for src in sources:
            for e in src.essences:
                sim = cosine_similarity(node.essences[i], e)
                score = (1.0 - sim) * src.trust

                if score > best_score:
                    best_score = score
                    best_vec = e

        if best_vec is not None:
            sim = cosine_similarity(node.essences[i], best_vec)

            if sim < 0.6:
                node.essences[i] += 0.12 * (best_vec - node.essences[i])
            else:
                node.essences[i] -= 0.06 * (best_vec - node.essences[i])

            node.essences[i] /= (np.linalg.norm(node.essences[i]) + 1e-8)

def propagate(node, neighbors):
    if not neighbors:
        return

    weights = np.array([n.trust for n in neighbors])
    weights = weights / (np.sum(weights) + 1e-8)

    for n, w in zip(neighbors, weights):
        for i in range(len(n.essences)):
            src = random.choice(node.essences)
            sim = cosine_similarity(n.essences[i], src)

            if sim < 0.6:
                n.essences[i] += w * 0.1 * (src - n.essences[i])
            elif sim > 0.85:
                n.essences[i] -= w * 0.05 * (src - n.essences[i])

            n.essences[i] /= (np.linalg.norm(n.essences[i]) + 1e-8)

def search_nodes(node, nodes):
    candidates = [n for n in nodes if n.id != node.id]

    scored = []
    for n in candidates:
        sim = essence_similarity(node, n)
        age_factor = np.log(n.age + 1)
        pow_factor = 2.0 if valid_pow(n) else 0.5

        score = (0.6 * n.trust + 0.4 * sim) * age_factor * pow_factor
        scored.append((score, n))

    scored.sort(reverse=True, key=lambda x: x[0])

    selected = []
    for _, n in scored:
        if len(selected) >= SEARCH_K:
            break
        if all(essence_similarity(n, s) < 0.8 for s in selected):
            selected.append(n)

    while len(selected) < SEARCH_K:
        cand = random.choice(candidates)
        if cand not in selected:
            selected.append(cand)

    return selected

def compress_essences(node):
    new_ess = []
    for e in node.essences:
        if all(cosine_similarity(e, ne) < 0.85 for ne in new_ess):
            new_ess.append(e)

    while len(new_ess) < ESSENCE_SLOTS:
        base = random.choice(new_ess) if new_ess else np.random.randn(len(node.essences[0]))
        noise = np.random.randn(len(base)) * 0.05
        vec = base + noise
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        new_ess.append(vec)

    node.essences = new_ess[:ESSENCE_SLOTS]

def enforce_structure(node):
    for i in range(len(node.essences)):
        for j in range(i+1, len(node.essences)):
            sim = cosine_similarity(node.essences[i], node.essences[j])
            if sim > 0.75:
                diff = node.essences[i] - node.essences[j]
                node.essences[i] += 0.05 * diff
                node.essences[j] -= 0.05 * diff
                node.essences[i] /= (np.linalg.norm(node.essences[i]) + 1e-8)
                node.essences[j] /= (np.linalg.norm(node.essences[j]) + 1e-8)

def update_connections(nodes):
    for n in nodes:
        n.in_deg = 0

    incoming = {n.id: 0 for n in nodes}

    for node in nodes:
        targets = search_nodes(node, nodes)[:MAX_CONNECTIONS]
        node.connections = set([n.id for n in targets])

        for t in node.connections:
            incoming[t] += 1
            nodes[t].in_deg += 1

    for node in nodes:
        adjusted = set()
        for t in node.connections:
            penalty = incoming[t] / len(nodes)
            if random.random() > penalty:
                adjusted.add(t)

        while len(adjusted) < MAX_CONNECTIONS:
            cand = [n.id for n in nodes if n.id != node.id and n.id not in adjusted]
            if not cand:
                break
            adjusted.add(random.choice(cand))

        node.connections = adjusted

def get_neighbors(node, nodes):
    return [nodes[i] for i in node.connections]

def update_trust(nodes, neighbors_dict):
    scores = []

    for node in nodes:
        neighbors = neighbors_dict[node.id]
        sims = [essence_similarity(node, n) for n in neighbors] if neighbors else [0.5]
        scores.append(np.mean(sims))

    scores = np.array(scores)
    rel = scores - np.mean(scores)
    rel /= (np.std(rel) + 1e-8)

    for i, node in enumerate(nodes):
        node.trust = float(np.clip(
            0.6 * node.trust +
            0.4 * (0.5 + 0.3 * rel[i]),
            0.05, 1.0
        ))

def step(nodes):
    for n in nodes:
        n.age += 1
        mine_pow(n)

    update_connections(nodes)
    neighbors_dict = {n.id: get_neighbors(n, nodes) for n in nodes}
    update_trust(nodes, neighbors_dict)

    for node in nodes:
        sources = query_nodes(node, nodes)
        extract(node, sources)
        propagate(node, neighbors_dict[node.id])
        enforce_structure(node)
        compress_essences(node)

nodes = [Node(i) for i in range(10)]

for _ in range(400):
    step(nodes)

all_ess = []
for n in nodes:
    all_ess.extend(n.essences)

essences = np.array(all_ess)
variances = np.var(essences, axis=0)

print("mean_variance:", float(np.mean(variances)))
print("trust:", [round(n.trust, 3) for n in nodes])
print("roles:", [n.role for n in nodes])
print("connections:", [list(n.connections) for n in nodes])
