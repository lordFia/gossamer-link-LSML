import numpy as np
import random

NUM_ROLES = 3
MAX_CONNECTIONS = 4
SEARCH_K = 5
ESSENCE_SLOTS = 3

class Node:
    def __init__(self, id, dim=8):
        self.id = id
        self.essences = [self._normalize(np.random.randn(dim)) for _ in range(ESSENCE_SLOTS)]
        self.trust = 0.5
        self.role = np.random.randint(0, NUM_ROLES)
        self.connections = set()
        self.in_deg = 0

    def _normalize(self, x):
        return x / (np.linalg.norm(x) + 1e-8)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def essence_similarity(node_a, node_b):
    return max(
        cosine_similarity(ea, eb)
        for ea in node_a.essences
        for eb in node_b.essences
    )

# ★ スロット間の分離（少し弱める：過剰分離防止）
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

def search_nodes(node, nodes):
    candidates = [n for n in nodes if n.id != node.id]
    sampled = random.sample(candidates, min(SEARCH_K, len(candidates)))

    scored = []
    for n in sampled:
        sim = essence_similarity(node, n)
        score = 0.6 * n.trust + 0.4 * sim
        scored.append((score, n))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [n for _, n in scored]

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

    # ★ 相対差を強化
    rel = scores - np.mean(scores)
    rel /= (np.std(rel) + 1e-8)

    for i, node in enumerate(nodes):
        node.trust = float(np.clip(
            0.6 * node.trust +   # 元を少し残す
            0.4 * (0.5 + 0.3 * rel[i]),  # 差を強く反映
            0.05, 1.0
        ))

def propagate(node, neighbors):
    if not neighbors:
        return

    for n in neighbors:
        for i in range(len(n.essences)):
            src = random.choice(node.essences)
            sim = cosine_similarity(n.essences[i], src)

            if sim < 0.6:
                n.essences[i] += 0.05 * (src - n.essences[i])
            elif sim > 0.85:
                n.essences[i] -= 0.05 * (src - n.essences[i])

            n.essences[i] /= (np.linalg.norm(n.essences[i]) + 1e-8)

def step(nodes):
    update_connections(nodes)
    neighbors_dict = {n.id: get_neighbors(n, nodes) for n in nodes}
    update_trust(nodes, neighbors_dict)

    for node in nodes:
        propagate(node, neighbors_dict[node.id])
        enforce_structure(node)

nodes = [Node(i) for i in range(10)]

for _ in range(400):
    step(nodes)

all_ess = []
for n in nodes:
    all_ess.extend(n.essences)

essences = np.array(all_ess)
variances = np.var(essences, axis=0)

print("平均分散:", float(np.mean(variances)))
print("信頼スコア:", [round(n.trust, 3) for n in nodes])
print("役割:", [n.role for n in nodes])
print("接続:", [list(n.connections) for n in nodes])
