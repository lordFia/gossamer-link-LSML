import numpy as np
import random

NUM_ROLES = 3
MAX_CONNECTIONS = 4
SEARCH_K = 5

class Node:
    def __init__(self, id, dim=8):
        self.id = id
        self.essence = self._normalize(np.random.randn(dim))
        self.trust = 0.5
        self.role = np.random.randint(0, NUM_ROLES)
        self.usage = 0.0
        self.connections = set()
        self.in_deg = 0

    def _normalize(self, x):
        return x / (np.linalg.norm(x) + 1e-8)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def search_nodes(node, nodes):
    candidates = [n for n in nodes if n.id != node.id]
    sampled = random.sample(candidates, min(SEARCH_K, len(candidates)))

    scored = []
    for n in sampled:
        sim = cosine_similarity(node.essence, n.essence)
        trust = n.trust
        noise = random.uniform(0.9, 1.1)
        score = (0.6 * trust + 0.4 * sim) * noise
        scored.append((score, n))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [n for _, n in scored]

def update_connections(nodes):
    for n in nodes:
        n.in_deg = 0

    incoming_count = {n.id: 0 for n in nodes}

    for node in nodes:
        searched = search_nodes(node, nodes)
        node.connections = set([n.id for n in searched[:MAX_CONNECTIONS]])

        for cid in node.connections:
            incoming_count[cid] += 1
            nodes[cid].in_deg += 1

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

# ===== 修正認証（相対評価）=====
def compute_auth(node, neighbors):
    if not neighbors:
        return 0.5

    sims = np.array([cosine_similarity(node.essence, n.essence) for n in neighbors])
    trust = np.array([n.trust for n in neighbors])

    consistency = np.mean(sims)
    trust_alignment = np.mean(trust)

    raw = 0.5 * consistency + 0.5 * trust_alignment

    return raw

def update_trust(nodes, neighbors_dict, lam=0.05):
    auths = []

    # 全ノードのauth取得
    for node in nodes:
        neighbors = neighbors_dict[node.id]
        auth = compute_auth(node, neighbors)
        auths.append(auth)

    auths = np.array(auths)

    # ★ 相対化（平均との差分）
    auths -= np.mean(auths)

    for i, node in enumerate(nodes):
        new_T = (1 - lam) * node.trust + 0.8 * auths[i]

        if new_T < 0.1:
            new_T += 0.05

        node.trust = float(np.clip(new_T, 0.05, 1.0))

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

def propagate(node, neighbors, nodes):
    if not neighbors:
        return

    weights = []
    for n in neighbors:
        w = 1.0 / (1 + n.in_deg)
        weights.append(w)

    weights = np.array(weights)
    weights = weights / (np.sum(weights) + 1e-8)

    for n, w in zip(neighbors, weights):
        sim = cosine_similarity(node.essence, n.essence)
        same_role = (node.role == n.role)

        n.essence += w * attraction(n.essence, node.essence, sim, same_role)
        n.essence += w * repulsion(n.essence, node.essence)
        n.essence = add_noise(n.essence)

    target = random.choice(nodes)
    if target.id != node.id:
        target.essence += 0.1 * node.essence
        target.essence = target.essence / (np.linalg.norm(target.essence) + 1e-8)

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
        propagate(node, neighbors_dict[node.id], nodes)

nodes = [Node(i) for i in range(10)]

for t in range(400):
    step(nodes)

essences = np.array([n.essence for n in nodes])
variances = np.var(essences, axis=0)

print("平均分散:", float(np.mean(variances)))
print("信頼スコア:", [round(float(n.trust), 3) for n in nodes])
print("役割:", [n.role for n in nodes])
print("接続:", [list(n.connections) for n in nodes])
