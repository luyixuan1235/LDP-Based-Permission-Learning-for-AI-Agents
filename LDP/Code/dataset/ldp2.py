import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def laplace_noise(size, sensitivity, epsilon, seed=None):
    rng = np.random.default_rng(seed)
    scale = sensitivity / epsilon
    return rng.laplace(0, scale, size)

def select_nodes(nodes, beta, gamma, seed=42):
    random.seed(seed)
    total_num = len(nodes)
    attacker_num = int(total_num * beta)
    target_num = int(total_num * gamma)
    fake_nodes = random.sample(nodes, attacker_num)
    remaining = list(set(nodes) - set(fake_nodes))
    target_nodes = random.sample(remaining, target_num)
    return fake_nodes, target_nodes

def add_attack_edges(G, fake_nodes, target_nodes, attack_type, edges_per_fake, seed=42):
    random.seed(seed)
    nodes = list(G.nodes())
    for fake in fake_nodes:
        if attack_type == "randomValueAttack":
            possible_targets = [n for n in nodes if n != fake and not G.has_edge(fake, n)]
            targets = random.sample(possible_targets, min(edges_per_fake, len(possible_targets)))
        elif attack_type == "randomNodeAttack":
            targets = [random.choice(target_nodes)]
        elif attack_type == "maximumGainAttack":
            possible_targets = [n for n in target_nodes + fake_nodes if n != fake and not G.has_edge(fake, n)]
            random.shuffle(possible_targets)
            targets = possible_targets[:edges_per_fake]
        elif attack_type == "untargetedAttack":
            possible_targets = [n for n in nodes if n != fake and not G.has_edge(fake, n)]
            targets = random.sample(possible_targets, min(edges_per_fake, len(possible_targets)))
        else:
            raise ValueError("Unknown attack type")
        for t in targets:
            G.add_edge(fake, t)

def experiment(G_origin, beta, gamma, edges_per_fake, epsilons, attack_types, repeat=5):
    nodes = list(G_origin.nodes())
    results = {atype: [] for atype in attack_types}
    for epsilon in epsilons:
        for atype in attack_types:
            max_degrees = []
            for _ in range(repeat):
                G = G_origin.copy()
                fake_nodes, target_nodes = select_nodes(nodes, beta, gamma)
                add_attack_edges(G, fake_nodes, target_nodes, atype, edges_per_fake)
                degrees = np.array([G.degree(node) for node in nodes])
                noisy_degrees = degrees + laplace_noise(len(degrees), 2.0, epsilon)
                # Get the maximum noisy degree among target nodes
                target_indices = [nodes.index(t) for t in target_nodes]
                max_noisy_degree = np.max(noisy_degrees[target_indices])
                max_degrees.append(max_noisy_degree)
            avg_max_degree = np.mean(max_degrees)
            results[atype].append(avg_max_degree)
    return results

def plot_results(epsilons, results):
    plt.figure(figsize=(8,6))
    for atype, max_degrees in results.items():
        plt.plot(epsilons, max_degrees, marker='o', label=atype)
    plt.xlabel("Epsilon (privacy parameter)")
    plt.ylabel("Max noisy degree of target nodes")
    plt.title("Max noisy degree of target nodes vs. Epsilon for different attacks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Load real Google+ data (or replace with your dataset)
    # Make sure the file path is correct!
    G = nx.read_edgelist('dataset/gplus_combined.txt', nodetype=int)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    beta = 0.05  # Fraction of attacker nodes
    gamma = 0.05  # Fraction of target nodes
    edges_per_fake = 10  # Number of edges each fake node adds
    epsilons = [1, 2, 3, 4, 5, 6, 7, 8]
    attack_types = ["randomValueAttack", "randomNodeAttack", "maximumGainAttack", "untargetedAttack"]
    results = experiment(G, beta, gamma, edges_per_fake, epsilons, attack_types, repeat=5)
    plot_results(epsilons, results)

if __name__ == "__main__":
    main()