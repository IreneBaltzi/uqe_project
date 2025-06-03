import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import random


def stratified_sampling(data, f, K=10, sampling_ratio=0.1, seed=42, return_rows=False):
    """
    Estimate E[f(Ti, cond)] using stratified sampling as described in UQE Algorithm 1.

    Args:
        data (List[Dict]): List of rows. Each row must include a precomputed 'embedding'.
        f (Callable[[Dict], float]): Function to evaluate f(Ti, cond). Typically calls an LLM.
        K (int): Number of clusters.
        sampling_ratio (float): Fraction of each cluster to sample.
        seed (int): Random seed for reproducibility.

    Returns:
        float: Estimated expectation of f over the dataset.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Step 1: Extract embeddings
    embeddings = np.array([row['embedding'] for row in data])

    # Step 2: Cluster embeddings into K groups
    kmeans = KMeans(n_clusters=K, random_state=seed).fit(embeddings)
    labels = kmeans.labels_

    for i, row in enumerate(data):
        row['cluster'] = int(labels[i])

    # Step 3: Stratified sampling
    cluster_to_indices = defaultdict(list)
    for i, row in enumerate(data):
        cluster_to_indices[row['cluster']].append(i)

    sampled_indices = []
    for cluster, indices in cluster_to_indices.items():
        n_sample = max(1, int(len(indices) * sampling_ratio))
        sampled = random.sample(indices, min(n_sample, len(indices)))
        sampled_indices.extend(sampled)

    S = sampled_indices
    weights = {}

    # Step 4: Compute weights and evaluate f
    cluster_counts_in_sample = defaultdict(int)
    for i in S:
        cluster_counts_in_sample[data[i]['cluster']] += 1

    for i in S:
        cluster = data[i]['cluster']
        total_in_cluster = len(cluster_to_indices[cluster])
        sampled_in_cluster = cluster_counts_in_sample[cluster]
        weights[i] = total_in_cluster / sampled_in_cluster

    # Step 5: Compute weighted estimate
    weighted_sum = 0.0
    weight_total = 0.0
    for i in S:
        wi = weights[i]
        fi = f(data[i])
        weighted_sum += wi * fi
        weight_total += wi

    estimate = weighted_sum / weight_total
    return (estimate, [data[i] for i in S]) if return_rows else estimate
