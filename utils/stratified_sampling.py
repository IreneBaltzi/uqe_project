import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import random

def stratified_sampling(data, f, K=10, sampling_ratio=0.1, seed=42, return_rows=False):
    """
    Estimate E[f(Ti)] via stratified sampling (UQE Alg. 1).

    Args:
        data (List[Dict]): Each row must include a precomputed 'embedding'.
        f (Callable[[Dict], float]): Oracle function (e.g. LLM call) returning 0/1 or a real value.
        K (int): Number of clusters.
        sampling_ratio (float): Fraction of each cluster to sample.
        seed (int): Random seed for reproducibility.
        return_rows (bool): If True, also return the list of sampled rows and their weights.

    Returns:
        If return_rows=False:
            float: Estimated mean ∑₍i∈S₎ wᵢ f(data[i]) / ∑₍i∈S₎ wᵢ  
        If return_rows=True:
            (estimate, sampled_rows, sampled_weights)
            - estimate (float): as above.
            - sampled_rows (List[Dict]): the actual row dicts sampled.
            - sampled_weights (List[float]): the corresponding wᵢ for each sampled row.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Step 1: Extract embeddings into NumPy array
    embeddings = np.array([row['embedding'] for row in data])

    # Step 2: Cluster into K groups
    kmeans = KMeans(n_clusters=K, random_state=seed).fit(embeddings)
    labels = kmeans.labels_

    # Attach the cluster label to each row
    for i, row in enumerate(data):
        row['cluster'] = int(labels[i])

    # Step 3: Build cluster → list of indices
    cluster_to_indices = defaultdict(list)
    for i, row in enumerate(data):
        cluster_to_indices[row['cluster']].append(i)

    # Step 4: Sample from each cluster
    sampled_indices = []
    for cluster, indices in cluster_to_indices.items():
        n_total = len(indices)
        # Number to sample from this cluster
        n_sample = max(1, int(n_total * sampling_ratio))
        # If the cluster is tiny, we still sample at least one
        chosen = random.sample(indices, min(n_sample, n_total))
        sampled_indices.extend(chosen)

    S = sampled_indices  # the set of sampled indices
    weights = {}         # index → weight

    # Count how many sampled per cluster
    cluster_counts_in_sample = defaultdict(int)
    for i in S:
        c = data[i]['cluster']
        cluster_counts_in_sample[c] += 1

    # Step 5: Compute each sampled index’s weight wᵢ = |Cₖ| / |S ∩ Cₖ|
    for i in S:
        c = data[i]['cluster']
        total_in_cluster  = len(cluster_to_indices[c])
        sampled_in_cluster = cluster_counts_in_sample[c]
        weights[i] = total_in_cluster / sampled_in_cluster

    # Step 6: Evaluate f(row) on each sampled index and form weighted estimate
    weighted_sum = 0.0
    weight_total = 0.0
    for i in S:
        w_i = weights[i]
        f_i = f(data[i])
        weighted_sum += w_i * f_i
        weight_total += w_i

    estimate = weighted_sum / weight_total

    if not return_rows:
        return estimate

    # If return_rows=True, also return the list of sampled rows and weights
    sampled_rows = [data[i] for i in S]
    sampled_weights = [weights[i] for i in S]
    return estimate, sampled_rows, sampled_weights
