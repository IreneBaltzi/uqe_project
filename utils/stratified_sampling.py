# --- stratified_sampling.py (Refactored) ---
import os
import json
import numpy as np
import faiss
from collections import defaultdict


def compute_and_save_clusters(embeddings_jsonl: str, num_clusters: int, output_cluster_file: str):
    embeddings_list = []
    with open(embeddings_jsonl, 'r') as f:
        for line in f:
            record = json.loads(line)
            emb = record.get('embedding')
            if emb is None:
                raise ValueError("Missing 'embedding' in record")
            embeddings_list.append(emb)

    embeddings = np.asarray(embeddings_list, dtype='float32')
    N, D = embeddings.shape

    kmeans = faiss.Kmeans(d=D, k=num_clusters, niter=20, verbose=True, gpu=False)
    kmeans.train(embeddings)
    centroids = kmeans.centroids

    index = faiss.IndexFlatL2(D)
    index.add(centroids)
    _, assignments = index.search(embeddings, 1)
    assignments = assignments.flatten().tolist()

    with open(output_cluster_file, 'w') as out_f:
        json.dump({
            'assignments': assignments,
            'centroids': centroids.tolist()
        }, out_f)


def stratified_sample_and_weight(data, cluster_file, sampling_ratio, seed=42):
    np.random.seed(seed)
    with open(cluster_file, 'r') as f:
        clusters = json.load(f)
    assignments = np.array(clusters['assignments'])
    num_clusters = len(clusters['centroids'])

    cluster_to_indices = defaultdict(list)
    for i, c in enumerate(assignments):
        cluster_to_indices[c].append(i)

    sampled_indices = []
    for c, indices in cluster_to_indices.items():
        n_sample = max(1, int(len(indices) * sampling_ratio))
        chosen = np.random.choice(indices, min(n_sample, len(indices)), replace=False)
        sampled_indices.extend(chosen)

    cluster_counts_in_sample = defaultdict(int)
    for i in sampled_indices:
        c = assignments[i]
        cluster_counts_in_sample[c] += 1

    weights = {}
    for i in sampled_indices:
        c = assignments[i]
        weights[i] = len(cluster_to_indices[c]) / cluster_counts_in_sample[c]

    sampled_rows = [data[i] for i in sampled_indices]
    sampled_weights = [weights[i] for i in sampled_indices]
    return sampled_rows, sampled_weights, assignments


if __name__ == "__main__":
    EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "/data/imdb_embed_clustering.json")
    CLUSTERED_OUTPUT = os.getenv("ASSIGNMENTS_FILE", "/data/imdb_clustered_data.json")
    NUM_CLUSTERS = int(os.getenv("K_CLUSTERS", "50"))

    print(f"Clustering {EMBEDDINGS_FILE} into {NUM_CLUSTERS} clusters...")
    compute_and_save_clusters(EMBEDDINGS_FILE, NUM_CLUSTERS, CLUSTERED_OUTPUT)
    print(f"Cluster info saved to {CLUSTERED_OUTPUT}")
