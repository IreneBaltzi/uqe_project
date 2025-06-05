import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random

def online_retrieval(data, f, batch_size=10, max_samples=200, seed=42, limit=None):
    """
    Implements Algorithm 2 (Online Active Learning) with two stopping criteria:
      1. Stop once we have made `max_samples` LLM calls (labels).
      2. Stop early if we have already found `limit` positives (when limit is not None).

    Args:
        data (List[Dict]): Dataset; each row must have a key 'embedding'.
        f (Callable): Function that returns 0/1 (simulating LLM oracle).
        batch_size (int): How many rows to query per iteration.
        max_samples (int): Maximum number of rows to ever query (budget).
        seed (int): Random seed for reproducibility.
        limit (int or None): If set, stop as soon as we have found `limit` rows with f(row)==1.

    Returns:
        List[Dict]: All rows for which f(row)==1 (but truncated to `limit` if specified).
    """
    random.seed(seed)
    np.random.seed(seed)

    N = len(data)
    embeddings = np.array([row['embedding'] for row in data])
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    # Step 2: random initial surrogate scores
    g_hat = np.random.rand(N)

    S = set()       # indices we have queried so far
    labels = {}     # dictionary: index -> f(data[index])

    # Helper to count how many positives we have so far
    def num_positives():
        return sum(labels[i] for i in S if labels[i] == 1)

    # Step 3: loop until we exhaust budget OR find enough positives
    while len(S) < max_samples and (limit is None or num_positives() < limit):
        # Add a small, decaying noise term for exploration:
        noise_scale = 1.0 / (1 + len(S))    # decays as we query more
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=N)
        scores = g_hat + noise

        # Choose all indices not yet in S, sorted by descending (g_hat + noise)
        candidates = [i for i in range(N) if i not in S]
        if not candidates:
            break

        sorted_candidates = sorted(candidates, key=lambda i: scores[i], reverse=True)

        # Don’t overshoot the budget: only pick up to (max_samples - len(S))
        remaining_budget = max_samples - len(S)
        to_query = sorted_candidates[:min(batch_size, remaining_budget)]

        # Query f(row) on each selected index
        for i in to_query:
            S.add(i)
            labels[i] = f(data[i])

            # If we have reached the desired `limit` of positives, stop immediately
            if limit is not None and num_positives() >= limit:
                break  # break out of the inner for‐loop

        # Step 4: if we have seen at least one positive & one negative, retrain surrogate
        y_train = [labels[j] for j in S]
        if len(set(y_train)) >= 2:
            X_train = [embeddings[j] for j in S]
            clf = LogisticRegression(solver='liblinear')
            clf.fit(X_train, y_train)
            g_hat = clf.predict_proba(embeddings)[:, 1]
        # else: keep using the old g_hat until we have both classes

    # Step 5: collect all indices i in S where labels[i] == 1
    positives = [i for i in S if labels[i] == 1]
    # If a limit was specified, truncate to that many
    if limit is not None:
        positives = positives[:limit]

    # Return the actual row objects
    return [data[i] for i in positives]
