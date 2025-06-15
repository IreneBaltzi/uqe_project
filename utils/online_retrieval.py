import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random

def online_retrieval(data, f, batch_size=10, max_samples=200, seed=42, limit=None):
    """
    Algorithm 2 (Online Active Learning) with two stopping criteria:
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

    g_hat = np.random.rand(N)

    S = set()       
    labels = {}     

    def num_positives():
        return sum(labels[i] for i in S if labels[i] == 1)

    while len(S) < max_samples and (limit is None or num_positives() < limit):
        noise_scale = 1.0 / (1 + len(S))    
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=N)
        scores = g_hat + noise

        
        candidates = [i for i in range(N) if i not in S]
        if not candidates:
            break

        sorted_candidates = sorted(candidates, key=lambda i: scores[i], reverse=True)

        
        remaining_budget = max_samples - len(S)
        to_query = sorted_candidates[:min(batch_size, remaining_budget)]

        
        for i in to_query:
            S.add(i)
            labels[i] = f(data[i])

            if labels[i] == 1:
                print(f"Accepted positive row {i} — total positives: {num_positives()}")

            
            if limit is not None and num_positives() >= limit:
                break  
        if limit is not None and num_positives() >= limit:
            break  
        
        y_train = [labels[j] for j in S]
        if len(set(y_train)) >= 2:
            X_train = [embeddings[j] for j in S]
            clf = LogisticRegression(solver='liblinear')
            clf.fit(X_train, y_train)
            g_hat = clf.predict_proba(embeddings)[:, 1]
       
   
    positives = [i for i in S if labels[i] == 1]
  
    if limit is not None:
        positives = positives[:limit]

   
    return [data[i] for i in positives]
