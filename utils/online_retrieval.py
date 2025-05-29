import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random

def online_retrieval(data, f, batch_size=10, max_samples=200, seed=42):
    """
    Implements Algorithm 2: Online Active Learning for Non-Aggregation Retrieval.

    Args:
        data (List[Dict]): Dataset with 'embedding'
        f (Callable): Ground-truth or simulated LLM function
        batch_size (int): Number of samples per iteration
        max_samples (int): Maximum number of samples to query
        seed (int): Random seed

    Returns:
        List[Dict]: List of rows where f(row) == 1 (predicted matches)
    """
    random.seed(seed)
    np.random.seed(seed)

    N = len(data)
    embeddings = np.array([row['embedding'] for row in data])
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    g_hat = np.random.rand(N)  # Step 2: random initial scores
    S = set()                  # Step 3: sampled set
    labels = {}               # Labeled data: i -> f(row)

    while len(S) < max_samples:
        # Step 4: select a batch of highest scoring rows not in S
        candidate_indices = [i for i in range(N) if i not in S]
        if not candidate_indices:
            break

        top_indices = sorted(candidate_indices, key=lambda i: g_hat[i], reverse=True)[:batch_size]

        # Evaluate f(row) and add to S
        for i in top_indices:
            S.add(i)
            labels[i] = f(data[i])

        # Fit logistic regression model on current samples
        X_train = [embeddings[i] for i in labels.keys()]
        y_train = [labels[i] for i in labels.keys()]

        if len(set(y_train)) < 2:
            continue  # Skip model update if only one class seen so far

        clf = LogisticRegression(solver='liblinear')
        clf.fit(X_train, y_train)
        g_hat = clf.predict_proba(embeddings)[:, 1]  # Probability of class 1

    # Step 5: return positives from labeled set
    return [data[i] for i in S if labels[i] == 1]
