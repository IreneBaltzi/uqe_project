from utils.tokenizer_parser import parse_query
from utils.stratified_sampling import stratified_sampling
from utils.online_retrieval import online_retrieval
import json
import os
from datetime import datetime
LOG_PATH = "./logs/query_results.jsonl"

def load_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def log_result(query, result):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as log_file:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result
        }
        log_file.write(json.dumps(log_entry) + "\n")

def f_positive(row):
    return 1 if row.get("label") == "positive" else 0

def is_aggregation_query(parsed_query):
    """Check if the parsed query contains an aggregation function."""
    select_clause = parsed_query[1]
    return any(
        clause for clause in select_clause[1]
        if isinstance(clause, tuple) and clause[0] == 'AGG'
    )

def has_where_clause(parsed_query):
    optional_clauses = parsed_query[3] if len(parsed_query) > 3 else []
    return next((c for c in optional_clauses if isinstance(c, tuple) and c[0] == 'WHERE'), None)


def evaluate_retrieval_metrics(retrieved_rows, full_data, f):
    retrieved_ids = set(row["id"] for row in retrieved_rows)
    true_positives = sum(1 for row in retrieved_rows if f(row) == 1)
    false_positives = len(retrieved_rows) - true_positives

    total_positives = sum(1 for row in full_data if f(row) == 1)
    false_negatives = total_positives - true_positives

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, total_positives)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall
    }
def validate_sampling_accuracy(data, f, stratified_sampling_fn, K=10, sampling_ratio=0.1):
    """
    Compare stratified sampling estimate against true ground truth.

    Args:
        data (List[Dict]): Dataset with embeddings and labels
        f (Callable[[Dict], float]): Evaluation function (simulates LLM)
        stratified_sampling_fn (Callable): Your stratified sampling function
        K (int): Number of clusters
        sampling_ratio (float): Fraction of each cluster to sample

    Returns:
        dict: Contains estimate, true count, errors, and ratio
    """
    N = len(data)
    true_count = sum(f(row) for row in data)
    estimate_mean = stratified_sampling_fn(data, f, K=K, sampling_ratio=sampling_ratio)
    estimated_count = estimate_mean * N

    abs_error = abs(estimated_count - true_count)
    rel_error = abs_error / max(1, true_count)

    return {
        "estimated_count": estimated_count,
        "true_count": true_count,
        "abs_error": abs_error,
        "rel_error": rel_error
    }

def execute_query(query_str, validate=True):
    parsed = parse_query(query_str)

    if parsed is None:
        raise ValueError(f"Failed to parse query: {query_str}")
    if parsed[0] != 'QUERY':
        raise ValueError("Unsupported query format")

    agg = is_aggregation_query(parsed)
    where = has_where_clause(parsed)

    # Load appropriate dataset based on query type
    if agg and where:
        data = load_data("Datasets\IMDB\currated\imdb_embed_clustering.json")
        result = {}
        estimate = stratified_sampling(data, f_positive, K=10, sampling_ratio=0.1)
        estimated_count = estimate * len(data)
        result['estimated_count'] = estimated_count

        if validate:
            metrics = validate_sampling_accuracy(data, f_positive, stratified_sampling, K=10, sampling_ratio=0.1)
            result.update(metrics)

        log_result(query_str, result)
        return result

    if not agg and where:
        data = load_data("Datasets\IMDB\currated\imdb_embed_retrieval.json")
        matched_rows = online_retrieval(data, f_positive, batch_size=10, max_samples=200)
        result = {
            "retrieved_rows": matched_rows,
            "retrieved_count": len(matched_rows)
        }

        if validate:
            full_data = load_data("Datasets\IMDB\currated\imdb_embed_retrieval.json")
            metrics = evaluate_retrieval_metrics(matched_rows, full_data, f_positive)
            result.update(metrics)
        log_result(query_str, result)
        return result

    raise NotImplementedError("Only SELECT COUNT(*) and SELECT * with simple WHERE conditions are supported for now")

if __name__ == "__main__":
    query = 'SELECT * FROM reviews WHERE sentiment = "positive review"'
    result = execute_query(query)
    print(json.dumps(result, indent=2)) 