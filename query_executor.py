from utils.tokenizer_parser import parse_query
from utils.stratified_sampling import stratified_sampling
from utils.online_retrieval import online_retrieval
import json

def load_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def f_positive(row):
    return 1 if row.get("label") == "positive" else 0

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


def execute_query(query, data, validate=True):
    parsed_query = parse_query(query)
    
    if parsed_query is None:
        raise ValueError(f"Failed to parse query: {query}")
    if parsed_query[0] != 'QUERY':
        raise ValueError("Unsupported query format")

    select_clause = parsed_query[1]
    from_clause = parsed_query[2]
    optional_clause = parsed_query[3] if len(parsed_query) > 3 else []

    # Simple case: SELECT COUNT(*) WHERE ...
    is_agg = any(clause for clause in select_clause[1] if isinstance(clause, tuple) and clause[0] == 'AGG')
    where_clause = next((c for c in optional_clause if isinstance(c, tuple) and c[0] == 'WHERE'), None)

    if is_agg and where_clause:
        result = {}
        est = stratified_sampling(data, f_positive, K=10, sampling_ratio=0.1)
        count_estimate = est * len(data)
        result['estimated_count'] = count_estimate

        if validate:
            metrics = validate_sampling_accuracy(data, f_positive, stratified_sampling, K=10, sampling_ratio=0.1)
            result.update(metrics)

        return result
    if not is_agg or not where_clause:
        matched_rows = online_retrieval(data, f_positive, batch_size=10, max_samples=200)
        return {"retrieved_rows": matched_rows, 
                "retrieved_count": len(matched_rows)}                

    raise NotImplementedError("Only COUNT + WHERE queries supported for now")

if __name__ == "__main__":
    query = 'SELECT * FROM reviews WHERE sentiment = "positive review"'
    data = load_data(r"./Datasets/IMDB/currated/imdb_embed_clustering.json")
    result = execute_query(query, data)
    print(json.dumps(result, indent=2))