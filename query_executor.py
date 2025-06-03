import json
import os
from datetime import datetime
from utils.tokenizer_parser import parse_query
from utils.stratified_sampling import stratified_sampling
from utils.online_retrieval import online_retrieval
import requests
from collections import Counter

LOG_PATH = "logs/query_results.jsonl"

LLAMA_ENDPOINT = "http://localhost:11434/api/generate"

def f_llm(row):
    prompt = f"""
The following is a movie review:

\"{row['text']}\"

Is this review expressing a positive sentiment? Answer only with "yes" or "no".
"""
    try:
        response = requests.post(LLAMA_ENDPOINT, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "max_tokens": 5}
        })
        result = response.json()
        answer = result.get("response", "").strip().lower()
        return 1 if "yes" in answer else 0
    except Exception as e:
        print(f"LLM call failed: {e}")
        return 0

def extract_taxonomy(rows):
    examples = "\n".join(f"{i+1}. {row['text']}" for i, row in enumerate(rows[:10]))
    prompt = f"""
Here are some customer reviews:
{examples}

What are 3-5 distinct reasons for dissatisfaction mentioned in these reviews?
List them as short category labels.
"""
    try:
        response = requests.post(LLAMA_ENDPOINT, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "max_tokens": 200}
        })
        result = response.json()
        categories = result.get("response", "").strip().split("\n")
        return [c.strip("-•123. ") for c in categories if c.strip()]
    except Exception as e:
        print(f"LLM taxonomy extraction failed: {e}")
        return []

def classify_group(row, taxonomy):
    label_list = "\n".join(f"- {label}" for label in taxonomy)
    prompt = f"""
Review:
"{row['text']}"

Which of the following categories best describes this review?
{label_list}

Respond with exactly one label.
"""
    try:
        response = requests.post(LLAMA_ENDPOINT, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "max_tokens": 20}
        })
        result = response.json()
        answer = result.get("response", "").strip()
        return answer
    except Exception as e:
        print(f"LLM classification failed: {e}")
        return "Unknown"
    

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
    groupby_clause = next((c for c in parsed[3] if isinstance(c, tuple) and c[0] == 'GROUP_BY'), None) if len(parsed) > 3 else None

    if agg and groupby_clause:
        data = load_data("Datasets/IMDB/currated/imdb_embed_clustering.json")
        estimate, sampled_rows = stratified_sampling(data, f_llm, K=10, sampling_ratio=0.1, return_rows=True)
        taxonomy = extract_taxonomy(sampled_rows)

        if not taxonomy:
            raise ValueError("Failed to extract taxonomy.")

        group_counts = Counter()
        for row in sampled_rows:
            group = classify_group(row, taxonomy)
            group_counts[group] += 1

        log_result(query_str, group_counts)
        return dict(group_counts)

    if agg and where:
        data = load_data("Datasets/IMDB/currated/imdb_embed_clustering.json")
        result = {}
        estimate = stratified_sampling(data, f_llm, K=10, sampling_ratio=0.1)
        estimated_count = estimate * len(data)
        result['estimated_count'] = estimated_count

        if validate:
            metrics = validate_sampling_accuracy(data, f_llm, stratified_sampling, K=10, sampling_ratio=0.1)
            result.update(metrics)

        log_result(query_str, result)
        return result

    if not agg and where:
        data = load_data("Datasets/IMDB/currated/imdb_embed_retrieval.json")
        matched_rows = online_retrieval(data, f_llm, batch_size=10, max_samples=200)
        result = {
            "retrieved_rows": matched_rows,
            "retrieved_count": len(matched_rows)
        }

        if validate:
            full_data = load_data("Datasets/IMDB/currated/imdb_embed_retrieval.json")
            metrics = evaluate_retrieval_metrics(matched_rows, full_data, f_llm)
            result.update(metrics)

        log_result(query_str, result)
        return result

    raise NotImplementedError("Only SELECT COUNT(*), SELECT * with WHERE, and SELECT COUNT(*) GROUP BY queries are supported")

if __name__ == "__main__":
    query = 'SELECT * FROM reviews WHERE sentiment = "positive review"'
    result = execute_query(query)
    print(json.dumps(result, indent=2))
