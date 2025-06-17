import os
import sys
import json
import random
import argparse
import requests
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils.tokenizer_parser import parse_query
from utils.stratified_sampling import stratified_sample_and_weight
from utils.online_retrieval import online_retrieval  # we’ll modify this below
from utils.virtual_cache import VirtualColumnCache
from planner import plan_query
from cost_est import CostEstimator

# =============================================================================
# GLOBAL DEFAULTS & PARAMETERIZATION
# =============================================================================

RETRIEVAL_FILE = os.getenv("RETRIEVAL_FILE", "/data/imdb_embed_retrieval_final.json")
CLUSTERED_DATA  = os.getenv("CLUSTERED_DATA",   "/data/imdb_clustered_data.json")
CLUSTER_FILE  = os.getenv("CLUSTER_FILE",   "/data/imdb_embed_clustering_final.json")

DEFAULT_RATIO        = float(os.getenv("SAMPLING_RATIO", 0.01)) # fraction to sample per cluster
DEFAULT_BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 10))     # for active retrieval
DEFAULT_MAX_SAMPLES  = int(os.getenv("MAX_SAMPLES", 64))   # budget for active retrieval
DEFAULT_SEED         = int(os.getenv("SEED", 42))

# =============================================================================
# HELPER FUNCTIONS FOR LLM CALLS
# =============================================================================

def call_llm_binary(prompt: str) -> str:
    payload = {"model": "llama3.1:8b", "prompt": prompt, "max_tokens": 4, "temperature": 0.0}
    response = requests.post("http://ollama-container:11434/api/generate", json=payload)
    chunks = response.text.strip().split("\n")
    full_response = "".join(json.loads(line).get("response", "") for line in chunks)
    return "yes" if "yes" in full_response.strip().lower() else "no"

def call_llm_list(prompt: str) -> list:
    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt + "\n\nPlease output each label on its own line.",
        "max_tokens": 150,
        "temperature": 0.7
    }
    response = requests.post("http://ollama-container:11434/api/generate", json=payload)
    try:
        chunks = response.text.strip().split("\n")
        full_response = ""
        for line in chunks:
            data = json.loads(line)
            full_response += data.get("response", "")

        # Filter and clean lines
        lines = [l.strip("-•1234567890. ").strip() for l in full_response.split("\n")]
        labels = [l for l in lines if l and not l.lower().startswith("here are")]

        return labels
    except Exception as e:
        print("Failed to parse LLM list response:", e)
        print("Raw response text:", response.text)
        raise e

def call_llm_single(prompt: str) -> str:
    payload = {"model": "llama3.1:8b", "prompt": prompt, "max_tokens": 10, "temperature": 0.0}
    response = requests.post("http://ollama-container:11434/api/generate", json=payload)
    chunks = response.text.strip().split("\n")
    full_response = "".join(json.loads(line).get("response", "") for line in chunks)
    return full_response.strip()

# =============================================================================
# LOW-LEVEL KERNEL IMPLEMENTATIONS
# =============================================================================

def execute_agg_where(condition_nl: str, sampling_ratio: float, seed: int, validate: bool, cost=None):
    cache = VirtualColumnCache()

    def f_where(row):
        row_id = row["id"]
        column = condition_nl
        def compute_fn():
            prompt = (
                "Read the following movie review and determine if its sentiment is positive or negative. "
                "If the review is mixed, classify based on the overall tone.\n"
                f"Review: \"{row['review']}\"\n"
                f"Question: Does this review satisfy the condition — {condition_nl}?\n"
                "Answer with yes or no only."
            )
            return 1 if call_llm_binary(prompt) == "yes" else 0
        return cache.get_or_compute(row_id, column, compute_fn)

    docs = [json.loads(line) for line in open(CLUSTER_FILE, 'r')]
    sampled_rows, sampled_weights, _ = stratified_sample_and_weight(docs, CLUSTERED_DATA, sampling_ratio, seed)

    total_weighted, total_weight = 0.0, 0.0
    evaluated = 0
    for row, weight in zip(sampled_rows, sampled_weights):
        result = f_where(row)
        evaluated += 1 

        total_weighted += weight * result
        total_weight += weight

    if cost:
        cost.add(evaluated, f"WHERE on {len(sampled_rows)} sampled rows")

    if total_weight == 0:
        print("[WARN] Total weight is zero.")
        return 0

    estimate_count = int(round((total_weighted / total_weight) * len(docs)))
    print(f"[RESULT] Estimate = {estimate_count}")

    if validate and condition_nl[1].strip('"').lower() in ["the review is positive", "the review is negative"]:
        label = condition_nl[1].strip('"').lower().split()[-1]
        true_count = sum(1 for doc in docs if doc.get('label','').lower() == label)
        print(f"[Validation] True count = {true_count}, Estimate = {estimate_count}")
    
    if cost:
        # print("[COST REPORT]")
        print(cost.report())

    return estimate_count, cost

def execute_semantic_retrieval(condition_nl: str,
                                batch_size: int,
                                max_samples: int,
                                seed: int,
                                limit: int = None,
                                validate: bool = False,
                                cost=None):
    from utils.online_retrieval import online_retrieval
    cache = VirtualColumnCache()

    def f_where(row):
        row_id = row["id"]
        column = condition_nl
        def compute_fn():
            prompt = (
                f"Review: \"{row['review']}\"\n"
                f"Does the following condition hold? {condition_nl}\n"
                "Answer yes or no only."
            )
            return 1 if call_llm_binary(prompt) == "yes" else 0
        return cache.get_or_compute(row_id, column, compute_fn)

    data = [json.loads(line) for line in open(RETRIEVAL_FILE, 'r')]
    rows = online_retrieval(
        data,
        f=f_where,
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed,
        limit=limit
    )

    if cost:
        cost.add(len(rows), f"Online retrieval of {len(rows)} matching rows")
    
    for i, row in enumerate(rows):
        print(f"--- Row {i+1} ---")
        print(f"Title  : {row.get('title')}")
        print(f"Review : {row.get('review')}")
        print(f"Rating : {row.get('rating')}")
        print()
    if cost:
        print("[COST REPORT]")
        print(cost.report())

def execute_where_groupby(condition_nl: str, groupby_nl: str, sampling_ratio: float, seed: int,
                            cost=None, fused=False, share_sampling=False):
    
    print(f"[DEBUG] execute_where_groupby called with condition: {condition_nl}, groupby: {groupby_nl}")

    cache = VirtualColumnCache()
    docs = [json.loads(line) for line in open(CLUSTER_FILE, 'r')]

    sampled_rows, sampled_weights, _ = stratified_sample_and_weight(docs, CLUSTERED_DATA, sampling_ratio, seed)

    if condition_nl:
        def f_where(row):
            row_id = row["id"]
            column = condition_nl
            def compute_fn():
                prompt = (
                    "You will be given a review and a condition.\n"
                    "Your task is to decide whether the review satisfies the condition below.\n\n"
                    f"Condition: {condition_nl}\n"
                    f"Review: \"{row['review']}\"\n\n"
                    "Answer only with yes or no. If unsure, answer no."
                )
                return 1 if call_llm_binary(prompt) == "yes" else 0
            return cache.get_or_compute(row_id, column, compute_fn)
        
        print(f"[DEBUG] Applying WHERE condition to {len(sampled_rows)} sampled rows")
        filtered = [(r, w) for r, w in zip(sampled_rows, sampled_weights) if f_where(r) == 1]
        print(f"[DEBUG] {len(filtered)} rows matched the condition")

        if cost:
            cost.add(len(sampled_rows), f"WHERE predicate over {len(sampled_rows)} sampled rows")
    else:
        filtered = list(zip(sampled_rows, sampled_weights))

    if not filtered:
        print("[WARN] No rows matched condition.")
        return {}
    
    for i, (row, _) in enumerate(filtered[:10]):
        print(f"[DEBUG][Filtered Review #{i+1}] {row['review'][:300]}")
        
    top_reviews = "\n---\n".join(r['review'] for r, _ in filtered[:5])
    taxonomy_prompt = (
    f"You are given a set of movie reviews that all satisfy the following condition:\n"
    f"{condition_nl}\n\n"
    "Your task is to propose 3-5 concise and distinct category labels that describe common patterns or themes "
    "that explain *why* these reviews meet the given condition.\n"
    "Make sure the categories are mutually exclusive and based on the content or reasoning found in the reviews.\n\n"
    "Reviews:\n"
    f"{top_reviews}\n\n"
    "Return each category label on its own line, without explanations, numbering or notes, just the categories."
)
    taxonomy = call_llm_list(taxonomy_prompt) or ["Other"]
    print(f"[DEBUG] Taxonomy generated: {taxonomy}")
    taxonomy_key = '|'.join(taxonomy)

    if cost:
        cost.add(1, "GROUPBY taxonomy generation")

    result_counts = defaultdict(float)
    for row, weight in filtered:
        row_id = row["id"]
        column = f"FUSED|SELECT_GROUPBY|{condition_nl}|{groupby_nl}|{taxonomy_key}"
        def compute_fn():
            if fused:
                prompt = (
                    f"You are given the following list of category labels:\n"
                    f"{', '.join(taxonomy)}\n\n"
                    f"Review: \"{row['review']}\"\n\n"
                    f"Select the single best category from the list that explains why this review satisfies the condition: {condition_nl}.\n"
                    "You must choose **exactly one** label from the list. Return only the label text without modification."
                )
            else:
                prompt = (
                    f"You are given a list of categories: {', '.join(taxonomy)}.\n"
                    f"Review: \"{row['review']}\"\n"
                    f"Select the best matching category for grouping.\n"
                    "Return only the category name."
                )
            return call_llm_single(prompt)

        label = cache.get_or_compute(row_id, column, compute_fn)
        if label not in taxonomy:
            label = taxonomy[0]
        result_counts[label] += weight

    if cost:
        cost.add(len(filtered), f"GROUPBY classification on {len(filtered)} rows")

    result = {k: int(round(v)) for k, v in result_counts.items()}
    print("[RESULT] Grouped counts:")
    for label, count in result.items():
        print(f"  {label}: {count}")
    if cost:
        print("[COST REPORT]")
        print(cost.report())
    return result, cost

def execute_orderby(attribute_key: str, rows=None, cost=None):
    if rows is None:
        rows = [json.loads(line) for line in open(CLUSTER_FILE, 'r')]
    if cost:
        cost.add(0, "ORDERBY is free")
    sorted_rows = sorted(rows, key=lambda x: x.get(attribute_key, ""))
    for i, row in enumerate(sorted_rows[:10]):
        print(f"[RESULT #{i+1}] ID: {row['id']}")
        print(f"  {attribute_key.capitalize()}: {row.get(attribute_key)}")
        print(f"  Sentiment: {row.get('the sentiment of the review', 'N/A')}")
        print(f"  Review: {row['review'][:200]}...\n")
    return sorted_rows

def execute_select_limit(condition_nl: str,
                         limit_num: int,
                         batch_size: int,
                         max_samples: int,
                         seed: int,
                         validate: bool,
                         cost=None):
    """
    Online retrieval with WHERE + LIMIT fused (early stopping).
    """

    cache = VirtualColumnCache()
    random.seed(seed)

    def f_where(row):
        row_id = row['id']
        column = condition_nl
        def compute_fn():
            prompt = (
                "You will be given a movie review and a condition to evaluate.\n"
                "Your task is to determine if the review satisfies the condition below.\n\n"
                f"Condition: {condition_nl}\n"
                f"Review: \"{row['review']}\"\n\n"
                "Does the review satisfy the condition above?\n"
                "Answer with only: yes or no."
            )
            return 1 if call_llm_binary(prompt) == "yes" else 0
        return cache.get_or_compute(row_id, column, compute_fn)

    data = [json.loads(line) for line in open(RETRIEVAL_FILE, 'r')]
    indices = list(range(len(data)))
    random.shuffle(indices)

    retrieved = []
    total_evaluated = 0

    for start in range(0, len(indices), batch_size):
        if total_evaluated >= max_samples or len(retrieved) >= limit_num:
            break

        batch_indices = indices[start:start + batch_size]
        batch = [data[i] for i in batch_indices]

        for row in batch:
            if total_evaluated >= max_samples or len(retrieved) >= limit_num:
                break
            if f_where(row) == 1:
                retrieved.append(row)
            total_evaluated += 1

    print(f"[INFO] Evaluated {total_evaluated} rows to retrieve {len(retrieved)} matches")

    if cost:
        cost.add(total_evaluated, f"WHERE+LIMIT evaluation on {total_evaluated} rows")

    if validate and condition_nl[1].strip('"').lower() in ["the review is positive", "the review is negative"]:
        label = condition_nl[1].strip('"').lower().split()[-1]
        true_matches = sum(1 for d in data if d.get('label', '').lower() == label)
        print(f"[Validation] True matches = {true_matches}, Retrieved = {len(retrieved)}")

    print(f"[RESULT] Retrieved {len(retrieved)} rows after WHERE + LIMIT.")
    for i, row in enumerate(retrieved[:10]):
        print(f"{i+1}. Title: {row.get('title')} - Review: {row.get('review')[:100]}")
    if cost:
        print("[COST REPORT]")
        print(cost.report())
    return retrieved

# =============================================================================
#  THE “COMPILER” FUNCTION
# =============================================================================

def compile_and_execute(query_str, sampling_ratio, batch_size,
                        max_samples, seed, validate):
    cost = CostEstimator()
    parsed = parse_query(query_str)
    plan = plan_query(parsed)

    # === Track kernels ===
    rows = None
    kernels_by_type = {k.kind: k for k in plan}
    fused_pairs = {(k.kind, k.fuse_with) for k in plan if k.fuse_with}

    # === Detect SELECT * + WHERE only ( semantic retrieval) ===
    if "SELECT" in kernels_by_type and "WHERE" in kernels_by_type:
        select_kernel = kernels_by_type["SELECT"]
        where_kernel = kernels_by_type["WHERE"]
        select_expr = select_kernel.config["expression"]

        # If SELECT is just '*', no AGG or GROUPBY, run semantic retrieval
        if select_expr == '*' and "GROUPBY" not in kernels_by_type:
            condition_nl = where_kernel.config["condition"]
            limit = kernels_by_type.get("LIMIT", {}).get("config", {}).get("limit", None)

            rows = execute_semantic_retrieval(
                condition_nl=condition_nl,
                batch_size=batch_size,
                max_samples=max_samples,
                seed=seed,
                limit=limit,
                validate=validate,
                cost=cost
            )

    # === Handle COUNT(*) or other AGGREGATION ===
    if "SELECT" in kernels_by_type and "WHERE" in kernels_by_type:
        select_kernel = kernels_by_type["SELECT"]
        where_kernel = kernels_by_type["WHERE"]
        select_items = select_kernel.config["expression"]

        # Fix: make sure select_items is a list
        if isinstance(select_items, tuple):
            select_items = [select_items]

        # Aggregation detection
        if any(t[0] == "AGG" for t in select_items):
            result = execute_agg_where(
                condition_nl=where_kernel.config["condition"],
                sampling_ratio=sampling_ratio,
                seed=seed,
                validate=validate,
                cost=cost
            )
            return result

    # === GROUPBY ===
    if "GROUPBY" in kernels_by_type:
        group_kernel = kernels_by_type["GROUPBY"]
        where_condition = group_kernel.config.get("condition", "")
        group_attr = group_kernel.config["attribute"]
        fused = ("GROUPBY", "WHERE") in fused_pairs

        result = execute_where_groupby(
            condition_nl=where_condition,
            groupby_nl=group_attr,
            sampling_ratio=sampling_ratio,
            seed=seed,
            cost=cost,
            fused=fused
        )
        return result

    # === WHERE + LIMIT ===
    if "LIMIT" in kernels_by_type and "WHERE" in kernels_by_type:
        where_kernel = kernels_by_type["WHERE"]
        limit_kernel = kernels_by_type["LIMIT"]

        rows = execute_select_limit(
            condition_nl=where_kernel.config["condition"],
            limit_num=limit_kernel.config["limit"],
            batch_size=batch_size,
            max_samples=max_samples,
            seed=seed,
            validate=validate,
            cost=cost
        )    

    # === ORDER BY ===
    if "ORDERBY" in kernels_by_type:
        rows = execute_orderby(
            attribute_key=kernels_by_type["ORDERBY"].config["attribute"],
            rows=rows,
            cost=cost
        )

    # return rows, cost.report()


# =============================================================================
#  MAIN: PARSE ARGS AND RUN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--sampling_ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    result, cost_report = compile_and_execute(
        query_str=args.query,
        sampling_ratio=args.sampling_ratio,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        validate=args.validate
    )

