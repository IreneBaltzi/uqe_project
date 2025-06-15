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

RETRIEVAL_FILE = os.getenv("RETRIEVAL_FILE", "/data/imdb_embed_retrieval.json")
CLUSTERED_DATA  = os.getenv("CLUSTER_FILE",   "/data/imdb_clustered_data.json")
CLUSTER_FILE  = os.getenv("CLUSTER_FILE",   "/data/imdb_embed_clustering.json")

DEFAULT_RATIO        = float(os.getenv("SAMPLING_RATIO", 0.01)) # fraction to sample per cluster
DEFAULT_BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 10))     # for active retrieval
DEFAULT_MAX_SAMPLES  = int(os.getenv("MAX_SAMPLES", 256))   # budget for active retrieval
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

    return estimate_count


def execute_select_groupby(condition_nl: str, groupby_nl: str, sampling_ratio: float, seed: int,
                            cost=None, fused=False, share_sampling=False):
    cache = VirtualColumnCache()
    docs = [json.loads(line) for line in open(CLUSTER_FILE, 'r')]

    sampled_rows, sampled_weights, _ = stratified_sample_and_weight(docs, CLUSTERED_DATA, sampling_ratio, seed)

    if condition_nl:
        def f_where(row):
            row_id = row["id"]
            column = condition_nl
            def compute_fn():
                prompt = (
                    "Read the following movie review and determine if it satisfies the given condition.\n"
                    f"Review: \"{row['review']}\"\n"
                    f"Condition: {condition_nl}\n"
                    "Answer yes or no."
                )
                return 1 if call_llm_binary(prompt) == "yes" else 0
            return cache.get_or_compute(row_id, column, compute_fn)

        filtered = [(r, w) for r, w in zip(sampled_rows, sampled_weights) if f_where(r) == 1]

        if cost:
            cost.add(len(sampled_rows), f"WHERE predicate over {len(sampled_rows)} sampled rows")
    else:
        filtered = list(zip(sampled_rows, sampled_weights))

    if not filtered:
        print("[WARN] No rows matched condition.")
        return {}

    top_reviews = "\n---\n".join(r['review'] for r, _ in filtered[:5])
    taxonomy_prompt = (
        f"We have reviews satisfying the condition: {condition_nl}\n"
        f"---\n{top_reviews}\n---\n"
        "Propose 3–5 concise, non-overlapping category labels for why these reviews meet the condition."
    )
    taxonomy = call_llm_list(taxonomy_prompt) or ["Other"]
    if cost:
        cost.add(1, "GROUPBY taxonomy generation")

    result_counts = defaultdict(float)
    for row, weight in filtered:
        row_id = row["id"]
        column = f"FUSED|SELECT_GROUPBY|{condition_nl}|{groupby_nl}"
        def compute_fn():
            if fused:
                prompt = (
                    f"You are given a list of categories: {', '.join(taxonomy)}.\n"
                    f"Review: \"{row['review']}\"\n"
                    f"Select the single best category from the list that describes why this review satisfies the condition: {condition_nl}.\n"
                    "Return only the category name."
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

    total_weight = sum(result_counts.values())
    return {k: int(round(v / total_weight * len(docs))) for k, v in result_counts.items()}


def execute_select_only(select_expr, rows=None, validate=False, cost=None):
    """
    Handles SELECT "..." FROM ... (no WHERE, no GROUPBY)
    """
    cache = VirtualColumnCache()
    if rows is None:
        rows = [json.loads(line) for line in open(CLUSTER_FILE, 'r')]

    if cost:
        cost.add(len(rows), f"SELECT extraction on {len(rows)} rows")

    # results = []

    _, select_items = select_expr 

    for item in select_items:
        expr_type = item[0]

        if expr_type == "NL":
            nl_text = item[1]
            alias = nl_text.strip('"')
        elif expr_type == "NL_AS":
            nl_text = item[1]
            alias = item[2]
        else:
            print(f"[SKIP] Unsupported SELECT type: {expr_type}")
            continue

        for row in rows:
            row_id = row["id"]

            def compute_fn():
                prompt = (
                    f"Read this review: \"{row['review']}\"\n"
                    f"What is the value for: {nl_text}?\n"
                    "Respond with a short phrase."
                )
                return call_llm_single(prompt)

            val = cache.get_or_compute(row_id, alias, compute_fn)
            row[alias] = val

    for row in rows[:10]:
        print(f"[SELECT RESULT] {row['id']}: {row.get(alias)}")

    return rows

def execute_orderby(attribute_key: str, rows=None, cost=None):
    if rows is None:
        rows = [json.loads(line) for line in open(CLUSTER_FILE, 'r')]
    if cost:
        cost.add(0, "ORDERBY is free")
    sorted_rows = sorted(rows, key=lambda x: x.get(attribute_key, ""))
    for row in sorted_rows[:10]:
        print(f"[ORDERED] {row['id']} -> {row.get(attribute_key)}")
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

    return retrieved

# =============================================================================
#  THE “COMPILER” FUNCTION
# =============================================================================

def compile_and_execute(query_str: str, sampling_ratio: float, batch_size: int,
                        max_samples: int, seed: int, validate: bool):
    cost = CostEstimator()
    parsed = parse_query(query_str)
    plan = plan_query(parsed)

    fused_pairs = set()
    for i, kernel in enumerate(plan):
        if kernel.fuse_with:
            fused_pairs.add((kernel.kind, kernel.fuse_with))

    rows = None

    for kernel in plan:
        kind = kernel.kind
        config = kernel.config
        fused = kernel.fuse_with

        if kind == "WHERE" and not fused:
            result = execute_agg_where(
                condition_nl=config["condition"],
                sampling_ratio=sampling_ratio,
                seed=seed,
                validate=validate,
                cost=cost
            )
            return result

        elif kind == "LIMIT":
            rows = execute_select_limit(
                condition_nl=config["condition"],
                limit_num=config["limit"],
                batch_size=batch_size,
                max_samples=max_samples,
                seed=seed,
                validate=validate,
                cost=cost
            )

        elif kind == "GROUPBY":
            is_fused = ("GROUPBY", "SELECT") in fused_pairs
            result = execute_select_groupby(
                condition_nl=config.get("condition", ""),
                groupby_nl=config["attribute"],
                sampling_ratio=sampling_ratio,
                seed=seed,
                cost=cost,
                fused=is_fused
            )
            return result

        elif kind == "SELECT" and ("GROUPBY", "SELECT") not in fused_pairs:
            rows = execute_select_only(
                select_expr=config["expression"],
                rows=rows,
                validate=validate,
                cost=cost
            )

        elif kind == "ORDERBY":
            rows = execute_orderby(
                attribute_key=config["attribute"],
                rows=rows,
                cost=cost
            )

        else:
            raise NotImplementedError(f"Unknown or unsupported kernel type: {kind}")

    cost.report()
    return rows
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

    result = compile_and_execute(
        query_str=args.query,
        sampling_ratio=args.sampling_ratio,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        validate=args.validate
    )
    print(result)
