import os
import sys
import json
import random
import argparse
import numpy as np
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils.tokenizer_parser import parse_query
from utils.stratified_sampling import stratified_sample_and_weight
from utils.online_retrieval import online_retrieval  # we’ll modify this below
from utils.virtual_cache import VirtualColumnCache


# =============================================================================
# GLOBAL DEFAULTS & PARAMETERIZATION
# =============================================================================

RETRIEVAL_FILE = os.getenv("RETRIEVAL_FILE", "/data/imdb_embed_retrieval.json")
CLUSTERED_DATA  = os.getenv("CLUSTER_FILE",   "/data/imdb_clustered_data.json")
CLUSTER_FILE  = os.getenv("CLUSTER_FILE",   "/data/imdb_embed_clustering.json")

# Default hyperparameters
DEFAULT_K            = int(os.getenv("K_CLUSTERS", 5))      # number of clusters for stratified sampling
DEFAULT_RATIO        = float(os.getenv("SAMPLING_RATIO", 0.005)) # fraction to sample per cluster
DEFAULT_BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 10))     # for active retrieval
DEFAULT_MAX_SAMPLES  = int(os.getenv("MAX_SAMPLES", 256))   # budget for active retrieval
DEFAULT_SEED         = int(os.getenv("SEED", 42))

# =============================================================================
# HELPER FUNCTIONS FOR LLM CALLS
# =============================================================================

def call_llm_binary(prompt: str) -> str:
    """
    Send `prompt` to your local LLM endpoint and return either "yes" or "no".
    Adjust the URL or payload as needed for your server.
    """
    # NOTE: You likely already have a function called `call_llm` in your repo;
    # adapt it so that it returns the raw text (e.g. "yes" or "no") without extra whitespace.
    import requests

    payload = {
        "model": "llama3.1:8b", 
        "prompt": prompt,
        "max_tokens": 4,
        "temperature": 0.0
    }
    response = requests.post("http://ollama-container:11434/api/generate", json=payload)
    try:
        # Stitch together all 'response' chunks
        chunks = response.text.strip().split("\n")
        full_response = ""
        for line in chunks:
            data = json.loads(line)
            full_response += data.get("response", "")

        answer = full_response.strip().lower()
        return "yes" if "yes" in answer else "no"

    except Exception as e:
        print("Failed to parse LLM response:", e)
        print("Raw response text:", response.text)
        raise e

def call_llm_list(prompt: str) -> list:
    """
    Send `prompt` to LLM and expect a newline-separated list of labels,
    e.g. "Acting\nPlot\nCinematography". Return as a Python list of strings.
    """
    import requests

    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt + "\n\nPlease output each label on its own line.",
        "max_tokens": 100,
        "temperature": 0.7
    }
    response = requests.post("http://ollama-container:11434/api/generate", json=payload)
    try:
        chunks = response.text.strip().split("\n")
        full_response = ""
        for line in chunks:
            data = json.loads(line)
            full_response += data.get("response", "")
        labels = [line.strip() for line in full_response.split("\n") if line.strip()]
        return labels
    except Exception as e:
        print("Failed to parse LLM list response:", e)
        print("Raw response text:", response.text)
        raise e

def call_llm_single(prompt: str) -> str:
    """
    Send `prompt` to LLM and expect exactly one label (or short answer).
    For example, "Plot" or "Acting". We return as a stripped string.
    """
    import requests

    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt + "\n\nAnswer with exactly one of the above labels.",
        "max_tokens": 10,
        "temperature": 0.0
    }
    response = requests.post("http://ollama-container:11434/api/generate", json=payload)
    try:
        chunks = response.text.strip().split("\n")
        full_response = ""
        for line in chunks:
            data = json.loads(line)
            full_response += data.get("response", "")
        return full_response.strip()
    except Exception as e:
        print("Failed to parse LLM single response:", e)
        print("Raw response text:", response.text)
        raise e

# =============================================================================
# LOW-LEVEL KERNEL IMPLEMENTATIONS
# =============================================================================

def execute_agg_where(condition_nl: str, sampling_ratio: float, seed: int, validate: bool):
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
    for row, weight in zip(sampled_rows, sampled_weights):
        result = f_where(row)
        total_weighted += weight * result
        total_weight += weight

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


def execute_select_groupby(condition_nl: str,
                           groupby_nl: str,
                           sampling_ratio: float,
                           seed: int):
    """
    Abstraction + aggregation via stratified sampling and taxonomy.
    """
    cache = VirtualColumnCache()
    def f_where(row):
        row_id = row["id"]
        column = condition_nl
        def compute_fn():
            prompt = (
                f'Review: "{row["review"]}"\n'
                f'Does this review satisfy: {condition_nl}? Answer "yes" or "no".'
            )
            return 1 if call_llm_binary(prompt) == "yes" else 0
        return cache.get_or_compute(row_id, column, compute_fn)

    docs = [json.loads(line) for line in open(RETRIEVAL_FILE, 'r')]
    N = len(docs)

    sampler = stratified_sample_and_weight(CLUSTER_FILE, seed=seed)
    num_samples = max(1, int(sampling_ratio * N))
    sample_indices = sampler.sample(num_samples)
    sampled_rows = [docs[i] for i in sample_indices]
    sampled_weights = []
    for idx in sample_indices:
        pid = sampler.assignments[idx]
        p = sampler.cluster_sizes[pid] / N
        sampled_weights.append(1.0 / p)

    # Taxonomy extraction
    few = sampled_rows[:5]
    examples = "\n---\n".join(r['review'] for r in few)
    prompt_tax = (
        f"We have example reviews that match: {condition_nl}\n"
        f"---\n{examples}\n---\n"
        "Please propose 3 to 5 concise category labels."
    )
    taxonomy = call_llm_list(prompt_tax) or ["Cat1","Cat2"]

    # Classification
    counts = defaultdict(float)
    for row, w in zip(sampled_rows, sampled_weights):
        row_id = row['id']
        column = f"reason|{condition_nl}|{groupby_nl}"
        def comp():
            prompt = (
                f"Possible reasons: {taxonomy}\n"
                f"Review: \"{row['review']}\"\n"
                "Which single reason best explains why this review matches?"
            )
            lbl = call_llm_single(prompt)
            return lbl if lbl in taxonomy else taxonomy[0]
        lbl = cache.get_or_compute(row_id, column, comp)
        counts[lbl] += w

    # Normalize to total counts
    total_w = sum(counts.values())
    result = {lbl: int(round((w/total_w)*N)) for lbl, w in counts.items()}
    return result


def execute_select_limit(condition_nl: str,
                         limit_num: int,
                         batch_size: int,
                         max_samples: int,
                         seed: int,
                         validate: bool):
    """
    Semantic retrieval via online active learning.
    """
    cache = VirtualColumnCache()
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
    retrieved = online_retrieval(
        data=data,
        f=f_where,
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed,
        limit=limit_num
    )
    if validate and condition_nl[1].strip('"').lower() in ["the review is positive", "the review is negative"]:
        lbl = condition_nl[1].strip('"').lower().split()[-1]
        true = sum(1 for d in data if d.get('label','').lower()==lbl)
        print(f"[Validation] True matches = {true}, Retrieved = {len(retrieved)}")
    return retrieved


# =============================================================================
#  THE “COMPILER” FUNCTION
# =============================================================================

def compile_and_execute(query_str: str,
                        sampling_ratio: float,
                        batch_size: int,
                        max_samples: int,
                        seed: int,
                        validate: bool):
    parsed = parse_query(query_str)
    _, select_clause, _, where_clause, groupby_clause, orderby_clause, limit_clause = parsed
    is_agg = any(lit[0] in ('AGG','AGG_AS') for lit in select_clause[1])
    has_gb = groupby_clause is not None
    has_lim = limit_clause is not None


    if is_agg and has_gb:
        return execute_select_groupby(
            condition_nl=groupby_clause[1],
            groupby_nl=groupby_clause[1],
            sampling_ratio=sampling_ratio,
            seed=seed
        )
    elif is_agg:
        return execute_agg_where(
            condition_nl=where_clause[1],
            sampling_ratio=sampling_ratio,
            seed=seed,
            validate=validate
        )
    elif has_lim:
        return execute_select_limit(
            condition_nl=where_clause[1],
            limit_num=limit_clause[1],
            batch_size=batch_size,
            max_samples=max_samples,
            seed=seed,
            validate=validate
        )
    else:
        print("Error: Unsupported query.")
        sys.exit(1)


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
