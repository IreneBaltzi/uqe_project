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
from utils.stratified_sampling import stratified_sampling
from utils.online_retrieval import online_retrieval  # we’ll modify this below

# =============================================================================
# 1. GLOBAL DEFAULTS & PARAMETERIZATION
# =============================================================================

RETRIEVAL_FILE = os.getenv("RETRIEVAL_FILE", "Datasets/IMDB/currated/imdb_embed_retrieval.json")
CLUSTER_FILE   = os.getenv("CLUSTER_FILE",   "Datasets/IMDB/currated/imdb_embed_clustering.json")

# Default hyperparameters
DEFAULT_K            = int(os.getenv("K_CLUSTERS", 10))      # number of clusters for stratified sampling
DEFAULT_RATIO        = float(os.getenv("SAMPLING_RATIO", 0.1)) # fraction to sample per cluster
DEFAULT_BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 10))     # for active retrieval
DEFAULT_MAX_SAMPLES  = int(os.getenv("MAX_SAMPLES", 256))   # budget for active retrieval
DEFAULT_SEED         = int(os.getenv("SEED", 42))

# =============================================================================
# 2. HELPER FUNCTIONS FOR LLM CALLS
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
        "model": "llama3", 
        "prompt": prompt,
        "max_tokens": 4,
        "temperature": 0.0
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    text = response.json()["text"].strip().lower()
    # We expect exactly "yes" or "no"; if something else comes back, handle gracefully
    if "yes" in text:
        return "yes"
    elif "no" in text:
        return "no"
    else:
        # If ambiguous, default to "no"
        return "no"

def call_llm_list(prompt: str) -> list:
    """
    Send `prompt` to LLM and expect a newline-separated list of labels,
    e.g. "Acting\nPlot\nCinematography". Return as a Python list of strings.
    """
    import requests

    payload = {
        "model": "llama3",
        "prompt": prompt + "\n\nPlease output each label on its own line.",
        "max_tokens": 100,
        "temperature": 0.7
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    raw = response.json()["text"].strip()
    # Split by newline, filter out empty lines
    labels = [line.strip() for line in raw.split("\n") if line.strip()]
    return labels

def call_llm_single(prompt: str) -> str:
    """
    Send `prompt` to LLM and expect exactly one label (or short answer).
    For example, "Plot" or "Acting". We return as a stripped string.
    """
    import requests

    payload = {
        "model": "llama3",
        "prompt": prompt + "\n\nAnswer with exactly one of the above labels.",
        "max_tokens": 10,
        "temperature": 0.0
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    return response.json()["text"].strip()

# =============================================================================
# 3. LOW-LEVEL KERNEL IMPLEMENTATIONS
# =============================================================================

def execute_agg_where(condition_nl: str, K: int, sampling_ratio: float, seed: int, validate: bool):
    """
    Implements: SELECT COUNT(*) FROM movie_reviews WHERE condition_nl
    Uses stratified_sampling over imdb_embed_clustering.json.
    Returns: estimated_count (integer); if validate=True also prints true count.
    """

    # Build f_where(row) that returns 1 if LLM says "yes", else 0.
    def f_where(row):
        prompt = f"""Review: "{row['review']}"
Does this review satisfy: {condition_nl} ? Answer "yes" or "no"."""
        return 1 if call_llm_binary(prompt) == "yes" else 0

    # 1. Load clustered data
    data = []
    with open(CLUSTER_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 2. Run stratified sampling (returns a fraction estimate)
    estimate_frac = stratified_sampling(
        data,
        f_where,
        K=K,
        sampling_ratio=sampling_ratio,
        seed=seed,
        return_rows=False  # we only need the fraction
    )

    N = len(data)
    est_count = int(round(estimate_frac * N))

    if validate:
        # Compute the true count (costly; invokes LLM N times)
        true_count = sum(f_where(row) for row in data)
        print(f"[Validation] True count = {true_count}, Estimate = {est_count}")

    return est_count


def execute_select_groupby(condition_nl: str, groupby_nl: str, K: int, sampling_ratio: float, seed: int, validate: bool):
    """
    Implements:
      SELECT "groupby_nl" AS alias, COUNT(*) 
        FROM movie_reviews 
       WHERE condition_nl 
    GROUP BY "groupby_nl" AS alias

    1. Stratified sample to get a small subset of matching rows (with weights).
    2. Ask LLM to propose 3-5 category labels for why reviews match (taxonomy).
    3. Ask LLM to classify each sampled row into one label, then multiply by its weight.
    4. Return a dict {label: estimated_count}.
    If validate=True, also scan all rows for ground-truth grouping (very costly).
    """

    # Build f_where(row) exactly as above
    def f_where(row):
        prompt = f"""Review: "{row['review']}"
Does this review satisfy: {condition_nl} ? Answer "yes" or "no"."""
        return 1 if call_llm_binary(prompt) == "yes" else 0

    # 1. Load clustered data
    data = []
    with open(CLUSTER_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 2. Stratified sample THIS subset to get (estimate_frac, sampled_rows, sampled_weights)
    estimate_frac, sampled_rows, sampled_weights = stratified_sampling(
        data,
        f_where,
        K=K,
        sampling_ratio=sampling_ratio,
        seed=seed,
        return_rows=True
    )

    # 3. Taxonomy extraction: ask LLM to propose labels (3-5) based on first few sampled examples
    #    We typically show up to 5 examples, but you can adjust.
    few_examples = sampled_rows[:5]
    example_texts = "\n---\n".join(r["review"] for r in few_examples)
    prompt_taxonomy = f"""
We have example reviews that match this condition: {condition_nl}
---
{example_texts}
---
Please propose 3 to 5 concise category labels that explain why these reviews match.
"""
    taxonomy = call_llm_list(prompt_taxonomy)
    if len(taxonomy) == 0:
        # Fallback if LLM returns nothing
        taxonomy = ["Category1", "Category2"]

    # 4. Classify each sampled row into one of the taxonomy labels (weighted count)
    group_counts = defaultdict(float)
    for idx, row in enumerate(sampled_rows):
        prompt_classify = f"""
Possible reasons: {taxonomy}
Review: "{row['review']}"
Which single reason best explains why this review matches?"""
        label = call_llm_single(prompt_classify)
        if label not in taxonomy:
            # If LLM output is slightly off, pick the first taxonomy label as default
            label = taxonomy[0]
        group_counts[label] += sampled_weights[idx]

    # 5. If validate=True, compute ground-truth on ALL matching rows (very expensive)
    if validate:
        true_group_counts = defaultdict(int)
        for row in data:
            if f_where(row) == 1:
                # Classify each truly-positive row
                prompt_val = f"""
Possible reasons: {taxonomy}
Review: "{row['review']}"
Which single reason best explains why this review matches?"""
                true_label = call_llm_single(prompt_val)
                if true_label not in taxonomy:
                    true_label = taxonomy[0]
                true_group_counts[true_label] += 1
        print("[Validation] True group counts:", dict(true_group_counts))

    return dict(group_counts)


def execute_select_limit(condition_nl: str,
                         limit_num: int,
                         batch_size: int,
                         max_samples: int,
                         seed: int,
                         validate: bool):
    """
    Implements:
      SELECT * 
        FROM movie_reviews 
       WHERE condition_nl 
      LIMIT limit_num

    This calls the online_retrieval(...) function (from online_retrieval.py),
    passing `limit=limit_num` so that it stops as soon as `limit_num` positives
    are found (or once `max_samples` labels have been used).

    Returns a list of up to `limit_num` matching rows. If validate=True, also
    computes the true positives by scanning all rows.
    """

    # 1. Define f_where(row) which asks the LLM (or oracle) yes/no for each row.
    def f_where(row):
        prompt = f"""Review: "{row['review']}"
Does this review satisfy: {condition_nl} ? Answer "yes" or "no"."""
        return 1 if call_llm_binary(prompt) == "yes" else 0

    # 2. Load the entire retrieval dataset from disk
    data = []
    with open(RETRIEVAL_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 3. Call online_retrieval from online_retrieval.py, passing limit=limit_num
    retrieved_rows = online_retrieval(
        data=data,
        f=f_where,
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed,
        limit=limit_num
    )

    # 4. If validate=True, compute the true positives by scanning every row
    if validate:
        true_positives = [row for row in data if f_where(row) == 1]
        print(f"[Validation] True positives = {len(true_positives)}, Retrieved = {len(retrieved_rows)}")

    return retrieved_rows


# =============================================================================
# 4. THE “COMPILER” FUNCTION
# =============================================================================

def compile_and_execute(query_str: str,
                        K: int, sampling_ratio: float, batch_size: int,
                        max_samples: int, seed: int, validate: bool):
    """
    1. Parse the query string into a fixed 7-tuple.
    2. Decide which case (AGG_WHERE, SELECT_GROUPBY, SELECT_LIMIT).
    3. Call the appropriate kernel above and return its result.
    """
    parsed = parse_query(query_str)
    # parsed is a 7-tuple: ('QUERY', select_clause, from_clause,
    #                       where_clause_or_None,
    #                       groupby_clause_or_None,
    #                       orderby_clause_or_None,
    #                       limit_clause_or_None)

    _, select_clause, from_clause, where_clause, groupby_clause, orderby_clause, limit_clause = parsed

    # 1. Is this an aggregation (COUNT)?
    is_agg   = any(lit[0] in ('AGG', 'AGG_AS') for lit in select_clause)
    # 2. Is there a GROUP BY?
    has_gb   = (groupby_clause is not None)
    # 3. Is there a LIMIT?
    has_lim  = (limit_clause is not None)

    if is_agg and has_gb:
        # CASE: SELECT ... GROUP BY ... (abstraction + aggregation)
        # Extract the natural‐language text from groupby_clause
        # groupby_clause is ('GROUP_BY_AS', nl_text, alias)
        groupby_nl = groupby_clause[1]  # the quoted text
        # Extract the WHERE text
        if where_clause is None:
            print("Error: GROUP BY query must have a WHERE clause.")
            sys.exit(1)
        condition_nl = where_clause[1]  # e.g. '"the review is positive"'
        return execute_select_groupby(
            condition_nl=condition_nl,
            groupby_nl=groupby_nl,
            K=K,
            sampling_ratio=sampling_ratio,
            seed=seed,
            validate=validate
        )

    elif is_agg and not has_gb:
        # CASE: SELECT COUNT(*) ... WHERE ... (conditional aggregation)
        if where_clause is None:
            print("Error: COUNT(*) query must have a WHERE clause.")
            sys.exit(1)
        condition_nl = where_clause[1]
        return execute_agg_where(
            condition_nl=condition_nl,
            K=K,
            sampling_ratio=sampling_ratio,
            seed=seed,
            validate=validate
        )

    elif not is_agg and has_lim:
        # CASE: SELECT * ... WHERE ... LIMIT ... (semantic retrieval)
        if where_clause is None:
            print("Error: SELECT * ... LIMIT must have a WHERE clause.")
            sys.exit(1)
        condition_nl = where_clause[1]
        limit_num    = limit_clause[1]
        return execute_select_limit(
            condition_nl=condition_nl,
            limit_num=limit_num,
            batch_size=batch_size,
            max_samples=max_samples,
            seed=seed,
            validate=validate
        )

    else:
        print("Error: Unsupported or malformed query.")
        sys.exit(1)


# =============================================================================
# 5. MAIN: PARSE ARGS AND RUN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a UQE query on IMDB.")
    parser.add_argument("--query", type=str, required=True,
                        help="UQL query string, e.g. SELECT COUNT(*) FROM movie_reviews WHERE \"the review is positive\"")
    parser.add_argument("--K", type=int, default=DEFAULT_K,
                        help="Number of clusters for stratified sampling")
    parser.add_argument("--sampling_ratio", type=float, default=DEFAULT_RATIO,
                        help="Fraction to sample per cluster")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for active retrieval")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="Budget (max number of LLM calls) for active retrieval")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--validate", action="store_true",
                        help="If set, also compute the ground-truth by scanning all data")
    args = parser.parse_args()

    result = compile_and_execute(
        query_str=args.query,
        K=args.K,
        sampling_ratio=args.sampling_ratio,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        validate=args.validate
    )
    print("Result:", result)
