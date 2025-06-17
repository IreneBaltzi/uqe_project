# UQE Paper Implementation By Baltzi Eirini & Karavangeli Eftychia

A lightweight, extensible query engine that translates natural language queries into LLM-driven execution over a movie reviews dataset. Designed for experiments in natural language query compilation, stratified sampling, and active retrieval with LLMs. This implementation is a reproduction of the system described in the paper [UQE: A Query Engine for Unstructured Databases](https://arxiv.org/abs/2407.09522). All modules and algorithms closely follow the methods outlined in the official work.

---

## Features

- SQL-like natural language queries over JSONL data
- Stratified sampling with FAISS clustering
- Online active learning retrieval using LLMs
- Caching and cost tracking of LLM invocations
- Docker-based setup with DevContainer support

---

## Project Structure

```
.
├── main.py                    # Entry point for running queries
├── planner.py                 # Converts parsed query into execution plan
├── cost_est.py                # Tracks number of LLM calls
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker environment
├── .devcontainer/             # VS Code DevContainer setup
├── docker-compose.yaml        # Docker compose
├── utils/                     # Utility modules
│   ├── tokenizer_parser.py    # UQL lexer and parser
│   ├── stratified_sampling.py  # Clustering & sampling 
│   ├── online_retrieval.py   # Online active learning 
│   ├── virtual_cache.py      # LLM call caching
│   └── parsetab.py           # PLY-generated parsing table
├── data_curration/           # Utility modules
│   ├── load_data.py          # Data loading and preparation
│   ├── data_curration.py     # Data preprocessing helper
└── README.md
```


---

## Setup

### Docker (recommended)

Build and run inside Docker with GPU and data volume:

```bash
docker build -t uqe .
docker run --rm -it \
  --network=ollama-net \
  --gpus all \
  --cpus=24 \
  -v {data_path}:/data \
  -v $(pwd):/workspace \
  -w /workspace \
  uqe_engine
```

Or use **VS Code DevContainer** (`.devcontainer/devcontainer.json`) for development with remote containers.

---

## Installation (non-Docker)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

Run a query:

```bash
python main.py --query 'SELECT COUNT(*) FROM reviews WHERE "the review is positive"' --validate
```

### Available Arguments:

- `--query`: Your natural language query string
- `--sampling_ratio`: Fraction for stratified sampling (default: 0.005)
- `--batch_size`: Batch size for active retrieval (default: 10)
- `--max_samples`: LLM call budget (default: 256)
- `--validate`: Compare estimate against true labels (optional and only for execution_agg_where function)

---

## Example Queries

```sql
SELECT COUNT(*) FROM reviews WHERE "the review is positive"
SELECT COUNT(*) FROM reviews WHERE "the review is negative" GROUP BY "reason"
SELECT "the sentiment of the review" FROM reviews ORDER BY rating
```

---

## Environment Variables

| Variable         | Default                          | Description                       |
|------------------|----------------------------------|-----------------------------------|
| `RETRIEVAL_FILE` | `/data/imdb_embed_retrieval.json`| Source file for active retrieval  |
| `CLUSTER_FILE`   | `/data/imdb_embed_clustering.json`| Embedding file for clustering    |
| `CLUSTERED_DATA` | `/data/imdb_clustered_data.json` | Precomputed clusters from CLUSTER FILE|
| `SAMPLING_RATIO` | `0.01`                          | Sample fraction per cluster       |
| `SEED`           | `42`                             | Random seed                       |
| `MAX SAMPLES`    | `256`                            | Budget for active retrieval |
| `BATCH_SIZE`     | `10`                             | Batch size for active retrieval

---

## Cost Estimation

The system tracks the number of LLM calls made per query operation via `CostEstimator`.

---

## Dependencies

Key libraries:
- `scikit-learn`
- `faiss-cpu`
- `requests`
- `ply` (parser)
- `numpy`, `scipy`

---

## Docker Compose Setup

To start both **Ollama** and the **UQE Engine** services:

```bash
docker-compose up --build
```

This launches:
- `ollama`: A container exposing the LLM API on `http://ollama:11434`
- `uqe_engine`: Your project container with shared volume and network

To open a shell in your project container:

```bash
docker exec -it uqe_engine bash
```

---

## Stopping Services

To stop everything cleanly:

```bash
docker-compose down
```
