# Redrob Intelligent Candidate Ranker

A modular, CPU-optimized machine learning pipeline built for the Redrob Hackathon. It ranks 100,000 candidate profiles against a complex Job Description (Senior AI Engineer) using a hybrid approach of dense semantic embeddings, classical BM25 retrieval, and strict heuristic rule-based filtering.

## Architecture & Approach

This system employs a **Two-Phase Architecture** to strictly adhere to the hackathon's compute constraints (≤5 min runtime, ≤16GB RAM, CPU-only, no network) while maximizing ranking accuracy and traceability.

### Phase 1: Offline Precomputation (`scripts/precompute.py`)
This phase handles all heavy lifting and model inference. It is allowed to exceed the 5-minute constraint.
1.  **Data Ingestion:** Streams the 465MB `candidates.jsonl.gz` file memory-efficiently using a Python generator.
2.  **BM25 Retrieval:** Tokenizes candidate text (headline, summary, career history) and scores it against a tailored "requirements-only" JD text using classical BM25.
3.  **Dense Embeddings:** Utilizes `all-MiniLM-L6-v2` to generate 384-dimensional vectors for candidates and the JD, computing cosine similarity to capture semantic intent (bypassing keyword-stuffing traps).
4.  **Heuristic Extraction:** Calculates structured features (Years of Experience distance, consulting-only history flags, behavioral availability modifiers).
5.  **Honeypot Detection:** Applies logical rule-based checks to flag subtly impossible profiles (e.g., skill duration exceeding total career duration).
6.  **Artifact Generation:** Consolidates all scores, flags, and features into a single highly-compressed Parquet file (`data/interim/features.parquet`).

### Phase 2: Online Ranking (`src/pipeline.py`)
This is the Stage 3 reproduced step. It completes in milliseconds.
1.  **Load Artifacts:** Reads the precomputed Parquet feature table.
2.  **Filter-First:** Drops all honeypots and hard-disqualifier candidates (e.g., career-long IT consulting) *before* scoring.
3.  **Composite Scoring:** Normalizes BM25 scores and applies a hand-engineered formula: `(Dense_Sim + BM25_Norm) * Behavior_Score - YoE_Penalty`.
4.  **Reasoning Generation:** Generates 1-2 sentence, fact-grounded, zero-hallucination justifications for the top 100 candidates.
5.  **CSV Output:** Writes the final `submission.csv` in the exact format required by the hackathon validators.

## Project Structure

```text
India-Runs-Hackathon/
├── data/
│   ├── interim/                 # Gitignored. Stores Parquet & decompressed JSONL
│   └── raw/                     # Gitignored. Stores candidates.jsonl.gz & JD text
├── scripts/
│   └── precompute.py            # Offline master pipeline
├── src/
│   ├── data/
│   │   └── loader.py            # Memory-efficient JSONL streamer
│   ├── features/
│   │   ├── embedder.py          # MiniLM model wrapper
│   │   ├── heuristic_extractor  # YoE, consulting, and behavior logic
│   │   └── honeypot.py          # Logical trap detection
│   ├── ranking/
│   │   ├── heuristic_scorer.py  # Composite score calculation
│   │   └── reasoning_gen.py     # Templated fact generation
│   ├── utils/
│   │   ├── config.py            # Dynamic env-based path resolution
│   │   └── logger.py            # Centralized logging
│   └── pipeline.py              # Online phase orchestrator
├── tests/
│   ├── fixtures/                # Mock data for isolated testing
│   ├── test_data_loader.py
│   ├── test_heuristic_extractor.py
│   └── test_precompute.py
├── .env.example
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Prerequisites

*   Python 3.11+
*   `uv` (Astral's fast Python package installer) - [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
*   Docker (for Stage 3 reproduction simulation)

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd India-Runs-Hackathon
    ```

2.  **Create your local environment file:**
    ```bash
    cp .env.example .env
    ```
    *(By default, this sets `APP_ENV=prod`. Change to `APP_ENV=test` to run against mock data fixtures).*

3.  **Install dependencies:**
    ```bash
    make setup
    ```
    *(This creates a `.venv` directory and installs all requirements).*

4.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

5.  **Place Data:**
    Put the `candidates.jsonl.gz` file into the `data/raw/` directory.

## Usage

### Running the Full Pipeline (Production)

1.  **Run the Offline Precomputation:**
    This generates the feature table. It may take 3-10 minutes depending on your CPU.
    ```bash
    make precompute
    ```

2.  **Run the Online Ranking:**
    This loads the Parquet, scores, and generates the final CSV.
    ```bash
    make run
    ```
    *Output: `submission.csv` in the project root.*

### Running Tests

To run the unit test suite (uses mock data, requires no heavy downloads):
```bash
make test
```

### Docker Execution

To simulate the Stage 3 reproduction environment:
```bash
make build
docker run --rm --cpus="2" --memory="16g" redrob-ranker:v3
```

## Configuration & Environments

The system uses a dynamic configuration module (`src/utils/config.py`) that reads the `APP_ENV` environment variable via `python-dotenv`.

*   **`APP_ENV=prod` (Default):** Routes paths to `data/raw/` and `data/interim/`. Use this for processing the real 100K dataset.
*   **`APP_ENV=test`:** Routes paths to `tests/fixtures/data/` and `tests/fixtures/interim/`. Use this for rapid development and unit testing without downloading the 80MB embedding model or 465MB dataset.

## Constraints Compliance

*   **≤5 min wall-clock:** The online phase (`pipeline.py`) performs zero model inference and completes in milliseconds. Only `precompute.py` exceeds time limits, which is explicitly permitted by the spec.
*   **≤16 GB RAM:** Data loading uses Python generators (`yield`). Pandas operations in the online phase are limited to the ~50MB Parquet file.
*   **CPU-only / No Network:** All models (`all-MiniLM-L6-v2`, `rank_bm25`) run locally. No OpenAI/Anthropic API calls are made during the ranking phase.
*   **Honeypot Defense:** Logical checks in `honeypot.py` catch impossible profiles (e.g., `signup_date > last_active_date`) and hard-zero their scores before ranking.