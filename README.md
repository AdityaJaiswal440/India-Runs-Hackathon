# Redrob Intelligent Candidate Ranker

A modular, CPU-optimized machine learning pipeline built for the Redrob Hackathon. It ranks 100,000 candidate profiles against a complex Job Description (Senior AI Engineer) using a hybrid approach of dense semantic embeddings, classical BM25 retrieval, and strict heuristic rule-based filtering.

## Architecture & Approach

This system employs a **Dynamic One-Pass Architecture** that strictly adheres to the hackathon's compute constraints (≤5 min runtime, ≤16GB RAM, CPU-only, no network) while maximizing ranking accuracy and traceability.

### The Ranking Pipeline (`src/pipeline.py` & `src/challenge/redrob_ranker.py`)
With our latest refactor, the legacy two-phase offline/online split has been consolidated into a single, highly-optimized pipeline that completes the ranking for 100,000 candidates in under 2 minutes:
1.  **Data Ingestion:** Streams the `candidates.jsonl` file memory-efficiently using a Python generator.
2.  **Dense Embeddings Retrieval:** Utilizes pre-generated `all-MiniLM-L6-v2` dense embeddings to compute semantic cosine similarity against the JD text.
3.  **Heuristic & Honeypot Checks:** Dynamically applies logical rule-based checks (like `is_honeypot`) and structure-based modifiers (Years of Experience, consulting-only, behavioral availability) directly during the scoring loop.
4.  **Composite Scoring:** Calculates a final hybrid score for each candidate using the semantic baseline merged with the heuristic multipliers and penalties.
5.  **Reasoning Generation:** Generates 1-2 sentence, fact-grounded, zero-hallucination justifications for the top candidates, guaranteeing absolute traceability.
6.  **CSV Output:** Automatically writes the final `submission.csv` in the exact format required by the validators.

## Project Structure

```text
India-Runs-Hackathon/
├── artifacts/                   # Precomputed dense embeddings & ID mappings
├── data/                        # Contains raw datasets, interim files, and synced models
├── sandbox/                     # FastAPI endpoint for Sandbox evaluations
│   └── app.py
├── scripts/                     # Helper and audit scripts (including legacy precompute)
├── src/                         # Core source code
│   ├── challenge/               # Core scoring logic (redrob_ranker.py, embeddings, honeypot)
│   ├── config/                  # Configuration defaults
│   ├── data/                    # JSONL streaming logic
│   ├── features/                # Embedding wrappers & legacy extraction logic
│   ├── ranking/                 # Ranking heuristics and reasoning generation
│   └── utils/                   # Loggers, path config, text sanitization
├── Dockerfile                   # Single-step Docker configuration
├── Makefile                     # Gameday commands
├── README.md                    # Documentation
├── run_pipeline.py              # Entry point script
└── submission.csv               # The final generated ranking output
```

## Step-by-Step Guide for Judges (Sandbox Evaluation)

We have provided a FastAPI sandbox so judges can easily evaluate the model in an isolated environment by uploading `.jsonl` payloads and receiving immediate rankings.

**Step 1: Start the Sandbox Server**
You can launch the API natively or inside Docker. To run it natively:
```bash
# Activate the environment
source .venv/bin/activate
# Start the FastAPI server
uvicorn sandbox.app:app --host 0.0.0.0 --port 8000
```
*The server will boot up and load the ranking logic on `http://localhost:8000`.*

**Step 2: Submit a Test File**
Using a secondary terminal, you can submit a `.jsonl` payload of candidates to the `/rank` endpoint. 
```bash
curl -X POST "http://localhost:8000/rank" -F "file=@data/raw/candidates.jsonl"
```

**Step 3: Review the Output**
The API will return a JSON response containing the top candidates, their calculated composite scores, and the generated zero-hallucination reasoning strings proving exactly why they were ranked in their respective positions.

## Setup & Execution (Full Pipeline)

**1. Clone and Setup:**
```bash
git clone <repo-url>
cd India-Runs-Hackathon
make setup
```

**2. The Single Execution Command:**
To build the Docker container and execute the entire ranking pipeline (outputting `submission.csv` to your host directory in under 2 minutes), simply run:
```bash
make gameday
```

## Constraints Compliance

*   **≤5 min wall-clock:** The unified pipeline performs zero network inference and completes all scoring dynamically in under 2 minutes.
*   **≤16 GB RAM:** Data loading uses Python generators (`yield`). Memory consumption peaks at ~3GB for embeddings lookup.
*   **CPU-only / No Network:** All models (`all-MiniLM-L6-v2`) run locally. No OpenAI/Anthropic API calls are made.
*   **Honeypot Defense:** Logical checks in `honeypot.py` catch impossible profiles (e.g., `signup_date > last_active_date`) and hard-zero their scores before ranking.