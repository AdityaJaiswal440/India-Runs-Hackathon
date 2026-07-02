# sandbox/app.py
"""
FastAPI Sandbox Server — Redrob Intelligent Candidate Ranker

Two endpoints:
  POST /rank      Upload a small .jsonl file → returns output.csv
                  (same pipeline, same scoring as the submission era.csv)
  GET  /reproduce Returns the committed era.csv that was produced on the full
                  candidates.jsonl dataset (reproducibility proof for judges).

The /rank endpoint delegates entirely to `rank_candidates()` in
`src/challenge/redrob_ranker.py` — the exact same function used by
`make gameday` / `python run_pipeline.py`.  No custom re-implementation
of scoring lives here; this guarantees the output is identical in
structure and logic to the production era.csv.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.challenge.redrob_ranker import rank_candidates
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Redrob Ranker Sandbox",
    description=(
        "Interactive sandbox for evaluating the Redrob Intelligent Candidate "
        "Ranker pipeline. Upload a custom .jsonl candidate file and receive a "
        "ranked era.csv back — produced by the exact same pipeline that "
        "generated the submission artifact."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Paths that are baked into the Docker image
# ---------------------------------------------------------------------------
_COMMITTED_ERA_CSV = Path("era.csv")          # the reproduction artifact
_CANDIDATES_JSONL  = Path("data/raw/candidates.jsonl")  # full dataset

# ---------------------------------------------------------------------------
# ASCII-safe reasoning formatter (same as pipeline.py)
# ---------------------------------------------------------------------------
_NON_ASCII_MAP = str.maketrans({
    "\u2014": "-",    # em dash
    "\u00b7": "-",    # middle dot
    "\u2026": "...",  # ellipsis
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
})


def _ascii_reasoning(text: str) -> str:
    return text.translate(_NON_ASCII_MAP).encode("ascii", "ignore").decode("ascii")


def _rows_to_csv(rows) -> str:
    """Convert a list of ScoredCandidate objects to a CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for i, row in enumerate(rows):
        writer.writerow([
            row.candidate_id,
            i + 1,
            f"{row.score:.6f}",
            _ascii_reasoning(row.reasoning),
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    """Quick health check — confirms the server is up."""
    return {
        "status": "ok",
        "message": (
            "Redrob Ranker Sandbox is running. "
            "POST /rank to score a custom dataset. "
            "GET /reproduce to download the committed era.csv. "
            "Visit /docs for the interactive Swagger UI."
        ),
    }


@app.post(
    "/rank",
    tags=["Sandbox"],
    summary="Rank a custom candidate list",
    response_description="A downloadable output.csv with ranked candidates",
)
async def rank_custom_candidates(
    file: UploadFile = File(
        ...,
        description=(
            "A .jsonl or .jsonl.gz file where every line is a JSON object "
            "matching the candidate schema (see data/raw/candidate_schema.json). "
            "A 10-candidate sample ships at data/raw/sample_candidates.jsonl."
        ),
    )
) -> Response:
    """
    Upload a candidate JSONL file and receive a ranked **output.csv**.

    The ranking logic is **identical** to the full pipeline (`make gameday`):

    * Hybrid BM25 + bi-encoder semantic scoring via `all-MiniLM-L6-v2`
    * Honeypot / fraud detection (26 structural rules)
    * Availability, assessment, career-depth, and location modifiers
    * Fact-grounded, ASCII-safe reasoning strings

    The only difference from the submission `era.csv` is that this runs on
    *your uploaded data* rather than the full 500 k-candidate dataset.
    """
    if not file.filename or not file.filename.endswith((".jsonl", ".jsonl.gz")):
        raise HTTPException(
            status_code=400,
            detail="Please upload a file ending with .jsonl or .jsonl.gz",
        )

    # Save the upload to a temporary directory (never touches repo files)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        logger.info(f"Received upload: {file.filename} → {tmp_path}")

        # Delegate entirely to the canonical ranker
        top_candidates = rank_candidates(tmp_path, top_k=100)

        if not top_candidates:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No valid candidates were found in the uploaded file. "
                    "Make sure every line is a valid JSON object with at least "
                    "'candidate_id', 'profile', 'skills', 'career_history', "
                    "and 'redrob_signals' fields."
                ),
            )

        csv_content = _rows_to_csv(top_candidates)
        logger.info(
            f"Ranked {len(top_candidates)} candidates from {file.filename}. "
            f"Top candidate: {top_candidates[0].candidate_id} "
            f"(score {top_candidates[0].score:.6f})"
        )

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="output.csv"'},
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Pipeline failed for {file.filename}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get(
    "/reproduce",
    tags=["Reproducibility"],
    summary="Download the committed era.csv",
    response_description="The reproducible submission artifact era.csv",
)
def get_committed_era_csv() -> Response:
    """
    Returns the **pre-computed `era.csv`** that was generated by running
    `make gameday` on the full `candidates.jsonl` dataset.

    This endpoint lets judges verify that the committed artifact is intact
    inside the Docker image without re-running the full 2-minute pipeline.
    """
    if not _COMMITTED_ERA_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "era.csv not found in the image. "
                "Ensure the Dockerfile includes `COPY era.csv .` or run "
                "`make gameday` and rebuild the image."
            ),
        )

    content = _COMMITTED_ERA_CSV.read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="era.csv"'},
    )