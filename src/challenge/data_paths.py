"""Canonical paths to the official India Runs challenge dataset."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_FILES = (
    "candidates.jsonl",
    "sample_candidates.json",
    "job_description.docx",
    "candidate_schema.json",
    "redrob_signals_doc.docx",
    "submission_spec.docx",
    "sample_submission.csv",
    "submission_metadata_template.yaml",
    "README.docx",
    "validate_submission.py",
)


def official_challenge_root() -> Path:
    """Directory containing the published challenge files."""
    override = os.environ.get("CHALLENGE_DATA_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _REPO_ROOT / "data"


def challenge_file(name: str) -> Path:
    """Resolve a challenge file — repo data/ first, then CHALLENGE_DATA_ROOT."""
    synced = _REPO_ROOT / "data" / name
    try:
        if synced.is_file() or (synced.is_symlink() and synced.resolve().is_file()):
            return synced.resolve()
    except OSError:
        pass

    official = official_challenge_root() / name
    if official.is_file():
        return official
    return synced


def repo_data_dir() -> Path:
    return _REPO_ROOT / "data"


def ensure_challenge_data_synced() -> list[str]:
    """Return list of missing official files (empty = all present)."""
    missing = []
    for name in OFFICIAL_FILES:
        path = challenge_file(name)
        if not path.is_file():
            missing.append(name)
    return missing


def candidates_not_found_help(requested: Path) -> str:
    """Portable guidance when candidates.jsonl is missing (no personal paths)."""
    return (
        f"ERROR: candidates file not found: {requested}\n"
        "Resolution order:\n"
        "  1. python rank.py --candidates /path/to/candidates.jsonl\n"
        "  2. export CHALLENGE_DATA_ROOT=/path/to/official/challenge/folder\n"
        "  3. ./scripts/sync_challenge_data.sh  (symlink bundle into ./data/)\n"
        "Default path: ./data/candidates.jsonl (judge-supplied; not committed)."
    )