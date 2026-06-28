"""Career description duplicate-template detection.

Synthetic candidate pools often share identical career blurb text across
thousands of unrelated profiles — a copy-paste artefact from the data
generator. This module fingerprints each role description and counts how
many candidates share the same prefix, then applies a graduated score
penalty to candidates whose blurbs are templated at high frequency.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from challenge.text_match import norm_text

_MIN_FINGERPRINT_LEN = 40
_FINGERPRINT_CHARS = 96


def blurb_fingerprint(description: str) -> str:
    """Return the normalised leading characters of a description as its fingerprint."""
    text = norm_text(description or "")
    if len(text) < _MIN_FINGERPRINT_LEN:
        return ""
    return text[:_FINGERPRINT_CHARS].strip()


def build_career_blurb_counts(candidates: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Stream through candidates and count how often each blurb prefix appears."""
    counts: Dict[str, int] = {}
    for raw in candidates:
        for role in raw.get("career_history", []) or []:
            fp = blurb_fingerprint(role.get("description", ""))
            if fp:
                counts[fp] = counts.get(fp, 0) + 1
    return counts


def build_career_blurb_counts_from_path(path: Any) -> Dict[str, int]:
    """Build blurb counts by streaming a JSONL candidates file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        return build_career_blurb_counts(rows)


def template_blurb_modifier(
    history: List[Dict[str, Any]],
    blurb_counts: Dict[str, int] | None,
) -> float:
    """Return a score multiplier penalising candidates with high-frequency template blurbs.

    Checks the first three roles only (the most visible part of the profile).
    The penalty is graduated by frequency — very common blurbs (800+) receive
    the harshest treatment; rare duplicates (< 25) are not penalised at all.
    """
    if not blurb_counts or not history:
        return 1.0

    worst = 1.0
    for role in history[:3]:
        fp = blurb_fingerprint(role.get("description", ""))
        if not fp:
            continue
        n = blurb_counts.get(fp, 1)
        if n >= 800:
            worst = min(worst, 0.62)
        elif n >= 400:
            worst = min(worst, 0.72)
        elif n >= 200:
            worst = min(worst, 0.80)
        elif n >= 100:
            worst = min(worst, 0.88)
        elif n >= 50:
            worst = min(worst, 0.94)
        elif n >= 25:
            worst = min(worst, 0.97)
    return worst
