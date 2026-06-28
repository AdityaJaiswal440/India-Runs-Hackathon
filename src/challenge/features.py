"""Per-candidate text index and feature extraction.

Builds a CandidateIndex once per candidate — a frozen bag of pre-tokenised
text representations — so all downstream scorers can read from it without
re-parsing or re-compiling patterns for every scoring dimension.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List

from challenge.jd_config import (
    CAREER_JD_WEIGHTS,
    CV_SPEECH_ROBOTICS,
    JD_OVERLAP_PHRASES,
    PRODUCTION_SIGNAL_PHRASES,
)
from challenge.semantic import career_semantic_from_blobs
from challenge.text_match import (
    align_to_word_end,
    align_to_word_start,
    clean_leading_ellipsis_fragment,
    compile_multi_patterns,
    count_phrases_fast,
    norm_text,
    split_phrases,
    strip_trailing_partial_token,
    tokenize,
    truncate_at_word_boundary,
)

_CAREER_JD_S, _CAREER_JD_M = split_phrases(tuple(CAREER_JD_WEIGHTS.keys()))
_JD_OVR_S, _JD_OVR_M = split_phrases(JD_OVERLAP_PHRASES)
_PROD_S, _PROD_M = split_phrases(PRODUCTION_SIGNAL_PHRASES)
_CV_S, _CV_M = split_phrases(CV_SPEECH_ROBOTICS)
_JD_OVR_P = compile_multi_patterns(_JD_OVR_M)
_PROD_P = compile_multi_patterns(_PROD_M)
_CV_P = compile_multi_patterns(_CV_M)

_CV_REGEX = re.compile(
    r"computer vision|image moderation|object detection|speech recognition|robotics|autonomous driving",
    re.I,
)
_IR_REGEX = re.compile(
    r"recommend|ranking|retrieval|search|embed|vector|hybrid",
    re.I,
)


@dataclass(frozen=True)
class CandidateIndex:
    """Immutable pre-computed text representations for a single candidate."""
    full_blob: str
    career_blob: str
    full_tokens: FrozenSet[str]
    career_tokens: FrozenSet[str]
    career_tf: Counter


def build_index(profile: Dict[str, Any], history: List[Dict[str, Any]]) -> CandidateIndex:
    """Build a CandidateIndex from raw profile and career history."""
    career = norm_text(" ".join(h.get("description", "") for h in history))
    full = norm_text(
        " ".join([
            profile.get("summary", ""),
            profile.get("headline", ""),
            *[h.get("description", "") for h in history],
        ])
    )
    career_tf: Counter = Counter(re.findall(r"[a-z0-9]{3,}", career))
    return CandidateIndex(
        full_blob=full,
        career_blob=career,
        full_tokens=tokenize(full),
        career_tokens=tokenize(career),
        career_tf=career_tf,
    )


def production_score(idx: CandidateIndex) -> float:
    """Fraction of production-signal phrases present in career text (capped at 1.0)."""
    hits = count_phrases_fast(_PROD_S, _PROD_M, idx.career_tokens, idx.career_blob, _PROD_P)
    return min(1.0, hits / 5.0)


def jd_overlap_score(idx: CandidateIndex) -> float:
    """Fraction of JD overlap phrases present in career text (capped at 1.0)."""
    hits = count_phrases_fast(_JD_OVR_S, _JD_OVR_M, idx.career_tokens, idx.career_blob, _JD_OVR_P)
    return min(1.0, hits / 7.0)


def semantic_score(idx: CandidateIndex, bi_encoder: float | None = None) -> float:
    """Blend lexical semantic score with bi-encoder cosine when available."""
    lex = career_semantic_from_blobs(idx.career_blob, idx.career_tf, idx.career_tokens)
    if bi_encoder is not None:
        return min(1.0, 0.30 * lex + 0.70 * bi_encoder)
    return lex


def cv_language_hits(idx: CandidateIndex) -> int:
    """Count CV/robotics domain signals in the full profile text."""
    hits = len(_CV_S & idx.full_tokens)
    hits += sum(1 for pat in _CV_P if pat.search(idx.full_blob))
    if _CV_REGEX.search(idx.career_blob):
        return max(hits, 2)
    return hits


def _snippet_around_match(text: str, match: re.Match, limit: int = 88) -> str:
    start = align_to_word_start(text, max(0, match.start() - 24))
    end = align_to_word_end(text, min(len(text), match.end() + 48))
    chunk = text[start:end].strip()
    if start > 0:
        chunk = "\u2026" + chunk
    if end < len(text):
        chunk = strip_trailing_partial_token(chunk + "\u2026")
    if len(chunk) > limit + 4:
        chunk = truncate_at_word_boundary(chunk, limit)
        chunk = strip_trailing_partial_token(chunk)
    return chunk


def ir_career_snippet(history: List[Dict[str, Any]], limit: int = 110) -> str:
    """Extract an IR-relevant snippet from the first matching role description."""
    for role in history[:3]:
        desc = (role.get("description") or "").strip()
        if len(desc) < 30:
            continue
        m = _IR_REGEX.search(desc)
        if m:
            snippet = clean_leading_ellipsis_fragment(_snippet_around_match(desc, m, limit))
            return f"{role.get('title', 'Role')} @ {role.get('company', '?')}: {snippet}"
    return ""
