"""Text normalisation and phrase-matching utilities for the ranking pipeline.

All modules that need to match skill phrases, tokenise career text, or
truncate reasoning strings should import from here rather than rolling
their own regex. Centralising this also lets us cache compiled patterns
once at import time instead of recompiling on every candidate.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import FrozenSet, Iterable, Pattern, Sequence, Tuple

_WORD_START = r"(?<![a-z0-9])"
_WORD_END = r"(?![a-z0-9])"
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def norm_text(s: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""
    if not s:
        return ""
    s = s.lower().strip()
    if "  " not in s and "\n" not in s and "\t" not in s:
        return s
    return re.sub(r"\s+", " ", s)


def norm_skill(name: str) -> str:
    """Normalise a skill name: lowercase + replace separators with spaces."""
    n = norm_text(name)
    return re.sub(r"[-_/]+", " ", n)


def tokenize(text: str) -> FrozenSet[str]:
    """Return a frozenset of alphanumeric tokens (len >= 2) from normalised text."""
    return frozenset(_TOKEN_RE.findall(norm_text(text)))


def split_phrases(phrases: Sequence[str]) -> Tuple[FrozenSet[str], Tuple[str, ...]]:
    """Partition phrases into single-token and multi-word groups.

    Single tokens are stored in a frozenset for O(1) membership tests.
    Multi-word phrases are stored as a tuple for regex compilation.
    """
    singles: set[str] = set()
    multi: list[str] = []
    for p in phrases:
        if " " in p.strip():
            multi.append(p)
        else:
            singles.add(p.strip().lower())
    return frozenset(singles), tuple(multi)


@lru_cache(maxsize=512)
def _phrase_pattern(phrase: str) -> Pattern[str]:
    """Return a compiled, word-boundary-anchored pattern for a phrase."""
    p = norm_text(phrase)
    body = re.escape(p).replace(r"\ ", r"\s+") if " " in p else re.escape(p)
    return re.compile(_WORD_START + body + _WORD_END, re.IGNORECASE)


def compile_multi_patterns(multi: Tuple[str, ...]) -> Tuple[Pattern[str], ...]:
    """Pre-compile all multi-word phrase patterns for reuse across candidates."""
    return tuple(_phrase_pattern(m) for m in multi)


def phrase_in_tokens(phrase: str, tokens: FrozenSet[str], text: str) -> bool:
    """Return True if phrase appears in the token set (single) or text (multi-word)."""
    if not phrase:
        return False
    if " " not in phrase:
        return phrase in tokens
    return bool(_phrase_pattern(phrase).search(text))


def count_phrases_fast(
    singles: FrozenSet[str],
    multi: Tuple[str, ...],
    tokens: FrozenSet[str],
    blob: str,
    multi_patterns: Tuple[Pattern[str], ...] | None = None,
) -> int:
    """Count how many phrases from a pre-split set appear in the given text."""
    hits = sum(1 for s in singles if s in tokens)
    patterns = multi_patterns if multi_patterns is not None else compile_multi_patterns(multi)
    hits += sum(1 for pat in patterns if pat.search(blob))
    return hits


def align_to_word_start(text: str, pos: int) -> int:
    """Advance pos forward to the next word boundary if it lands mid-token."""
    if pos <= 0 or pos >= len(text):
        return max(0, pos)
    if not text[pos - 1].isalnum() or not text[pos].isalnum():
        return pos
    ws = text.find(" ", pos)
    return pos if ws == -1 else min(ws + 1, len(text))


def align_to_word_end(text: str, pos: int) -> int:
    """Retreat pos backward to the previous word boundary if it lands mid-token."""
    if pos <= 0:
        return 0
    if pos >= len(text):
        return len(text)
    if not text[pos - 1].isalnum() or not text[pos].isalnum():
        return pos
    sp = text.rfind(" ", 0, pos)
    return sp if sp >= 0 else 0


_COMPLETE_SUFFIX = re.compile(
    r"(?:ing|tion|ment|ness|ally|ious|able|ive|ers?|ed|ly|ds|ms|cs|us|um|es|or|al|ic|on|s)$",
    re.I,
)


def strip_trailing_partial_token(chunk: str) -> str:
    """Drop an incomplete token before a terminal ellipsis (e.g. 'evolvin…' → '…')."""
    if not chunk.endswith("\u2026"):
        return chunk
    body = chunk[:-1].rstrip()
    if not body or not body[-1].isalnum():
        return chunk
    prefix = "\u2026" if chunk.startswith("\u2026") else ""
    core = body[1:] if prefix else body
    words = core.split()
    if not words:
        return chunk
    last = words[-1]
    if not last[-1].isalnum():
        return chunk
    if _COMPLETE_SUFFIX.search(last):
        return chunk
    trimmed = " ".join(words[:-1]).strip()
    if prefix:
        return f"\u2026{trimmed}\u2026" if trimmed else "\u2026"
    return f"{trimmed}\u2026" if trimmed else "\u2026"


def count_phrases(phrases: Iterable[str], text: str) -> int:
    """Convenience wrapper: count phrase hits without pre-splitting."""
    singles, multi = split_phrases(tuple(phrases))
    blob = norm_text(text)
    return count_phrases_fast(singles, multi, tokenize(blob), blob)


def truncate_at_word_boundary(text: str, limit: int, ellipsis: str = "\u2026") -> str:
    """Truncate text to at most `limit` characters without cutting mid-word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    budget = max(8, limit - len(ellipsis))
    cut = text[:budget]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    if (
        len(cut) < len(text)
        and cut
        and cut[-1].isalnum()
        and text[len(cut) : len(cut) + 1].isalnum()
    ):
        sp = cut.rfind(" ")
        if sp > 0:
            cut = cut[:sp]
    cut = cut.rstrip(".,;:")
    return cut + ellipsis if cut else ellipsis


def clean_leading_ellipsis_fragment(text: str) -> str:
    """Remove orphan short tokens immediately after a leading ellipsis.

    e.g. '…and shipped' becomes '…shipped' so reasoning snippets
    read naturally when sliced from mid-sentence career text.
    """
    if not text.startswith("\u2026"):
        return text
    rest = text[1:].lstrip()
    parts = rest.split(None, 1)
    if len(parts) == 2 and len(parts[0]) <= 3 and parts[0].isalpha():
        return "\u2026" + parts[1]
    return text
