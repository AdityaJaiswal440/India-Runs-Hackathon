"""Centralized runtime defaults for the offline ranking pipeline.

Provides environment-variable overrides for values that vary across
local development, CI, and production runs without requiring code changes.
"""

from __future__ import annotations

import os
from datetime import date


def _parse_iso_date(value: str, fallback: date) -> date:
    """Parse an ISO-8601 date string (YYYY-MM-DD), returning fallback on failure."""
    value = (value or "").strip()
    if len(value) < 10:
        return fallback
    try:
        return date(int(value[0:4]), int(value[5:7]), int(value[8:10]))
    except ValueError:
        return fallback


# Reference calendar date used for tenure, recency, and validity checks
# across the honeypot and availability scoring layers.
# Override via: export RANKING_REFERENCE_DATE=YYYY-MM-DD
_DEFAULT_REFERENCE_DATE = date(2026, 6, 22)
RANKING_REFERENCE_DATE: date = _parse_iso_date(
    os.environ.get("RANKING_REFERENCE_DATE", ""),
    _DEFAULT_REFERENCE_DATE,
)
