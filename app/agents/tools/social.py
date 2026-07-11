"""Supplementary evidence from public social posts, via official platform
APIs only (per the project's source-access scope) -- never scraped directly.

No general-purpose free social search API exists for this without
per-platform developer credentials, so this returns no results unless a
platform token is configured, rather than falling back to scraping.
"""
from __future__ import annotations

import os

_INSTAGRAM_TOKEN = os.environ.get("INSTAGRAM_GRAPH_API_TOKEN")


def search_social_mentions(mall: str, store_name: str) -> list[dict]:
    """Returns [{"platform", "post_url", "published_at", "excerpt"}, ...].
    Currently a stub: wire to the Instagram Graph API / Meta Content Library
    (or another official API) once credentials are available."""
    if not _INSTAGRAM_TOKEN:
        return []
    return []


def analyze_post_text(text: str) -> dict:
    """Extracts location clues (floor/unit mentions, "grand opening",
    "now open on level N", hashtags) from a social post caption/comment.
    Simple heuristic parse -- kept deterministic so it works without an
    LLM call for the common cases."""
    import re

    floor_match = re.search(r"\b(?:level|floor)\s*(\d)\b", text, re.IGNORECASE)
    return {
        "floor_mention": int(floor_match.group(1)) if floor_match else None,
        "hashtags": re.findall(r"#(\w+)", text),
        "is_opening_announcement": bool(re.search(r"grand opening|now open|coming soon", text, re.IGNORECASE)),
    }
