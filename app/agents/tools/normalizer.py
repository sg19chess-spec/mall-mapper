"""Canonicalizes name variants ("Nike", "Nike Factory Store", "Nike @ Mall of
America") before Entity Resolution merges evidence, using RapidFuzz
similarity plus a small alias table.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

_STOPWORDS = {"store", "shop", "outlet", "factory", "the"}
_ALIASES = {
    "lego": "LEGO Store",
    "sea life": "Sea Life Minnesota Aquarium",
}

SIMILARITY_THRESHOLD = 85


def _strip_noise(name: str) -> str:
    name = re.sub(r"@.*$", "", name)  # "Nike @ Mall of America" -> "Nike "
    name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    tokens = [t for t in name.split() if t not in _STOPWORDS]
    return " ".join(tokens).strip()


def normalize(raw_name: str) -> str:
    stripped = _strip_noise(raw_name)
    for alias_key, canonical in _ALIASES.items():
        if alias_key in stripped:
            return _strip_noise(canonical)
    return stripped


def cluster_names(names: list[str]) -> dict[str, str]:
    """Returns {raw_name: canonical_key} by fuzzy-clustering normalized names."""
    normalized = {n: normalize(n) for n in names}
    canonical_keys: list[str] = []
    mapping: dict[str, str] = {}
    for raw, norm in normalized.items():
        match = None
        for key in canonical_keys:
            if fuzz.ratio(norm, key) >= SIMILARITY_THRESHOLD:
                match = key
                break
        if match is None:
            canonical_keys.append(norm)
            match = norm
        mapping[raw] = match
    return mapping
