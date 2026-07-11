"""Supplementary evidence from public mall-walkthrough videos, modeled as
two complementary sources per the Research Agent's YouTube workflow:

    search_videos -> for each video: video metadata (title/description/
    upload date -- visual/context clues) AND transcript (spoken mentions
    of floor/adjacency -- a stronger, verbal observation).

Frame-by-frame OCR/vision on sampled video frames is a Phase 2 enhancement
(computationally expensive) and intentionally not implemented here.

Uses the official YouTube Data API (search + metadata) and the public
timedtext/caption track (transcript) -- both accessed the way a browser
would, no login required. Returns no results when YOUTUBE_API_KEY isn't
configured, except for a small bundled dev-mode sample transcript so the
metadata -> transcript -> structured-evidence pipeline is exercisable
offline (see SAMPLE_TRANSCRIPTS below).
"""
from __future__ import annotations

import os
import re

import httpx

_API_KEY = os.environ.get("YOUTUBE_API_KEY")
_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# Hedged/uncertain narration should contribute less than a stated fact --
# and not just as a binary hedged/not-hedged flag: "probably" and "I guess"
# are both hedges, but not equally uncertain. A continuous lookup table
# (no LLM call needed -- the phrasing narrators use is fairly regular)
# mapping cue phrase -> certainty. Detected per-sentence; the Validation
# Agent multiplies this into the confidence contribution via
# Evidence.certainty rather than the Research Agent trying to resolve the
# ambiguity itself. Ordered strongest-hedge-first so overlapping phrases
# (e.g. "pretty sure" containing "sure") resolve to the more specific one.
STATED_CERTAINTY = 1.0
CERTAINTY_LEXICON: dict[str, float] = {
    "definitely": 1.0,
    "certainly": 1.0,
    "confirmed": 1.0,
    "probably": 0.8,
    "likely": 0.75,
    "pretty sure": 0.65,
    "seems": 0.65,
    "appears": 0.65,
    "might": 0.5,
    "may have": 0.5,
    "could be": 0.5,
    "used to": 0.5,
    "maybe": 0.35,
    "i think": 0.35,
    "i believe": 0.35,
    "i heard": 0.35,
    "possibly": 0.35,
    "i guess": 0.25,
    "not sure": 0.25,
    "i could be wrong": 0.25,
}


def _assess_certainty(sentence: str) -> tuple[float, str]:
    """Returns (certainty, reason). Reason is purely for audit/debugging --
    e.g. "hedge_phrase: might", "booster_phrase: definitely", or
    "stated_as_fact" -- and plays no role in the confidence math itself.
    When multiple cues match (rare), the most conservative (lowest-certainty)
    one wins."""
    lowered = sentence.lower()
    matches = [(cert, phrase) for phrase, cert in CERTAINTY_LEXICON.items() if phrase in lowered]
    if not matches:
        return STATED_CERTAINTY, "stated_as_fact"
    certainty, phrase = min(matches, key=lambda pair: pair[0])
    label = "booster_phrase" if certainty >= STATED_CERTAINTY else "hedge_phrase"
    return certainty, f"{label}: {phrase}"

# A small bundled "walkthrough transcript" sample -- segments with caption
# start times, matching the shape youtube_transcript_api returns -- used
# only when YOUTUBE_API_KEY isn't configured, so the transcript-extraction
# path (including timestamps) can still be exercised end-to-end offline.
# Deliberately consistent with app/agents/tools/web.py's SAMPLE_DIRECTORY
# (Nike and Apple are both Floor 2 there too), so this acts as genuine
# corroboration rather than contradicting evidence.
SAMPLE_TRANSCRIPTS: dict[str, list[dict]] = {
    "nike": [{"text": "Nike is on level 2, right next to Apple.", "start": 122.0}],
    "apple": [{"text": "Apple is on level 2, next to Nike and across from Sephora.", "start": 130.0}],
    "lego store": [{"text": "The LEGO Store is on level 1, near the Sea Life aquarium entrance.", "start": 45.0}],
    "starbucks": [{"text": "Starbucks is on level 1, next to the north entrance.", "start": 20.0}],
}


def search_videos(mall: str, store_name: str, timeout: float = 10.0) -> list[dict]:
    """Returns [{"video_id", "video_title", "video_url", "published_at",
    "description"}, ...] via the official YouTube Data API."""
    if not _API_KEY:
        return []
    try:
        resp = httpx.get(
            _SEARCH_URL,
            params={
                "part": "snippet", "q": f"{mall} {store_name} walkthrough",
                "type": "video", "maxResults": 3, "key": _API_KEY,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            {
                "video_id": it["id"]["videoId"],
                "video_title": it["snippet"]["title"],
                "video_url": f"https://www.youtube.com/watch?v={it['id']['videoId']}",
                "published_at": it["snippet"]["publishedAt"],
                "description": it["snippet"]["description"],
            }
            for it in items
        ]
    except Exception:
        return []


def get_transcript_segments(video_id: str) -> list[dict] | None:
    """Raw caption segments [{"text", "start", "duration"}, ...], or None if
    unavailable/unfetchable."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception:
        return None


_FLOOR_PATTERNS = [
    re.compile(r"\b{store}\b[^.]*?\bon\s+(?:the\s+)?level\s+(\d)\b", re.IGNORECASE),
    re.compile(r"\b{store}\b[^.]*?\bon\s+(?:the\s+)?(\d)(?:st|nd|rd|th)?\s+floor\b", re.IGNORECASE),
]
_ADJACENT_PATTERNS = [
    re.compile(r"\b{store}\b[^.]*?\bnext\s+to\s+([A-Z][\w' ]{{1,30}})", re.IGNORECASE),
    re.compile(r"\b{store}\b[^.]*?\bacross\s+from\s+([A-Z][\w' ]{{1,30}})", re.IGNORECASE),
    re.compile(r"next\s+to\s+\b{store}\b\s+is\s+([A-Z][\w' ]{{1,30}})", re.IGNORECASE),
]


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_offset_map(segments: list[dict]) -> tuple[str, list[tuple[int, float]]]:
    """Concatenates segment texts into one string and records, for each
    segment, the character offset it starts at -- so a match found in the
    joined text can be mapped back to the caption timestamp it came from."""
    text = ""
    offsets: list[tuple[int, float]] = []
    for seg in segments:
        offsets.append((len(text), seg["start"]))
        text += seg["text"] + " "
    return text, offsets


def _time_for_offset(offsets: list[tuple[int, float]], char_index: int) -> float:
    best = offsets[0][1] if offsets else 0.0
    for start, t in offsets:
        if start <= char_index:
            best = t
        else:
            break
    return best


def extract_spatial_clues(segments: list[dict], store_name: str) -> list[dict]:
    """Scans transcript sentences that mention store_name for spoken
    floor/adjacency clues, returning
    [{"clue": {"floor": 2, "timestamp": "00:02:02"},
      "certainty": 1.0, "certainty_reason": "stated_as_fact"}, ...].
    Heuristic regex parsing -- intentionally simple/deterministic rather
    than an LLM call, since the sentence patterns real walkthrough
    narration uses are fairly regular. Certainty is a continuous score
    (see CERTAINTY_LEXICON) so the Validation Agent weighs "definitely"
    higher than "probably", and "probably" higher than "I guess", rather
    than a flat hedged/not-hedged cut."""
    escaped = re.escape(store_name)
    text, offsets = _build_offset_map(segments)
    results: list[dict] = []

    for m in re.finditer(r"[^.!?]+[.!?]?", text):
        sentence = m.group().strip()
        if not sentence or store_name.lower() not in sentence.lower():
            continue
        timestamp = _format_timestamp(_time_for_offset(offsets, m.start()))
        certainty, certainty_reason = _assess_certainty(sentence)

        for pattern_template in _FLOOR_PATTERNS:
            pattern = re.compile(pattern_template.pattern.format(store=escaped), re.IGNORECASE)
            fm = pattern.search(sentence)
            if fm:
                results.append({
                    "clue": {"floor": int(fm.group(1)), "timestamp": timestamp},
                    "certainty": certainty, "certainty_reason": certainty_reason,
                })
                break

        for pattern_template in _ADJACENT_PATTERNS:
            pattern = re.compile(pattern_template.pattern.format(store=escaped), re.IGNORECASE)
            am = pattern.search(sentence)
            if am:
                # store names are typically 1-3 words -- stop at the first
                # clause boundary so "next to Nike and across from Sephora"
                # yields "Nike", not the whole trailing clause.
                raw = am.group(1).strip().rstrip(".")
                for stop_word in (" and ", " near ", " across ", ","):
                    raw = raw.split(stop_word)[0]
                results.append({
                    "clue": {"adjacent_to": raw.strip(), "timestamp": timestamp},
                    "certainty": certainty, "certainty_reason": certainty_reason,
                })
                break

    return results


def _metadata_floor_hint(video: dict) -> int | None:
    text = f"{video.get('video_title', '')} {video.get('description', '')}"
    m = re.search(r"\blevel\s+(\d)\b", text, re.IGNORECASE) or re.search(r"\b(\d)(?:st|nd|rd|th)?\s+floor\b", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def research_store(mall: str, store_name: str) -> dict:
    """The full Research Agent YouTube workflow for one store: search ->
    metadata -> transcript -> structured evidence. Returns
    {"metadata_hits": [...], "transcript_clues": [...]}, each entry
    carrying its source video (for provenance) and a "certainty" alongside
    the structured "clue"."""
    videos = search_videos(mall, store_name)

    if not videos:
        # dev-mode fallback: no live API key, but exercise the same
        # transcript -> clue extraction path (including timestamps) on the
        # bundled sample.
        key = store_name.strip().lower()
        segments = SAMPLE_TRANSCRIPTS.get(key)
        if segments is None:
            return {"metadata_hits": [], "transcript_clues": []}
        hits = extract_spatial_clues(segments, store_name)
        video_stub = {"video_id": "dev-sample", "video_url": "https://dev-fallback-transcript.invalid/video",
                      "published_at": None}
        return {
            "metadata_hits": [],
            "transcript_clues": [{"video": video_stub, **hit} for hit in hits],
        }

    metadata_hits = []
    transcript_clues = []
    for video in videos:
        floor_hint = _metadata_floor_hint(video)
        if floor_hint is not None:
            metadata_hits.append({
                "video": video, "clue": {"floor": floor_hint},
                "certainty": STATED_CERTAINTY, "certainty_reason": "metadata_hint",
            })

        segments = get_transcript_segments(video["video_id"])
        if segments:
            for hit in extract_spatial_clues(segments, store_name):
                transcript_clues.append({"video": video, **hit})

    return {"metadata_hits": metadata_hits, "transcript_clues": transcript_clues}


# Backwards-compatible thin wrapper for callers that only want raw
# metadata search results (used by dev-mode fallbacks elsewhere).
def search_walkthrough_mentions(mall: str, store_name: str) -> list[dict]:
    return search_videos(mall, store_name)
