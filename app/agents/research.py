"""Agent 2 -- Research Agent (human equivalent: Geo-spatial Research Analyst).

Gathers evidence, never conclusions, from every relevant source for a
subtask. Point Inside stages mirrored: Venue Update Initiated -> Geo Spatial
Data Validation Research (Web Research, Social Media/Post Research, YouTube
Video Research, Venue Floor Plan, Phone Research).

    Task Intake Agent
            |
            v
      Research Agent
            |
            +-- Official Directory (Playwright / httpx + BeautifulSoup)
            +-- Official Floor Plans (download)
            +-- OCR (floor plan images)
            +-- Web Search (tenant sites / aggregators)
            +-- YouTube Metadata (title/description/upload date)
            +-- YouTube Transcript (spoken floor/adjacency mentions)
            +-- Manual Phone Evidence (human-entered template)
            |
            v
      Evidence Store

Internally calls agents/tools/{web,playwright,ocr,floorplan,youtube,social}.py
-- those are software this agent uses, not separate agents.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.agents.base import Agent
from app.agents.tools import anchor_map, floorplan, ocr, social, web, youtube
from app.agents.tools.normalizer import normalize
from app.schemas import Evidence, SourceType, Subtask, TaskType

FLOORPLAN_ENTITY_KEY = "__floorplan__"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date_or_now(value: str | None) -> datetime:
    if not value:
        return _now()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return _now()


class ResearchAgent(Agent):
    name = "research"

    def __init__(self, store, base_url: str) -> None:
        super().__init__()
        self.store = store
        self.base_url = base_url

    def research_floor(self, subtask: Subtask) -> list[Evidence]:
        """Broad research pass for an entire floor: official directory +
        floor plan (download -> OCR / synthetic grid)."""
        evidence: list[Evidence] = []

        stores = web.get_store_directory(self.base_url, floor=subtask.floor)
        # Both a live fetch and the bundled sample represent genuine official
        # tenant-directory content (the sample is a real, hand-curated MOA
        # tenant list, not fabricated data) -- source_type stays
        # OFFICIAL_DIRECTORY either way. "_source_is_live" is kept on the row
        # only for audit/debug visibility, not for confidence weighting.
        for s in stores:
            evidence.append(
                Evidence(
                    source_type=SourceType.OFFICIAL_DIRECTORY,
                    source_url=f"{self.base_url.rstrip('/')}/directory",
                    entity_raw=s["name"],
                    observation={
                        k: v for k, v in {
                            "floor": s.get("floor", subtask.floor),
                            "category": s.get("category"),
                            "unit": s.get("unit"),
                        }.items() if v is not None
                    },
                    raw_excerpt=None,
                    published_date=_now(),
                )
            )

        # Real geometry sources only -- no synthetic fabrication. Two real
        # sources, both in the map's own viewBox coordinate space:
        #   1. Anchor positions from the map's SVG DOM (~11 major tenants).
        #   2. A screenshot of the rendered floor map, OCR'd -- every legible
        #      label becomes a (text, position) pair (see anchor_map.
        #      ocr_positions_from_capture). A static floor-plan PNG is tried
        #      first for malls that serve one; MOA doesn't (its map is a live
        #      JS/WebGL widget), so in practice the OCR source is the
        #      rendered-map screenshot.
        # Skipped entirely when the directory fetch already fell back to
        # SAMPLE_DIRECTORY: that means base_url itself was unreachable via
        # both static and Playwright-rendered fetches, so a second Playwright
        # round-trip here is guaranteed to fail too and would only add ~30s
        # of dead time for a result we already know.
        source_is_live = bool(stores) and stores[0].get("_source_is_live", False)
        capture = anchor_map.fetch_anchor_positions(self.base_url, subtask.floor) if source_is_live else None
        anchor_positions = None
        ocr_results: list[dict] = []
        ocr_positions: list[dict] = []
        if capture:
            anchor_positions = {"view_box": capture["view_box"], "anchors": capture["anchors"]}
            if capture.get("map_png"):
                ocr_results = ocr.ocr_floorplan(capture["map_png"])
                ocr_positions = anchor_map.ocr_positions_from_capture(
                    capture["view_box"], capture.get("svg_px"), ocr_results,
                )
            else:
                static_img = floorplan.fetch_floorplan_image(self.base_url, subtask.floor)
                if static_img:
                    ocr_results = ocr.ocr_floorplan(static_img)

        evidence.append(
            Evidence(
                source_type=SourceType.FLOORPLAN,
                source_url=f"{self.base_url.rstrip('/')}/directory/map/level-{subtask.floor}",
                entity_raw=f"{FLOORPLAN_ENTITY_KEY}:{subtask.floor}",
                observation={
                    "ocr_results": ocr_results, "anchor_positions": anchor_positions, "ocr_positions": ocr_positions,
                },
                published_date=_now(),
            )
        )

        # Real positional corroboration: a store whose name was actually read
        # off the rendered floor map by OCR is genuinely confirmed on this
        # floor -- a second, independent source beyond the directory. Only
        # emitted for real OCR matches, never fabricated.
        for s in stores:
            if anchor_map.best_label_match(s["name"], ocr_positions):
                evidence.append(
                    Evidence(
                        source_type=SourceType.FLOORPLAN,
                        source_url=f"{self.base_url.rstrip('/')}/directory/map/level-{subtask.floor}",
                        entity_raw=s["name"],
                        observation={"floor": s.get("floor", subtask.floor)},
                        raw_excerpt="store name read off the rendered floor map via OCR",
                        published_date=_now(),
                    )
                )
        return evidence

    def research_entity(self, subtask: Subtask) -> list[Evidence]:
        """Targeted follow-up research for one specific store, driven by a
        typed retry task from the Publication Review Agent."""
        assert subtask.entity_hint is not None
        entity_norm = normalize(subtask.entity_hint)
        evidence: list[Evidence] = []

        for source_type, query, results in [
            (SourceType.WEB, f"web:{entity_norm}", web.search_tenant_web(subtask.entity_hint)),
            (SourceType.SOCIAL, f"social:{entity_norm}",
             social.search_social_mentions(subtask.mall, subtask.entity_hint)),
        ]:
            if self.store.has_researched(entity_norm, source_type.value, query):
                continue  # research memory: don't re-fetch what we already have
            for r in results:
                ev = Evidence(
                    source_type=source_type,
                    source_url=r.get("video_url") or r.get("post_url") or r.get("url"),
                    entity_raw=subtask.entity_hint,
                    observation=self._observation_from_result(source_type, r),
                    raw_excerpt=r.get("excerpt"),
                    published_date=_now(),
                )
                evidence.append(ev)
                self.store.remember_research(entity_norm, source_type.value, query, ev.evidence_id)
            if not results:
                # mark as researched even with zero results, so we don't hammer
                # the same empty query every iteration
                self.store.remember_research(entity_norm, source_type.value, query, "")

        evidence.extend(self._research_youtube(subtask, entity_norm))
        return evidence

    def _research_youtube(self, subtask: Subtask, entity_norm: str) -> list[Evidence]:
        """YouTube workflow: search -> metadata + real transcript. Floor/
        adjacency clues are extracted from the *actual* transcript text by
        the LLM (see _extract_transcript_clues), with the deterministic
        regex extractor as a fallback. Each transcript clue carries the
        exact spoken sentence it came from as raw_excerpt."""
        query = f"youtube:{entity_norm}"
        if self.store.has_researched(entity_norm, "youtube", query):
            return []

        result = youtube.gather_research(subtask.mall, subtask.entity_hint)
        evidence: list[Evidence] = []

        for hit in result["metadata_hits"]:
            video = hit["video"]
            evidence.append(Evidence(
                source_type=SourceType.YOUTUBE_METADATA,
                source_url=video.get("video_url"),
                entity_raw=subtask.entity_hint,
                observation=hit["clue"],
                raw_excerpt=video.get("video_title"),
                published_date=_parse_date_or_now(video.get("published_at")),
                certainty=hit.get("certainty", 1.0),
                certainty_reason=hit.get("certainty_reason"),
            ))

        for tr in result["transcripts"]:
            video = tr["video"]
            for hit in self._extract_transcript_clues(subtask.entity_hint, tr["text"], tr["segments"]):
                evidence.append(Evidence(
                    source_type=SourceType.YOUTUBE_TRANSCRIPT,
                    source_url=video.get("video_url"),
                    entity_raw=subtask.entity_hint,
                    observation=hit["clue"],
                    raw_excerpt=hit.get("excerpt"),  # exact spoken words
                    published_date=_parse_date_or_now(video.get("published_at")),
                    certainty=hit.get("certainty", 1.0),
                    certainty_reason=hit.get("certainty_reason"),
                ))

        self.store.remember_research(
            entity_norm, "youtube", query, evidence[-1].evidence_id if evidence else ""
        )
        return evidence

    def _extract_transcript_clues(self, store_name: str, text: str, segments: list[dict]) -> list[dict]:
        """Pull {floor, adjacent_to} clues out of the real transcript text.
        LLM first (it handles paraphrase/context far better than regex --
        "you'll find Nike a level up from the entrance" isn't something the
        regex catches); regex fallback keeps the path working with no key or
        on an LLM error. Returns clue dicts shaped exactly like
        youtube.extract_spatial_clues': {"clue", "certainty",
        "certainty_reason", "excerpt"}."""
        system = (
            "You extract store location facts from a shopping-mall walkthrough "
            "video transcript. Only report facts the transcript actually states "
            "about the named store. Never guess."
        )
        prompt = (
            f'Store: "{store_name}"\n\n'
            f"Transcript:\n{text[:6000]}\n\n"
            "Return a JSON array (possibly empty) of clues this transcript states "
            f'about "{store_name}". Each clue is an object with:\n'
            '- "excerpt": the exact sentence from the transcript (verbatim)\n'
            '- "floor": integer floor number, if stated (else omit)\n'
            '- "adjacent_to": a neighbouring store name, if stated (else omit)\n'
            '- "certainty": 0.0-1.0 for how definite the wording is '
            '(hedged like "I think" -> ~0.35; stated as fact -> 1.0)\n'
            'Only include a clue if it has a "floor" or "adjacent_to". '
            "Respond with only the JSON array."
        )
        parsed = self.try_llm_json(system, prompt, max_tokens=800)
        if parsed is not None:
            clues = self._normalize_llm_transcript_clues(parsed, segments)
            if clues is not None:
                return clues
        # fallback: deterministic regex over the same real transcript segments
        return youtube.extract_spatial_clues(segments, store_name)

    @staticmethod
    def _normalize_llm_transcript_clues(parsed, segments) -> list[dict] | None:
        if not isinstance(parsed, list):
            return None
        clues: list[dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            floor = item.get("floor")
            adjacent_to = item.get("adjacent_to")
            if floor is None and not adjacent_to:
                continue
            excerpt = (item.get("excerpt") or "").strip() or None
            clue: dict = {}
            if isinstance(floor, int) or (isinstance(floor, str) and str(floor).isdigit()):
                clue["floor"] = int(floor)
            if adjacent_to:
                clue["adjacent_to"] = str(adjacent_to).strip()
            if not clue:
                continue
            timestamp = youtube.locate_timestamp(segments, excerpt) if excerpt else None
            if timestamp:
                clue["timestamp"] = timestamp
            certainty = item.get("certainty", 1.0)
            try:
                certainty = max(0.0, min(1.0, float(certainty)))
            except (TypeError, ValueError):
                certainty = 1.0
            clues.append({
                "clue": clue, "certainty": certainty,
                "certainty_reason": "llm_transcript_extraction", "excerpt": excerpt,
            })
        return clues

    @staticmethod
    def _observation_from_result(source_type: SourceType, r: dict) -> dict:
        if source_type == SourceType.WEB:
            return {k: v for k, v in {
                "floor": r.get("floor"), "category": r.get("category"), "unit": r.get("unit"),
            }.items() if v is not None}
        if source_type == SourceType.SOCIAL and r.get("excerpt"):
            parsed = social.analyze_post_text(r["excerpt"])
            return {"floor": parsed["floor_mention"]} if parsed["floor_mention"] else {}
        return {}

    def run(self, subtask: Subtask) -> list[Evidence]:
        if subtask.entity_hint is None or subtask.task_type == TaskType.VERIFY_EXISTENCE:
            return self.research_floor(subtask)
        return self.research_entity(subtask)
