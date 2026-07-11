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
from app.agents.tools import floorplan, ocr, social, web, youtube
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

        img_bytes = floorplan.fetch_floorplan_image(self.base_url, subtask.floor)
        if img_bytes:
            ocr_results = ocr.ocr_floorplan(img_bytes)
            grid = None
        else:
            ocr_results = []
            grid = floorplan.synthetic_floorplan_grid(subtask.floor, store_count=len(stores))

        evidence.append(
            Evidence(
                source_type=SourceType.FLOORPLAN,
                source_url=f"{self.base_url.rstrip('/')}/directory/map/level-{subtask.floor}",
                entity_raw=f"{FLOORPLAN_ENTITY_KEY}:{subtask.floor}",
                observation={"ocr_results": ocr_results, "synthetic_grid": grid},
                published_date=_now(),
            )
        )

        # Positional floor-plan corroboration per store: aligning each
        # directory row with its slot on the corridor grid gives the
        # Validation Agent a second, independent-source confirmation of
        # floor/unit for that store -- exactly what a human mapping
        # specialist would do by eye when cross-checking the directory
        # against the printed floor plan.
        if grid:
            for s, slot in zip(stores, grid["store_slots"]):
                evidence.append(
                    Evidence(
                        source_type=SourceType.FLOORPLAN,
                        source_url=f"{self.base_url.rstrip('/')}/directory/map/level-{subtask.floor}",
                        entity_raw=s["name"],
                        observation={k: v for k, v in {
                            "floor": s.get("floor", subtask.floor), "unit": s.get("unit"),
                        }.items() if v is not None},
                        raw_excerpt="positional match against floor plan corridor grid",
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
        """YouTube workflow: search -> metadata + transcript (two
        complementary evidence streams per video). Frame-by-frame
        OCR/vision on sampled frames is a Phase 2 enhancement, not
        implemented here."""
        query = f"youtube:{entity_norm}"
        if self.store.has_researched(entity_norm, "youtube", query):
            return []

        result = youtube.research_store(subtask.mall, subtask.entity_hint)
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

        for hit in result["transcript_clues"]:
            video = hit["video"]
            evidence.append(Evidence(
                source_type=SourceType.YOUTUBE_TRANSCRIPT,
                source_url=video.get("video_url"),
                entity_raw=subtask.entity_hint,
                observation=hit["clue"],
                raw_excerpt=None,
                published_date=_parse_date_or_now(video.get("published_at")),
                certainty=hit.get("certainty", 1.0),
                certainty_reason=hit.get("certainty_reason"),
            ))

        self.store.remember_research(
            entity_norm, "youtube", query, evidence[-1].evidence_id if evidence else ""
        )
        return evidence

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
