"""Pydantic data model shared by all five agents.

Evidence flows: Research (writes Evidence) -> Validation (reads Evidence,
writes confidence) -> Indoor Mapping (writes IndoorFeature) -> Publication
Review (writes ReviewReport, decides pass/retry/human_review).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SourceType(str, Enum):
    OFFICIAL_DIRECTORY = "official_directory"
    FLOORPLAN = "floorplan"
    WEB = "web"
    SOCIAL = "social"
    # YouTube is split into two complementary evidence sources: metadata
    # (title/description/upload date -- weaker, visual/context clues only)
    # and transcript (spoken mentions of floor/adjacency -- stronger, an
    # actual verbal observation). Kept distinct rather than one YOUTUBE
    # source_type so Validation can weight them differently.
    YOUTUBE_METADATA = "youtube_metadata"
    YOUTUBE_TRANSCRIPT = "youtube_transcript"
    SATELLITE = "satellite"
    MANUAL_PHONE = "manual_phone"


# Source-reliability configuration table, used in the confidence formula:
#   confidence = source_prior x freshness x completeness x agreement x source_reliability
# Deliberately a plain string-keyed dict, not tied to the SourceType enum
# members directly, so these weights can be re-tuned (e.g. after looking at
# eval results) without touching the enum or any code that imports it. Keys
# must match SourceType values, which get_source_prior()/get_source_half_life()
# enforce at lookup time.
SOURCE_PRIORS: dict[str, float] = {
    "official_directory": 0.45,
    "floorplan": 0.25,
    "web": 0.10,
    "youtube_transcript": 0.09,  # a spoken observation, stronger than title/description hints
    "youtube_metadata": 0.06,
    "social": 0.05,
    "satellite": 0.04,
    "manual_phone": 0.03,  # additive trust boost, not a multiplicative prior
}

# Freshness half-life (days) per source type, used for exponential decay of confidence.
SOURCE_HALF_LIFE_DAYS: dict[str, float] = {
    "official_directory": 3650,  # effectively "always fresh" -- fetched live
    "floorplan": 1825,
    "web": 365,
    "youtube_transcript": 180,
    "youtube_metadata": 180,
    "social": 90,
    "satellite": 1825,
    "manual_phone": 365,
}


def get_source_prior(source_type: "SourceType | str") -> float:
    key = source_type.value if isinstance(source_type, SourceType) else source_type
    return SOURCE_PRIORS[key]


def get_source_half_life(source_type: "SourceType | str") -> float:
    key = source_type.value if isinstance(source_type, SourceType) else source_type
    return SOURCE_HALF_LIFE_DAYS[key]


class FeatureType(str, Enum):
    STORE = "store"
    ESCALATOR = "escalator"
    ELEVATOR = "elevator"
    CORRIDOR = "corridor"
    RESTROOM = "restroom"
    EXIT = "exit"
    ATM = "atm"
    PARKING = "parking"
    INFORMATION_DESK = "information_desk"
    FOOD_COURT = "food_court"
    ENTRANCE = "entrance"
    FIRE_EQUIPMENT = "fire_equipment"
    ACCESSIBILITY_FEATURE = "accessibility_feature"
    WAYFINDING_SIGN = "wayfinding_sign"
    # A named landmark/anchor tenant (Nordstrom, Nickelodeon Universe, a
    # parking rotunda, ...) whose real position is read from the mall's own
    # map. Drawn as the map's reference backbone, distinct from tenant stores.
    ANCHOR = "anchor"


class TaskType(str, Enum):
    VERIFY_EXISTENCE = "verify_existence"
    VERIFY_FLOOR = "verify_floor"
    VERIFY_UNIT = "verify_unit"
    VERIFY_CATEGORY = "verify_category"
    VERIFY_GEOMETRY = "verify_geometry"
    VERIFY_ENTRANCE = "verify_entrance"


# Which subtask fields each task type targets -- used by the Research Agent to
# know what it's being asked to (re-)gather evidence for.
TASK_TARGET_FIELDS: dict[TaskType, list[str]] = {
    TaskType.VERIFY_EXISTENCE: [],
    TaskType.VERIFY_FLOOR: ["floor"],
    TaskType.VERIFY_UNIT: ["unit"],
    TaskType.VERIFY_CATEGORY: ["category"],
    TaskType.VERIFY_GEOMETRY: ["geometry"],
    TaskType.VERIFY_ENTRANCE: ["entrance"],
}


class GeometryType(str, Enum):
    POINT = "Point"
    POLYGON = "Polygon"
    LINE_STRING = "LineString"


class GeometryFeature(BaseModel):
    """A typed, floor-local geometry (image-pixel-space CRS per floor plan)."""

    type: GeometryType
    coordinates: list  # nested per GeoJSON spec for the given type
    floor: int


class Evidence(BaseModel):
    """A single observation from one source. Never a conclusion."""

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: SourceType
    source_url: str | None = None
    entity_raw: str
    observation: dict = Field(default_factory=dict)
    raw_excerpt: str | None = None
    observation_date: datetime = Field(default_factory=_now)
    published_date: datetime = Field(default_factory=_now)
    last_verified: datetime = Field(default_factory=_now)
    # Linguistic certainty of the underlying statement, independent of the
    # source-type prior: "The Apple Store is on Level 2" is certainty=1.0,
    # "I think Apple used to be upstairs" is hedged and should contribute
    # less. A continuous scale (not just hedged/not-hedged) -- see
    # agents/tools/youtube.py's CERTAINTY_LEXICON. Detected by the
    # acquisition tool and multiplied into the confidence contribution by
    # the Validation Agent (Research observes; Validation decides how much
    # to trust it).
    certainty: float = 1.0
    # Which phrase drove the certainty score, e.g. "hedge_phrase: might" or
    # "stated_as_fact" -- purely for audit/debugging, not used in the math.
    certainty_reason: str | None = None

    def freshness(self, now: datetime | None = None) -> float:
        now = now or _now()
        half_life = get_source_half_life(self.source_type)
        age_days = max((now - self.published_date).total_seconds() / 86400, 0.0)
        # exponential decay: 0.5 ** (age / half_life)
        return 0.5 ** (age_days / half_life)


class EvidenceRef(BaseModel):
    """Lightweight pointer to an Evidence row, used inside ReviewReport/IndoorFeature."""

    evidence_id: str
    source_type: SourceType
    confidence_contribution: float
    certainty: float = 1.0
    certainty_reason: str | None = None


class ConflictType(str, Enum):
    IDENTITY = "identity"
    FLOOR = "floor"
    UNIT = "unit"
    CATEGORY = "category"
    GEOMETRY = "geometry"
    # Sources disagree not because one is wrong, but because the venue
    # changed between when they were captured -- e.g. an old evidence row
    # and a fresh one give different floors, and the age gap between them
    # roughly matches a plausible relocation window. Surfaced separately
    # from a plain FLOOR/UNIT conflict so a human reviewer knows to check
    # "did this move?" rather than "which source is wrong?".
    TEMPORAL = "temporal"


class ConflictReport(BaseModel):
    entity: str
    field: str
    conflict_type: ConflictType = ConflictType.IDENTITY
    values: list[dict]  # [{"source_type": ..., "value": ..., "evidence_id": ...}, ...]
    detected_at: datetime = Field(default_factory=_now)


class IndoorFeature(BaseModel):
    """The fundamental published object -- a Store is IndoorFeature(feature_type='store')."""

    feature_id: str
    feature_type: FeatureType
    geometry: GeometryFeature | None = None
    properties: dict = Field(default_factory=dict)  # name, category, unit, hours, ...
    confidence_by_attribute: dict[str, float] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    version: int = 1
    valid_from: datetime = Field(default_factory=_now)
    valid_until: datetime | None = None
    change_reason: str | None = None


class ReviewReport(BaseModel):
    feature_id: str
    confidence_by_attribute: dict[str, float]
    supporting_evidence: list[EvidenceRef]
    conflicting_evidence: list[EvidenceRef]
    recommendation: Literal["pass", "retry", "human_review"]
    reason: str
    # Short human-readable bullets a reviewer can read without re-deriving
    # the math, e.g. ["Official directory and floor plan agree on floor.",
    # "Transcript corroborates location.", "No conflicting evidence found."]
    # -- built by the Validation Agent, passed through unchanged here.
    explanation: list[str] = Field(default_factory=list)
    follow_up_tasks: list[TaskType] = Field(default_factory=list)
    iteration: int = 1
    created_at: datetime = Field(default_factory=_now)


class ReviewItem(BaseModel):
    feature_id: str
    issue: str
    evidence: list[EvidenceRef]
    priority: Literal["low", "medium", "high"]
    status: Literal["open", "in_review", "resolved"] = "open"
    resolution: str | None = None


class Subtask(BaseModel):
    """One unit of work created by the Task Intake Agent (or a retry from Publication Review)."""

    subtask_id: str = Field(default_factory=lambda: str(uuid4()))
    mall: str
    floor: int
    entity_hint: str | None = None  # a specific store name, when this is a targeted retry
    task_type: TaskType = TaskType.VERIFY_EXISTENCE
    priority: Literal["low", "medium", "high"] = "medium"
    iteration: int = 1


class RunConfig(BaseModel):
    mall: str
    floors: list[int]
    max_iterations: int = 6
    confidence_convergence_delta: float = 0.01
