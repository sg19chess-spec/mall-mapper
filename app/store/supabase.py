"""Postgres-backed system of record, via Supabase.

Tables: evidence, indoor_features, review_reports, review_queue, audit_logs,
research_memory, change_log, jobs. In dev mode these are created locally via
the DDL in _TABLES below (SQLite dialect). In production, run db/schema.sql
(the Postgres-dialect equivalent -- JSON columns as jsonb, TEXT timestamps
as timestamptz) against the Supabase project once before pointing
SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY at it.

Dev-mode fallback: if SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set,
everything is kept in an in-process SQLite file (./dev_data/mall_mapper.db)
instead. Same query surface either way, so the rest of the app never needs
to know which backend it's talking to -- this lets the pipeline be run and
demoed without a live Supabase project.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SUPABASE_URL = os.environ.get("SUPABASE_URL")
_SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

_DEV_DB_PATH = Path(__file__).resolve().parent.parent.parent / "dev_data" / "mall_mapper.db"

_TABLES = {
    "evidence": """
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_url TEXT,
            entity_raw TEXT NOT NULL,
            observation TEXT NOT NULL,
            raw_excerpt TEXT,
            observation_date TEXT NOT NULL,
            published_date TEXT NOT NULL,
            last_verified TEXT NOT NULL,
            certainty REAL NOT NULL DEFAULT 1.0,
            certainty_reason TEXT,
            mall TEXT NOT NULL,
            floor INTEGER NOT NULL
        )
    """,
    "indoor_features": """
        CREATE TABLE IF NOT EXISTS indoor_features (
            feature_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            feature_type TEXT NOT NULL,
            geometry TEXT,
            properties TEXT NOT NULL,
            confidence_by_attribute TEXT NOT NULL,
            evidence TEXT NOT NULL,
            valid_from TEXT NOT NULL,
            valid_until TEXT,
            change_reason TEXT,
            mall TEXT NOT NULL,
            floor INTEGER NOT NULL,
            PRIMARY KEY (feature_id, version)
        )
    """,
    "review_reports": """
        CREATE TABLE IF NOT EXISTS review_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            confidence_by_attribute TEXT NOT NULL,
            supporting_evidence TEXT NOT NULL,
            conflicting_evidence TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            reason TEXT NOT NULL,
            explanation TEXT,
            follow_up_tasks TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """,
    "review_queue": """
        CREATE TABLE IF NOT EXISTS review_queue (
            feature_id TEXT PRIMARY KEY,
            issue TEXT NOT NULL,
            evidence TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL,
            resolution TEXT
        )
    """,
    "audit_logs": """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            feature_id TEXT,
            event TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL
        )
    """,
    "research_memory": """
        CREATE TABLE IF NOT EXISTS research_memory (
            entity_normalized TEXT NOT NULL,
            source_type TEXT NOT NULL,
            query TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (entity_normalized, source_type, query)
        )
    """,
    "change_log": """
        CREATE TABLE IF NOT EXISTS change_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_id TEXT NOT NULL,
            change_type TEXT NOT NULL,
            from_version INTEGER,
            to_version INTEGER,
            detail TEXT,
            created_at TEXT NOT NULL
        )
    """,
    "jobs": """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            mall TEXT NOT NULL,
            floors TEXT NOT NULL,
            status TEXT NOT NULL,
            iteration INTEGER NOT NULL DEFAULT 0,
            report TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Thin query layer. Backed by Supabase Postgres in production, SQLite in dev mode."""

    def __init__(self) -> None:
        self.dev_mode = not (_SUPABASE_URL and _SUPABASE_KEY)
        if self.dev_mode:
            _DEV_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(_DEV_DB_PATH, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            for ddl in _TABLES.values():
                self._conn.execute(ddl)
            self._conn.commit()
            self._client = None
        else:
            from supabase import create_client  # type: ignore

            self._client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
            self._conn = None

    # -- evidence ---------------------------------------------------------

    def insert_evidence(self, evidence: dict, mall: str, floor: int) -> None:
        row = {**evidence, "mall": mall, "floor": floor}
        if self.dev_mode:
            self._conn.execute(
                "INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["evidence_id"], row["source_type"], row.get("source_url"),
                    row["entity_raw"], json.dumps(row["observation"]), row.get("raw_excerpt"),
                    row["observation_date"], row["published_date"], row["last_verified"],
                    row.get("certainty", 1.0), row.get("certainty_reason"), mall, floor,
                ),
            )
            self._conn.commit()
        else:
            self._client.table("evidence").upsert(row).execute()

    def get_evidence_for_entity(self, mall: str, floor: int, entity_normalized: str) -> list[dict]:
        if self.dev_mode:
            cur = self._conn.execute(
                "SELECT * FROM evidence WHERE mall=? AND floor=?", (mall, floor)
            )
            rows = [dict(r) for r in cur.fetchall()]
        else:
            rows = self._client.table("evidence").select("*").eq("mall", mall).eq("floor", floor).execute().data
        out = []
        for r in rows:
            if isinstance(r.get("observation"), str):
                r["observation"] = json.loads(r["observation"])
            if entity_normalized in r["entity_raw"].lower() or entity_normalized == r["entity_raw"].lower():
                out.append(r)
        return out

    def get_all_evidence(self, mall: str, floor: int) -> list[dict]:
        if self.dev_mode:
            cur = self._conn.execute("SELECT * FROM evidence WHERE mall=? AND floor=?", (mall, floor))
            rows = [dict(r) for r in cur.fetchall()]
        else:
            rows = self._client.table("evidence").select("*").eq("mall", mall).eq("floor", floor).execute().data
        for r in rows:
            if isinstance(r.get("observation"), str):
                r["observation"] = json.loads(r["observation"])
        return rows

    # -- research memory ----------------------------------------------------

    def has_researched(self, entity_normalized: str, source_type: str, query: str) -> bool:
        if self.dev_mode:
            cur = self._conn.execute(
                "SELECT 1 FROM research_memory WHERE entity_normalized=? AND source_type=? AND query=?",
                (entity_normalized, source_type, query),
            )
            return cur.fetchone() is not None
        res = (
            self._client.table("research_memory")
            .select("entity_normalized")
            .eq("entity_normalized", entity_normalized)
            .eq("source_type", source_type)
            .eq("query", query)
            .execute()
        )
        return len(res.data) > 0

    def remember_research(self, entity_normalized: str, source_type: str, query: str, evidence_id: str) -> None:
        row = {
            "entity_normalized": entity_normalized, "source_type": source_type,
            "query": query, "evidence_id": evidence_id, "created_at": _now_iso(),
        }
        if self.dev_mode:
            self._conn.execute(
                "INSERT OR REPLACE INTO research_memory VALUES (?,?,?,?,?)",
                tuple(row.values()),
            )
            self._conn.commit()
        else:
            self._client.table("research_memory").upsert(row).execute()

    # -- indoor features ------------------------------------------------------

    def publish_feature(self, feature: dict, mall: str, floor: int) -> None:
        row = {
            "feature_id": feature["feature_id"], "version": feature["version"],
            "feature_type": feature["feature_type"],
            "geometry": json.dumps(feature["geometry"]) if feature.get("geometry") else None,
            "properties": json.dumps(feature["properties"]),
            "confidence_by_attribute": json.dumps(feature["confidence_by_attribute"]),
            "evidence": json.dumps(feature["evidence"]),
            "valid_from": feature["valid_from"], "valid_until": feature.get("valid_until"),
            "change_reason": feature.get("change_reason"), "mall": mall, "floor": floor,
        }
        if self.dev_mode:
            # close previous open version of this feature_id
            self._conn.execute(
                "UPDATE indoor_features SET valid_until=? WHERE feature_id=? AND valid_until IS NULL AND version<?",
                (_now_iso(), feature["feature_id"], feature["version"]),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO indoor_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["feature_id"], row["version"], row["feature_type"], row["geometry"],
                    row["properties"], row["confidence_by_attribute"], row["evidence"],
                    row["valid_from"], row["valid_until"], row["change_reason"], mall, floor,
                ),
            )
            self._conn.commit()
        else:
            self._client.table("indoor_features").upsert(row).execute()

    def get_published_features(self, mall: str, floor: int) -> list[dict]:
        if self.dev_mode:
            cur = self._conn.execute(
                "SELECT * FROM indoor_features WHERE mall=? AND floor=? AND valid_until IS NULL",
                (mall, floor),
            )
            rows = [dict(r) for r in cur.fetchall()]
        else:
            rows = (
                self._client.table("indoor_features").select("*").eq("mall", mall).eq("floor", floor)
                .is_("valid_until", "null").execute().data
            )
        for r in rows:
            for key in ("geometry", "properties", "confidence_by_attribute", "evidence"):
                if isinstance(r.get(key), str):
                    r[key] = json.loads(r[key])
        return rows

    def get_feature_history(self, feature_id: str) -> list[dict]:
        if self.dev_mode:
            cur = self._conn.execute(
                "SELECT * FROM indoor_features WHERE feature_id=? ORDER BY version", (feature_id,)
            )
            return [dict(r) for r in cur.fetchall()]
        return (
            self._client.table("indoor_features").select("*").eq("feature_id", feature_id)
            .order("version").execute().data
        )

    # -- review reports / queue -----------------------------------------------

    def insert_review_report(self, report: dict) -> None:
        row = {
            "feature_id": report["feature_id"], "iteration": report["iteration"],
            "confidence_by_attribute": json.dumps(report["confidence_by_attribute"]),
            "supporting_evidence": json.dumps(report["supporting_evidence"]),
            "conflicting_evidence": json.dumps(report["conflicting_evidence"]),
            "recommendation": report["recommendation"], "reason": report["reason"],
            "explanation": json.dumps(report.get("explanation", [])),
            "follow_up_tasks": json.dumps(report["follow_up_tasks"]),
            "created_at": report["created_at"],
        }
        if self.dev_mode:
            self._conn.execute(
                "INSERT INTO review_reports "
                "(feature_id, iteration, confidence_by_attribute, supporting_evidence, "
                "conflicting_evidence, recommendation, reason, explanation, follow_up_tasks, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                tuple(row.values()),
            )
            self._conn.commit()
        else:
            self._client.table("review_reports").insert(row).execute()

    def upsert_review_item(self, item: dict) -> None:
        row = {
            "feature_id": item["feature_id"], "issue": item["issue"],
            "evidence": json.dumps(item["evidence"]), "priority": item["priority"],
            "status": item["status"], "resolution": item.get("resolution"),
        }
        if self.dev_mode:
            self._conn.execute("INSERT OR REPLACE INTO review_queue VALUES (?,?,?,?,?,?)", tuple(row.values()))
            self._conn.commit()
        else:
            self._client.table("review_queue").upsert(row).execute()

    def get_review_queue(self, status: str = "open") -> list[dict]:
        if self.dev_mode:
            cur = self._conn.execute("SELECT * FROM review_queue WHERE status=?", (status,))
            rows = [dict(r) for r in cur.fetchall()]
        else:
            rows = self._client.table("review_queue").select("*").eq("status", status).execute().data
        for r in rows:
            if isinstance(r.get("evidence"), str):
                r["evidence"] = json.loads(r["evidence"])
        return rows

    # -- audit log ------------------------------------------------------------

    def log_audit(self, job_id: str, iteration: int, event: str, feature_id: str | None = None, detail: Any = None) -> None:
        row = {
            "job_id": job_id, "iteration": iteration, "feature_id": feature_id,
            "event": event, "detail": json.dumps(detail) if detail is not None else None,
            "created_at": _now_iso(),
        }
        if self.dev_mode:
            self._conn.execute(
                "INSERT INTO audit_logs (job_id, iteration, feature_id, event, detail, created_at) "
                "VALUES (?,?,?,?,?,?)",
                tuple(row.values()),
            )
            self._conn.commit()
        else:
            self._client.table("audit_logs").insert(row).execute()

    def get_audit_trail(self, job_id: str) -> list[dict]:
        if self.dev_mode:
            cur = self._conn.execute("SELECT * FROM audit_logs WHERE job_id=? ORDER BY id", (job_id,))
            rows = [dict(r) for r in cur.fetchall()]
        else:
            rows = self._client.table("audit_logs").select("*").eq("job_id", job_id).order("id").execute().data
        for r in rows:
            if isinstance(r.get("detail"), str) and r["detail"]:
                r["detail"] = json.loads(r["detail"])
        return rows

    # -- change log -------------------------------------------------------------

    def log_change(self, feature_id: str, change_type: str, from_version: int | None, to_version: int, detail: Any = None) -> None:
        row = (feature_id, change_type, from_version, to_version, json.dumps(detail) if detail else None, _now_iso())
        if self.dev_mode:
            self._conn.execute(
                "INSERT INTO change_log (feature_id, change_type, from_version, to_version, detail, created_at) "
                "VALUES (?,?,?,?,?,?)",
                row,
            )
            self._conn.commit()
        else:
            self._client.table("change_log").insert({
                "feature_id": feature_id, "change_type": change_type, "from_version": from_version,
                "to_version": to_version, "detail": json.dumps(detail) if detail else None,
                "created_at": _now_iso(),
            }).execute()

    # -- jobs -------------------------------------------------------------------

    def create_job(self, job_id: str, mall: str, floors: list[int]) -> None:
        row = (job_id, mall, json.dumps(floors), "running", 0, None, _now_iso(), _now_iso())
        if self.dev_mode:
            self._conn.execute(
                "INSERT INTO jobs (job_id, mall, floors, status, iteration, report, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                row,
            )
            self._conn.commit()
        else:
            self._client.table("jobs").insert({
                "job_id": job_id, "mall": mall, "floors": json.dumps(floors), "status": "running",
                "iteration": 0, "report": None, "created_at": _now_iso(), "updated_at": _now_iso(),
            }).execute()

    def update_job(self, job_id: str, status: str | None = None, iteration: int | None = None, report: dict | None = None) -> None:
        if self.dev_mode:
            fields, values = [], []
            if status is not None:
                fields.append("status=?"); values.append(status)
            if iteration is not None:
                fields.append("iteration=?"); values.append(iteration)
            if report is not None:
                fields.append("report=?"); values.append(json.dumps(report))
            fields.append("updated_at=?"); values.append(_now_iso())
            values.append(job_id)
            self._conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id=?", values)
            self._conn.commit()
        else:
            patch: dict[str, Any] = {"updated_at": _now_iso()}
            if status is not None:
                patch["status"] = status
            if iteration is not None:
                patch["iteration"] = iteration
            if report is not None:
                patch["report"] = json.dumps(report)
            self._client.table("jobs").update(patch).eq("job_id", job_id).execute()

    def get_job(self, job_id: str) -> dict | None:
        if self.dev_mode:
            cur = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            out = dict(row)
        else:
            data = self._client.table("jobs").select("*").eq("job_id", job_id).execute().data
            if not data:
                return None
            out = data[0]
        out["floors"] = json.loads(out["floors"])
        if out.get("report"):
            out["report"] = json.loads(out["report"])
        return out


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
