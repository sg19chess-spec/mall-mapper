-- Supabase Postgres migration for the Indoor Mall Mapping platform.
--
-- This is the Postgres-dialect equivalent of the SQLite DDL in
-- app/store/supabase.py's _TABLES dict (dev mode). Column-for-column
-- parity is intentional: JSON-shaped columns become jsonb (the app passes
-- plain dicts/lists to the Supabase client in production, not pre-encoded
-- strings -- see Store.insert_evidence()'s `else:` branch), TEXT
-- timestamp columns become timestamptz, and SQLite's
-- "INTEGER PRIMARY KEY AUTOINCREMENT" becomes a bigint identity column.
--
-- Run this once against a fresh Supabase project (SQL Editor, or
-- `psql "$SUPABASE_DB_URL" -f db/schema.sql`) before pointing
-- SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY at it.

create table if not exists evidence (
    evidence_id       text primary key,
    source_type       text not null,
    source_url        text,
    entity_raw        text not null,
    observation       jsonb not null,
    raw_excerpt       text,
    observation_date  timestamptz not null,
    published_date    timestamptz not null,
    last_verified     timestamptz not null,
    certainty         real not null default 1.0,
    certainty_reason  text,
    mall              text not null,
    floor             integer not null
);
create index if not exists idx_evidence_mall_floor on evidence (mall, floor);
create index if not exists idx_evidence_entity on evidence (mall, floor, entity_raw);

create table if not exists indoor_features (
    feature_id                text not null,
    version                   integer not null,
    feature_type              text not null,
    geometry                  jsonb,
    properties                jsonb not null,
    confidence_by_attribute   jsonb not null,
    evidence                  jsonb not null,
    valid_from                timestamptz not null,
    valid_until               timestamptz,
    change_reason             text,
    mall                      text not null,
    floor                     integer not null,
    primary key (feature_id, version)
);
create index if not exists idx_indoor_features_mall_floor on indoor_features (mall, floor);
create index if not exists idx_indoor_features_open on indoor_features (mall, floor) where valid_until is null;

create table if not exists review_reports (
    id                        bigint generated always as identity primary key,
    feature_id                text not null,
    iteration                 integer not null,
    confidence_by_attribute   jsonb not null,
    supporting_evidence       jsonb not null,
    conflicting_evidence      jsonb not null,
    recommendation            text not null,
    reason                    text not null,
    explanation               jsonb,
    follow_up_tasks           jsonb not null,
    created_at                timestamptz not null
);
create index if not exists idx_review_reports_feature on review_reports (feature_id);

create table if not exists review_queue (
    feature_id  text primary key,
    issue       text not null,
    evidence    jsonb not null,
    priority    text not null,
    status      text not null,
    resolution  text
);
create index if not exists idx_review_queue_status on review_queue (status);

create table if not exists audit_logs (
    id          bigint generated always as identity primary key,
    job_id      text not null,
    iteration   integer not null,
    feature_id  text,
    event       text not null,
    detail      jsonb,
    created_at  timestamptz not null
);
create index if not exists idx_audit_logs_job on audit_logs (job_id);

create table if not exists research_memory (
    entity_normalized  text not null,
    source_type        text not null,
    query               text not null,
    evidence_id         text not null,
    created_at           timestamptz not null,
    primary key (entity_normalized, source_type, query)
);

create table if not exists change_log (
    id            bigint generated always as identity primary key,
    feature_id    text not null,
    change_type   text not null,
    from_version  integer,
    to_version    integer,
    detail        jsonb,
    created_at    timestamptz not null
);
create index if not exists idx_change_log_feature on change_log (feature_id);

create table if not exists jobs (
    job_id      text primary key,
    mall        text not null,
    floors      jsonb not null,
    status      text not null,
    iteration   integer not null default 0,
    report      jsonb,
    created_at  timestamptz not null,
    updated_at  timestamptz not null
);
