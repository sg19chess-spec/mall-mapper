# Indoor Mall Mapping — 5-Agent Geospatial Production System

An evidence-driven, multi-agent pipeline that reconstructs a mall's floor-by-floor
indoor map (stores, floors, categories, unit numbers, geometry) from public sources,
cross-validates every claim against multiple independent pieces of evidence, and
only publishes what's actually corroborated — escalating everything else to human
review instead of guessing.

**Live**: https://mall-mapper.onrender.com · **Source**: https://github.com/sg19chess-spec/mall-mapper

## The problem this solves

Indoor mapping vendors (this project is modeled on the Point Inside case study)
receive hundreds of venue updates a month from each mall — new stores, closures,
relocations — and have to verify each one against multiple sources (official
directory, floor plans, web, social media, YouTube walkthroughs, sometimes phone
calls) before publishing to the map consumers actually navigate by. That
verification work is normally manual and doesn't scale.

This system automates the *verification workflow*, not just the scraping: five
agents mirror the human specialist roles a real geospatial production team would
have, each using ordinary Python tools internally (HTTP/Playwright scraping, OCR,
geometry math, rule checking) — the tools are not agents themselves.

## Architecture

```
POST /run(mall, floors)
   │
   ▼
Agent 1 — Task Intake (Venue Update Coordinator)
   prioritized subtask queue, one per floor
   │
   ▼  for each subtask
Agent 2 — Research (Geo-spatial Research Analyst)
   gathers Evidence — never conclusions — from every relevant source
   │
   ▼
Agent 3 — Validation (Geo-spatial Validation Analyst)
   normalize → resolve → cross-check → confidence → conflicts → explain
   │
   ▼
Agent 4 — Indoor Mapping (Indoor Mapping Specialist)
   builds typed IndoorFeature geometry + indoor topology graph
   │
   ▼
Agent 5 — Publication Review (SME / QA Reviewer)
   Review Report → approve / retry (targeted) / human_review
   │           │                                  │
   │      retry (typed task,                 human_review
   │      e.g. VERIFY_FLOOR)                        │
   │           │                                  │
   │           ▼                                  ▼
   │    back to Agent 2,                  review_queue table
   │    but ONLY for that                  (SME resolves manually)
   │    feature/attribute
   │
   ▼ approve
indoor_features + feature_versions, Change Detection, GeoJSON export
```

The loop is genuinely iterative: Agent 5 can send targeted follow-up tasks back to
Agent 2 (e.g. "re-verify just the floor for this one store") repeatedly, converging
when no new evidence appears, conflicts stop changing, and confidence stabilizes —
not just after a fixed number of passes. If it stagnates with unresolved work
still queued, that work is force-escalated to `human_review` instead of silently
dropped.

### Agent tools (called by agents, not agents themselves)

| Tool | Used by | Purpose |
|---|---|---|
| `agents/tools/web.py` | Research | Official directory scraping (static `httpx` first, Playwright-rendered fallback for JS-heavy sites) |
| `agents/tools/playwright.py` | Research (via web.py) | Headless-browser rendered fetch for JS-rendered pages |
| `agents/tools/floorplan.py` | Research | Floor plan image download; synthetic corridor/slot grid fallback |
| `agents/tools/ocr.py` | Research | `pytesseract` extraction of unit numbers/labels from floor plan images |
| `agents/tools/youtube.py` | Research | YouTube Data API search + transcript extraction; two complementary evidence streams (metadata clues, spoken transcript clues) with a continuous linguistic-certainty scale |
| `agents/tools/social.py` | Research | Social evidence (stub — see Known Limitations) |
| `agents/tools/normalizer.py` | Validation | RapidFuzz-based canonicalization of name variants before entity resolution |
| `agents/tools/rule_engine.py` | Validation, Publication Review | Declarative spatial rules (`must_intersect`, `must_not_overlap`, `centroid_inside`) |
| `agents/tools/geometry.py` | Indoor Mapping | Typed GeoJSON geometry (Point/Polygon/LineString) construction |
| `agents/tools/indoor_graph.py` | Indoor Mapping | NetworkX routable topology graph |
| `agents/tools/spatial_index.py` | (Phase 2) | Shapely STRtree spatial index |

## The evidence model

Every claim is `Evidence`, never a conclusion — only the Validation Agent decides
what to trust:

```python
confidence = source_prior × freshness × completeness × agreement × certainty
```

- **`source_prior`** (`app/schemas/__init__.py::SOURCE_PRIORS`) — a plain
  string-keyed config table, deliberately decoupled from the `SourceType` enum so
  weights can be re-tuned without touching code: official directory 0.45, floor
  plan 0.25, web 0.10, YouTube transcript 0.09, YouTube metadata 0.06, social
  0.05, satellite 0.04, manual phone +0.03 (additive trust boost).
- **`freshness`** — exponential decay from `published_date`, half-life per source
  type.
- **`completeness`** — fraction of expected fields present in the observation.
- **`agreement`** — cross-source consensus on a field's value; when ≥3 *distinct
  source types* agree, an explicit agreement bonus is applied on top.
- **`certainty`** — linguistic hedging, on a continuous scale (`definitely`=1.0
  down to `i guess`=0.25, see `youtube.py::CERTAINTY_LEXICON`), so "I think Apple
  used to be upstairs" contributes less than "The Apple Store is on Level 2."

Conflicts are classified, not just flagged (`ConflictType`: `floor`, `unit`,
`category`, `geometry`, `temporal`, `identity`) — a disagreement between evidence
separated by a large time gap is reclassified `temporal` (probable relocation)
rather than treated as a plain data error.

Spatial reasoning over adjacency: if Evidence says "Apple is next to LEGO" and
LEGO has a confident unit number, Validation infers a starting-hypothesis unit for
Apple (neighbor's number + 1) or, if Apple already has its own unit evidence that
happens to match, treats it as corroboration and boosts confidence.

Every decision is explainable — `ReviewReport.explanation` is a list of
human-readable bullets built by Validation and extended by Publication Review, not
just a numeric score.

## Project structure

```
app/
  api/routes.py          POST /run, GET /status/{job_id}, GET /geojson/{floor}, GET /feature/{feature_id},
                          GET /review-queue, GET /audit/{feature_id}, POST /rerun/{feature_id}
  agents/
    base.py               shared Claude/OpenAI LLM wrapper (not currently called by any agent — see Known Limitations)
    task_intake.py         Agent 1
    research.py             Agent 2
    validation.py            Agent 3
    indoor_mapping.py         Agent 4
    publication_review.py      Agent 5
    tools/                      software the agents use -- not separate agents
  store/
    supabase.py           Postgres client/queries, with a transparent local-SQLite dev-mode fallback
    storage.py              Supabase Storage client, with a transparent local-file dev-mode fallback
  schemas/__init__.py     Evidence, IndoorFeature, ReviewReport, ConflictReport, TaskType, GeometryFeature, ...
  orchestrator.py         coordinates Agents 1-5, retry loop, convergence detection
  eval/                   accuracy.py (directory-agreement, evidence-agreement, geometry-validity metrics),
                          ground_truth.py
  main.py                 FastAPI entrypoint
db/schema.sql             Postgres migration for a real Supabase project
tests/                    62 tests: unit (validation agent, rule engine), integration
                          (publication review), end-to-end (full orchestrator + eval)
requirements.txt          production dependencies
requirements-dev.txt      + pytest
Dockerfile                installs tesseract-ocr + Playwright's Chromium, then the app
render.yaml               Render Web Service config
```

## Running it locally

### Dev mode (no credentials needed)

By default, with no environment variables set, everything runs against a local
SQLite file (`./dev_data/mall_mapper.db`) and local file storage
(`./dev_data/storage/`) instead of Supabase — the exact same code path either way,
just a different backend picked automatically in `store/supabase.py` /
`store/storage.py`.

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium   # only needed for live scraping; OCR needs tesseract-ocr installed separately
uvicorn app.main:app --reload
```

Then either open **http://localhost:8000/ui** — a simple web page (also live at
https://mall-mapper.onrender.com/ui) where you paste a mall name + website URL,
watch the 5 agents' live activity feed (evidence collected, review decisions),
and see the resulting floor map rendered as an SVG, color-coded by confidence
with a hover tooltip — or drive it directly via curl:

```bash
curl -X POST localhost:8000/run -H "Content-Type: application/json" \
  -d '{"mall": "Mall of America", "base_url": "https://www.mallofamerica.com", "floors": [1, 2, 3], "max_iterations": 6}'
curl localhost:8000/status/<job_id>
curl localhost:8000/jobs/<job_id>/trail   # full agent activity timeline
curl "localhost:8000/geojson/2?mall=Mall%20of%20America"
```

### Running the test suite

```bash
python -m pytest tests/ -v
```

62 tests, ~20-30s, no network or credentials required — `tests/test_accuracy_eval.py`
explicitly forces the directory scraper offline (`force_offline_scraping` fixture)
so the suite stays deterministic even when live network access happens to be
available in the environment running it.

### Environment variables (all optional — see fallback behavior)

| Variable | Purpose | If unset |
|---|---|---|
| `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` | Postgres + Storage backend | Falls back to local SQLite + local files |
| `ANTHROPIC_API_KEY` | Claude-backed reasoning (`agents/base.py`, preferred provider) | Not currently exercised by any agent — see Known Limitations |
| `OPENAI_API_KEY` | OpenAI-backed reasoning (`agents/base.py`, fallback provider) | Not currently exercised by any agent — see Known Limitations |
| `YOUTUBE_API_KEY` | Real YouTube Data API search + transcripts | Falls back to a bundled 4-store sample transcript set |
| `INSTAGRAM_GRAPH_API_TOKEN` | Real Instagram Graph API lookups | `social.py` returns no results regardless (permanent stub) |
| `MALL_BASE_URL` | Target mall site | Defaults to `https://www.mallofamerica.com` |
| `MALL_MAPPER_MODEL` | Claude model override | Defaults to `claude-sonnet-5` |
| `MALL_MAPPER_OPENAI_MODEL` | OpenAI model override | Defaults to `gpt-4o-mini` |

## Deployment

**Live**: https://mall-mapper.onrender.com — deployed on Render, backed by a real
Supabase project (Postgres + Storage), running the full pipeline against the real
Mall of America site.

1. **GitHub**: the repo is pushed to https://github.com/sg19chess-spec/mall-mapper.
2. **Supabase**: a real project is provisioned; `db/schema.sql` has been applied
   (all 8 tables confirmed created), and all 7 Storage buckets (`floorplans`,
   `images`, `ocr`, `geojson`, `reports`, `screenshots`, `youtube_frames`) exist.
   To reproduce: create a project at supabase.com, run `db/schema.sql` against
   it, create the same 7 buckets, and grab the project URL + `service_role` key
   (Project Settings → API).
3. **Docker**: `docker build -t mall-mapper .` — installs `tesseract-ocr` and
   Playwright's Chromium alongside the app (this is why Docker is required for
   deployment rather than Render's native Python runtime, which can't install
   those system-level binaries).
4. **Render**: connected via the GitHub repo (manual "New Web Service" flow,
   Docker language auto-detected), with auto-deploy on push to `master`. Secret
   env vars (`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`)
   set in the dashboard's Environment tab. `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`
   intentionally left unset (see Known Limitations).

### Bugs found only by actually deploying (not visible from local dev-mode testing)

Local dev-mode testing uses a throwaway SQLite file and local directories that
auto-create themselves — none of these four surfaced until the app ran against
a real, persistent Supabase project and a real network path:

1. **Missing Storage buckets.** Dev mode auto-creates local directories for each
   bucket; production mode assumes the buckets already exist server-side (Supabase
   Storage buckets don't auto-create). First live `/run` failed with `Bucket not
   found` when exporting GeoJSON/reports. Fixed by creating the 7 buckets directly.
2. **A downstream export failure was clobbering a successful job's status.**
   `orchestrator.run()` already marks a job `"completed"` with its real report as
   its last internal step; `routes.py`'s `_worker()` then does GeoJSON/report
   export *after* that, and a single shared `except` block was overwriting the
   already-`"completed"` status with `{"error": ...}` when the export step failed
   — even though the pipeline itself had fully succeeded (confirmed: 16 evidence
   rows, 4 published features, 8 review reports were correctly in Postgres despite
   `/status` reporting `"failed"`). Fixed by separating the two `try`/`except`
   blocks so an export failure is logged, not treated as a pipeline failure.
3. **Ground truth double-counted across repeated runs.** Evidence persists across
   separate `/run` calls against the same `(mall, floor)` in production, and
   Research always does a fresh broad scrape rather than skipping an
   already-scraped floor. Running the same floor twice inserted a second full set
   of `official_directory` evidence rows for every store, and
   `load_ground_truth_from_evidence` counted each duplicate row as a separate
   "true" store — observed live as `directory_agreement.precision: 3.0`, which is
   mathematically impossible (precision is bounded to [0, 1]). Fixed by
   deduplicating ground truth by normalized store name, keeping the freshest row.
4. **Live scraping was silently falling back to sample data on Render.** Every
   `/run` on Render returned exactly the 4 bundled `SAMPLE_DIRECTORY` floor-1
   stores, never real data — despite live scraping working from a local Docker
   container test. `fetch_rendered_html`'s exception was silently swallowed
   (`except Exception: return None`), so the failure was invisible until logging
   was added. The actual cause, revealed once logged: `page.goto()` uses
   Playwright's default `wait_until="load"`, which blocks on *every* subresource
   finishing — including third-party ad/tracking scripts (doubleclick.net etc.)
   and the Jibestream map SDK visible in the page's own HTML — and was timing out
   at 20s. Fixed by switching to `wait_until="domcontentloaded"` (the code already
   separately waits for the actual content selector via `wait_for_selector`, so
   waiting for ads to finish loading was never necessary) and bumping the timeout
   to 30s for margin.

All four confirmed fixed against the live deployment: a `/run` job now correctly
scrapes real data (`ground_truth_count: 17` on floor 1, vs. the 4-store sample
fallback), publishes what clears the confidence bar, escalates the rest to
`human_review`, and reports valid (≤1.0) accuracy metrics throughout.

## Real-world verification performed

- **Live directory scraping**: confirmed against the actual mallofamerica.com
  site, both locally and from the deployed Render service. It's a Drupal site
  whose directory renders via JavaScript — a static fetch returns the page shell
  with zero store rows, so `web.py::get_store_directory()` falls back to a
  Playwright-rendered fetch, parsed with a site-specific selector set
  (`.card__tile--details`). Successfully scraped 46+ real stores across floors
  1-4 with correct name/category/unit, including deriving floor number from
  MOA's own unit-numbering convention (leading digit of e.g. "228 West Market"
  → floor 2).
- **Real OCR**: MOA's floor plan is an interactive Jibestream vector map, not a
  static image — screenshotted the rendered Map View tab with Playwright and ran
  `pytesseract` against it; correctly extracted real labels ("NORDSTROM",
  "Parking", etc.) at high confidence.
- **Scraping reliability**: the live site is intermittently flaky under
  back-to-back automated requests (likely bot-detection/rate-limiting) — added a
  2-attempt retry in `get_store_directory()`. Render's environment additionally
  needed a `page.goto()` fix (see the deployment bugs above) that local testing
  never surfaced, since local network conditions happened to be fast enough to
  mask it.
- **Full live deployment**: end-to-end pipeline verified against the actual
  Render + Supabase stack — real scraping, real evidence persisted to Postgres,
  real GeoJSON exported to Storage, real confidence-gated publication vs.
  human-review escalation. See the deployment bugs section above for the four
  issues this surfaced that no amount of local/dev-mode testing would have.

## Known limitations

- **Neither `ANTHROPIC_API_KEY` nor `OPENAI_API_KEY` is currently exercised by
  any agent.** `agents/base.py` provides a working, tested (fully mocked, no
  real API calls) provider-agnostic interface — `ask_claude`/`ask_openai`/
  `ask_llm` (picks whichever provider is configured, Anthropic preferred) and
  their `_json` variants — but no agent currently calls them; all extraction
  so far is deterministic (BeautifulSoup selectors, regex). This was a
  deliberate scope decision (see below), not an oversight: the capability is
  real and unit-tested, just not wired into a live agent call path yet.
- **Social evidence (`social.py`) is a permanent stub** — returns no results
  regardless of whether `INSTAGRAM_GRAPH_API_TOKEN` is set. Never wired to a real
  API.
- **`PASS_THRESHOLD` is calibrated to 0.5, not the aspirational 0.75, and this is
  intentional, not temporary.** Verified against live data: most real stores only
  ever get 2 real corroborating sources (official directory + a synthetic
  floor-plan slot), and fields like `category` are structurally single-sourced
  (only the directory tags it) — a stricter bar wouldn't make the pipeline more
  accurate, it would just send more things to `human_review` for lack of any
  additional evidence to raise confidence with. Real single-sourced venue
  updates are *supposed* to land in human review rather than auto-publish,
  mirroring how Point Inside actually staffs SMEs to sign off on updates. The
  system's value is in correctly triaging what needs a human look, not in
  maximizing the auto-publish rate.
- **Deliberately not pursued further** (explicit scope decision, not a gap):
  Instagram/Facebook Graph API integration, more YouTube features, additional
  research-agent source types, alternative agent-orchestration frameworks
  (LangGraph, CrewAI). The next highest-value work is proving the existing
  architecture against live data end-to-end (this session's focus), not
  expanding source coverage.
- **Phase 2, explicitly out of scope for this MVP**: multi-layer indoor features
  (escalators, entrances, restrooms as first-class `IndoorFeature`s beyond
  stores), indoor routing/navigation, full versioning + change-detection UI,
  knowledge graph distinct from the evidence graph, semantic validation rules,
  spatial indexing in active use (`spatial_index.py` exists but isn't called by
  any agent yet).
### Building the Docker image: two real bugs found and fixed

Both only surfaced by actually running the build, not from code review:

1. **`FROM python:3.11-slim` silently resolved to Debian trixie (13)**, a release
   too new for Playwright's dependency installer to recognize — it fell back to
   installing Ubuntu 20.04 package names (`ttf-ubuntu-font-family`,
   `ttf-unifont`) that don't exist on Debian, and the build failed. Fixed by
   pinning to `python:3.11-slim-bookworm` (Debian 12), which Playwright
   officially supports.
2. **A nested f-string with the same quote character reused inside itself**
   (`f'{r['source_type']}...'`) in `validation.py` — legal in Python 3.12+ (PEP
   701 relaxed f-string parsing) but a `SyntaxError` on Python 3.11, which is
   what the Docker image runs, versus the 3.13 interpreter used for local
   development and testing all session. The container crashed on import before
   ever reaching Uvicorn's startup log. Fixed by building the string with a
   plain generator expression instead of a nested f-string.

After both fixes: `docker build` succeeds (1.42GB image), a container built
from it starts cleanly, responds on `/`, and successfully ran a full pipeline
job end-to-end inside the container — including live Playwright-driven
scraping of the real Mall of America site from within the container (not just
locally), confirming Chromium and its dependencies are correctly installed in
the image.

## Verification checklist (what's actually been proven vs. assumed)

| Claim | Status |
|---|---|
| 5-agent pipeline logic is correct | ✅ 62 passing tests |
| Confidence/conflict/spatial-reasoning math is correct | ✅ Unit-tested directly against `ValidationAgent` |
| Geometry rule checks correctly gate publication | ✅ Unit + integration tested; one real bug found and fixed (`floor_boundary` self-inclusion) |
| Live scraping works against a real, JS-rendered mall site | ✅ Verified locally, inside a local Docker container, and on the live Render deployment |
| Real OCR works against a real floor plan | ✅ Manually verified |
| Full pipeline publishes real scraped data end-to-end | ✅ Verified live: real evidence → real confidence-gated publication vs. human-review escalation, with valid (≤1.0) accuracy metrics |
| Postgres schema is valid | ✅ Applied to a real Supabase project; all 8 tables + 7 Storage buckets confirmed created |
| Docker image builds successfully | ✅ Builds, and a container from it runs a full pipeline job successfully |
| Service runs on Render | ✅ Live at https://mall-mapper.onrender.com, auto-deploying from `master` |

Four additional production-only bugs were found and fixed during the live
deployment itself (missing Storage buckets, job status clobbering, ground
truth double-counting, and a Playwright navigation hang) — see the deployment
section above for details. None of these were visible from local dev-mode
testing, which is itself a finding: dev-mode fallbacks are good for fast
iteration but cannot substitute for testing against the real, persistent,
network-exposed production path.
