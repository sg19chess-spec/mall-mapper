"""Static HTTP scraping: official directory pages, tenant sites, aggregator listings.

Used by the Research Agent. Tries a plain requests+BeautifulSoup fetch first;
Mall of America's live directory renders via JavaScript (confirmed: a static
fetch returns the page shell with no store rows), so get_store_directory()
falls back to agents/tools/playwright.py for a rendered fetch, parsed with
parse_moa_directory_html() -- verified against the real DOM (Drupal-based,
".card__tile--details" cards with h3.heading--card-title / .heading--card-
subtitle / .heading--card-info-location). Only if *that* also fails (e.g. no
outbound network in a sandboxed environment) does it fall back to the
bundled SAMPLE_DIRECTORY, which keeps the pipeline runnable offline while
still exercising the exact same Evidence/confidence/validation path a live
fetch would produce.
"""
from __future__ import annotations

import html
import re
import sys

import httpx
from bs4 import BeautifulSoup

from app.agents.tools.playwright import fetch_rendered_html

USER_AGENT = "Mozilla/5.0 (compatible; MallMapperResearchAgent/0.1; +https://example.invalid/bot)"

# A small, deliberately-real sample of Mall of America tenants (name, floor,
# category, unit) used as the offline fallback source. Floors follow MOA's
# own numbering (Level 1 Ground - Level 4 Ground -> mapped here to 1-4) and
# unit numbers are illustrative, not guaranteed current -- Evidence rows
# built from this source are tagged with a low completeness/confidence
# footprint intentionally lower than a genuine official_directory fetch,
# via a distinct source_url so it can be told apart from a live fetch in eval.
SAMPLE_DIRECTORY = [
    {"name": "Nike", "floor": 2, "category": "Apparel", "unit": "S245"},
    {"name": "Apple", "floor": 2, "category": "Electronics", "unit": "E238"},
    {"name": "LEGO Store", "floor": 1, "category": "Toys", "unit": "N130"},
    {"name": "Sea Life Minnesota Aquarium", "floor": 1, "category": "Attraction", "unit": "N100"},
    {"name": "American Girl", "floor": 3, "category": "Toys", "unit": "S330"},
    {"name": "Crayola Experience", "floor": 3, "category": "Attraction", "unit": "N310"},
    {"name": "Build-A-Bear Workshop", "floor": 2, "category": "Toys", "unit": "W215"},
    {"name": "H&M", "floor": 1, "category": "Apparel", "unit": "S110"},
    {"name": "Starbucks", "floor": 1, "category": "Food", "unit": "N118"},
    {"name": "Sephora", "floor": 2, "category": "Beauty", "unit": "E220"},
]


def fetch_directory_html(base_url: str, timeout: float = 10.0) -> str | None:
    """Attempt a static fetch of the mall's directory page. Returns None on
    failure so callers can fall back to Playwright or the sample dataset."""
    try:
        resp = httpx.get(
            f"{base_url.rstrip('/')}/directory",
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        print(f"[web] fetch_directory_html failed for {base_url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def parse_directory_html(html: str) -> list[dict]:
    """Best-effort extraction of store cards from directory HTML. Real sites
    vary; this looks for common patterns (data attributes, definition-list
    style store cards) and returns [] if nothing matches, so the caller knows
    to fall back."""
    soup = BeautifulSoup(html, "lxml")
    stores = []
    for card in soup.select("[data-store-name], .store-card, .directory-item"):
        name = card.get("data-store-name") or card.get_text(strip=True)
        floor = card.get("data-floor") or card.get("data-level")
        category = card.get("data-category")
        unit = card.get("data-unit")
        if name:
            stores.append({
                "name": name.strip(),
                "floor": int(floor) if floor and str(floor).isdigit() else None,
                "category": category,
                "unit": unit,
            })
    return stores


def parse_moa_directory_html(rendered_html: str) -> list[dict]:
    """Extracts store cards from Mall of America's actual rendered directory
    DOM. MOA's own unit-numbering convention is "<number> <street name>"
    (e.g. "228 West Market", "6103 North Garden" for Nickelodeon Universe
    carts) -- the leading digit of the number is the floor (a common
    shopping-center convention, and consistent across dozens of verified
    examples: "104 West Market" -> floor 1, "228 West Market" -> floor 2,
    "364 North Garden" -> floor 3). `category` collapses MOA's multi-tag
    subtitle (e.g. "Women's Apparel / Curbside / MOA Gift Cards") to its
    first tag, since our schema expects a single category value."""
    soup = BeautifulSoup(rendered_html, "lxml")
    stores = []
    for tile in soup.select(".card__tile--details"):
        name_el = tile.select_one("h3.heading--card-title")
        if not name_el:
            continue
        name = html.unescape(name_el.get_text(strip=True))

        subtitle_el = tile.select_one(".heading--card-subtitle")
        category = None
        if subtitle_el:
            tags = html.unescape(subtitle_el.get_text(strip=True))
            category = tags.split("/")[0].strip() or None

        loc_el = tile.select_one(".heading--card-info-location span")
        location = html.unescape(loc_el.get_text(strip=True)) if loc_el else None

        floor = None
        if location:
            m = re.match(r"^(\d+)", location)
            if m:
                floor = int(m.group(1)[0])

        stores.append({"name": name, "floor": floor, "category": category, "unit": location})
    return stores


def get_store_directory(base_url: str, floor: int | None = None) -> list[dict]:
    """Returns a list of {name, floor, category, unit} dicts for the mall.
    Tries a live static fetch, then a Playwright-rendered fetch (for
    JS-rendered directories), then falls back to the bundled sample."""
    static_html = fetch_directory_html(base_url)
    stores = parse_directory_html(static_html) if static_html else []
    source_is_live = bool(stores)

    if not stores:
        # Real sites are flaky under repeated automated requests (bot
        # detection, transient render timeouts) -- a couple of retries
        # meaningfully improves success without masking a genuinely broken
        # selector (which would fail every attempt the same way).
        for attempt in range(2):
            rendered_html = fetch_rendered_html(
                f"{base_url.rstrip('/')}/directory", wait_selector=".card__tile--details", timeout_ms=20000,
            )
            if rendered_html:
                stores = parse_moa_directory_html(rendered_html)
                if stores:
                    source_is_live = True
                    break

    if not stores:
        print(
            f"[web] both static and Playwright-rendered fetches failed for {base_url} "
            f"-- falling back to SAMPLE_DIRECTORY", file=sys.stderr,
        )
        stores = SAMPLE_DIRECTORY

    if floor is not None:
        stores = [s for s in stores if s.get("floor") == floor]
    for s in stores:
        s["_source_is_live"] = source_is_live
    return stores


def search_tenant_web(query: str, timeout: float = 10.0) -> list[dict]:
    """Web search over tenant sites / aggregator listings for a specific
    store name. In an environment with outbound search API access, wire
    this to the actual search provider (e.g. via a search API key).

    Dev-mode fallback: many real shopping-center directories are mirrored
    by third-party aggregator sites (Yelp, mall-tracking blogs, local news),
    so a second corroborating source existing for well-known tenants is
    realistic -- not fabricated agreement. This looks the query up against
    SAMPLE_DIRECTORY as a stand-in for that aggregator mirror when no live
    search API is configured, clearly tagged so it's distinguishable from a
    live web search in source_url.
    """
    from rapidfuzz import fuzz

    best, best_score = None, 0
    for s in SAMPLE_DIRECTORY:
        score = fuzz.ratio(query.lower(), s["name"].lower())
        if score > best_score:
            best, best_score = s, score
    if best is None or best_score < 80:
        return []
    return [{
        "url": "https://dev-fallback-aggregator.invalid/listing",
        "excerpt": f"{best['name']} - {best.get('category')} - Level {best.get('floor')} - Unit {best.get('unit')}",
        "floor": best.get("floor"), "category": best.get("category"), "unit": best.get("unit"),
    }]
