"""JS-rendered page fetch, for directory pages that load their store list via
client-side JavaScript (e.g. Mall of America's /directory). Only imported
lazily -- Playwright + browser binaries are an optional dependency; if
unavailable, callers fall back to the static web.py path / sample dataset.
"""
from __future__ import annotations

import sys

# Common containerized-environment gotchas for headless Chromium:
# --no-sandbox is needed when running as root (our Dockerfile doesn't set a
# non-root USER), and --disable-dev-shm-usage avoids crashes on platforms
# with a small /dev/shm (many cloud container platforms default to 64MB,
# which Chromium's default shared-memory usage can exceed under load).
_LAUNCH_ARGS = ["--no-sandbox", "--disable-dev-shm-usage"]


def fetch_rendered_html(url: str, wait_selector: str | None = None, timeout_ms: int = 15000) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        print(f"[playwright] not installed: {exc}", file=sys.stderr)
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=_LAUNCH_ARGS)
            page = browser.new_page()
            page.goto(url, timeout=timeout_ms)
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=timeout_ms)
            else:
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
            html = page.content()
            browser.close()
            return html
    except Exception as exc:
        # Deliberately not re-raised -- callers treat None as "fall back to
        # the next source" -- but logged so a silent fallback to sample
        # data is diagnosable from server logs instead of being invisible.
        print(f"[playwright] fetch_rendered_html failed for {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
