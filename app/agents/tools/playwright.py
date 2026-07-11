"""JS-rendered page fetch, for directory pages that load their store list via
client-side JavaScript (e.g. Mall of America's /directory). Only imported
lazily -- Playwright + browser binaries are an optional dependency; if
unavailable, callers fall back to the static web.py path / sample dataset.
"""
from __future__ import annotations


def fetch_rendered_html(url: str, wait_selector: str | None = None, timeout_ms: int = 15000) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, timeout=timeout_ms)
            if wait_selector:
                page.wait_for_selector(wait_selector, timeout=timeout_ms)
            else:
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None
