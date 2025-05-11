"""
Vision-grounded Playwright helpers for browser automation.
"""
from __future__ import annotations
import os
import json
import textwrap
import asyncio
import logging
from typing import Dict, Any, Optional, Union, Callable
from playwright.async_api import async_playwright, Page
from vision_service import VisionService, VisionServiceError

# Configure logging
logger = logging.getLogger(__name__)

# ───────── global singletons ─────────
_pw, _browser, _page = None, None, None
_vision_service = None


# ───────── helpers ─────────
async def get_vision_service() -> VisionService:
    """Return a singleton vision service instance."""
    global _vision_service
    if _vision_service is None:
        vision_model = os.environ.get("VISION_MODEL", "gpt-4o")
        _vision_service = VisionService(model_name=vision_model)
    return _vision_service


async def _get_page() -> Page:
    """Return a page with a locked viewport (1280×800, DPR=1)."""
    global _pw, _browser, _page
    if _page:
        return _page

    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(
        headless=os.environ.get("PW_HEADLESS", "true").lower() != "false"
    )

    context = await _browser.new_context(
        viewport={"width": 1280, "height": 800},
        device_scale_factor=1   # ⇒ CSS px == screenshot px
    )
    _page = await context.new_page()
    return _page


# ───────── low‑level guards ─────────
async def safe_click(x: int, y: int, must_contain: str | None = None) -> str:
    """
    Click (x,y) in CSS pixels **after** verifying an element exists there.
    If *must_contain* is given, its text must appear in/around the element.
    """
    page = await _get_page()

    # locate element under point INSIDE the browser process
    elem = await page.evaluate_handle(
        """([x,y]) => document.elementFromPoint(x, y)""",
        [x, y]
    )
    if not elem:
        raise AssertionError(f"No element at {x},{y}")

    if must_contain:
        text = await elem.evaluate(
            "el => el.innerText || el.getAttribute('aria-label') || ''"
        )
        if must_contain.lower() not in text.lower():
            raise AssertionError(
                f"'{must_contain}' not found in element text '{text.strip()}'"
            )

    await page.mouse.click(x, y)
    return "safe-clicked"


# ───────── classic DOM‑free tools ─────────
async def visit(url: str) -> str:
    """Navigate to a URL and return the page title."""
    page = await _get_page()
    await page.goto(url, wait_until="networkidle")
    return await page.title()


async def vision_click(description: str) -> str:
    """
    Locate *description* visually and click it.
    ① Screenshot viewport ② ask vision model for bbox + optional selector
    ③ Try selector click first; fallback to safe_click(center) with guard.
    """
    page = await _get_page()
    img = await page.screenshot(full_page=False, type="png")

    # Ask for bbox …
    bbox_prompt = textwrap.dedent(f"""
        You see a PNG of a web page. Return ONLY JSON:
        {{ "x": <int>, "y": <int>, "w": <int>, "h": <int> }}
        Coordinates correspond to the UI element described as:
        "{description}"
        Use CSS pixel space relative to top‑left of the screenshot.
    """)
    vision_service = await get_vision_service()
    bbox = await vision_service.ask_vision_json(img, bbox_prompt)

    # … and (optional) selector
    sel_prompt = textwrap.dedent(f"""
        Return a **single CSS or ARIA selector** (no JSON) uniquely locating
        the same element. If impossible, return "NULL".
    """)
    selector = (await vision_service.ask_vision_text(img, sel_prompt)).strip()

    # 1⃣  selector‑based click
    if selector and selector != "NULL":
        try:
            await page.click(selector, timeout=1500)
            return "clicked-via-selector"
        except Exception:
            logger.warning(
                f"Selector click failed, falling back to coordinates. Selector: {selector}")
            pass  # fallback

    # 2⃣  coordinate click with DOM guard
    cx, cy = bbox["x"] + bbox["w"] // 2, bbox["y"] + bbox["h"] // 2
    return await safe_click(cx, cy, must_contain=description.split()[0])


async def vision_fill(description: str, text: str) -> str:
    """Type *text* into the input visually described by *description*."""
    page = await _get_page()
    img = await page.screenshot(full_page=False, type="png")

    vision_service = await get_vision_service()
    bbox_prompt = textwrap.dedent(f"""
        JSON {{ "x": int, "y": int, "w": int, "h": int }} of the input field
        whose label/placeholder matches:
        "{description}"
    """)
    bbox = await vision_service.ask_vision_json(img, bbox_prompt)

    cx, cy = bbox["x"] + bbox["w"] // 2, bbox["y"] + bbox["h"] // 2
    await safe_click(cx, cy)
    await page.keyboard.type(text)
    return "vision-filled"


async def vision_expect_row(expected: dict) -> str:
    """
    Verify (visually) that ANY table row contains ALL key/value pairs in *expected*.
    """
    page = await _get_page()
    img = await page.screenshot(full_page=False, type="png")

    vision_service = await get_vision_service()
    task = textwrap.dedent(f"""
        Does any table row in this screenshot contain every key/value pair
        below?  Answer only YES or NO.

        {json.dumps(expected, ensure_ascii=False)}
    """)
    answer = (await vision_service.ask_vision_text(img, task)).upper()
    if answer.startswith("YES"):
        return "assertion-passed"
    raise AssertionError("Expected row not found")


# Additional utility functions for cleanup
async def close_browser():
    """Close the browser and clean up resources."""
    global _pw, _browser, _page
    if _page:
        await _page.close()
        _page = None
    if _browser:
        await _browser.close()
        _browser = None
    if _pw:
        await _pw.stop()
        _pw = None
