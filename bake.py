# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.48"]
# ///
"""Precompute every example for the static demo (a Hugging Face static Space, no server).

    uv run server.py      # in one terminal
    uv run bake.py        # in another; writes static/

It drives the real page in Chrome, so each answer is stored under the SHA-256 of the exact
request the page sends (first 16 hex digits). Without a server, index.html loads static/<hash>.json.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

URL, OUT = "http://localhost:8765", Path(__file__).parent / "static"
IDLE = "document.querySelector('#busy').className === '' && document.querySelector('#status').textContent.includes('ms')"


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    (OUT / "info.json").write_bytes(urllib.request.urlopen(URL + "/api/info").read())
    saved = {}

    def keep(resp):
        if resp.url.endswith("/api/analyze") and resp.ok:
            body = resp.request.post_data
            key = hashlib.sha256(body.encode()).hexdigest()[:16]
            (OUT / f"{key}.json").write_bytes(resp.body())
            saved[key] = json.loads(body)["text"][:48]

    with sync_playwright() as p:
        page = p.chromium.launch(channel="chrome").new_page()
        page.on("response", keep)
        page.goto(URL)
        page.wait_for_function(IDLE, timeout=120000)
        page.click("#landGo")
        analyzed = lambda: page.expect_response(lambda r: r.url.endswith("/api/analyze"), timeout=120000)
        for i, task in enumerate(b.get_attribute("data-task") for b in page.locator("#taskbar [data-task]").all()):
            if i:  # the first tab is already open and analyzed
                with analyzed():
                    page.click(f"#taskbar [data-task='{task}']")
                page.wait_for_function(IDLE, timeout=120000)
            for n in range(page.locator("#examples button").count()):
                with analyzed():
                    page.locator("#examples button").nth(n).click()
                page.wait_for_function(IDLE, timeout=120000)
    size = sum(f.stat().st_size for f in OUT.glob("*.json")) / 1e6
    print(f"{len(saved)} answers, {size:.1f} MB in {OUT}")
    for key, text in sorted(saved.items(), key=lambda kv: kv[1]):
        print(f"  {key}  {text}")


if __name__ == "__main__":
    main()
