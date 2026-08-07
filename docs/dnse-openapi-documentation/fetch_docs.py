#!/usr/bin/env python3
"""Mirror the DNSE OpenAPI documentation as Markdown.

DNSE's developer docs (https://developers.dnse.com.vn, a Docusaurus site) render
their endpoint schemas client-side, so the raw HTML is useless offline. Each page
does expose a raw-markdown export at ``{page_url}.md`` (the site's "Copy page"
button). This script keeps ``docs/dnse-openapi-documentation/`` in sync:

  1. read the docs sitemap to discover EVERY ``/docs/`` page (so newly published
     endpoints/guides are picked up automatically -- no hardcoded slug list),
  2. fetch each page's ``.md`` export from the docs host,
  3. write it next to this script as ``<flattened-slug>.md``
     (``/docs/guide/versioning/api`` -> ``guide-versioning-api.md``),
  4. report what changed (new / updated / unchanged / skipped).

Re-run any time to refresh the mirror::

    python docs/dnse-openapi-documentation/fetch_docs.py

Notes
-----
* The sitemap lists ``www.dnse.com.vn`` URLs, but the ``.md`` export only renders
  on ``developers.dnse.com.vn`` -- so paths are discovered from the sitemap and
  fetched from ``CONTENT_HOST``.
* Category-landing pages (e.g. ``/docs/dnse/account``) have no ``.md`` export and
  are skipped with a note; their child pages carry the real content.
* stdlib only -- no third-party dependencies, so it runs anywhere.
"""
from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

SITEMAP_URL = "https://developers.dnse.com.vn/sitemap.xml"
CONTENT_HOST = "https://developers.dnse.com.vn"
# Machine-readable OpenAPI spec per published API version (authoritative, and
# CDN-hosted so it can't be masked by rendered-HTML caching).
CDN_SPEC = "https://cdn.entrade.com.vn/dnse-openapi/doc/dnse-openapi-{version}.yaml"
DEST = Path(__file__).resolve().parent
USER_AGENT = "Mozilla/5.0 (dnse-doc-mirror)"
TIMEOUT = 30
PAUSE_SECONDS = 0.2  # be polite between requests


def _get(url: str) -> tuple[int, bytes]:
    """GET ``url``; return ``(status, body)``. Network error -> ``(0, message)``."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""
    except (urllib.error.URLError, TimeoutError) as error:
        return 0, str(error).encode()


def discover_doc_slugs() -> list[str]:
    """Return the sorted, unique doc slugs (path after ``/docs/``) in the sitemap."""
    status, body = _get(SITEMAP_URL)
    if status != 200 or not body:
        sys.exit(f"cannot read sitemap (http {status}): {SITEMAP_URL}")
    root = ElementTree.fromstring(body)
    slugs: set[str] = set()
    for element in root.iter():
        # Namespaced tags look like '{...}loc'; match by suffix.
        if element.tag.endswith("loc") and element.text and "/docs/" in element.text:
            slug = element.text.split("/docs/", 1)[1].strip("/")
            if slug:  # drop the bare "/docs" index
                slugs.add(slug)
    return sorted(slugs)


def looks_like_markdown(text: str) -> bool:
    """A real ``.md`` export is Markdown; a 404/redirect returns HTML or nothing."""
    head = text.lstrip()[:80].lower()
    return bool(text.strip()) and not head.startswith("<") and "<!doctype" not in head


def fetch_specs() -> None:
    """Download the OpenAPI spec YAML for each PUBLISHED API version.

    The published versions are the ``### YYYY-MM-DD`` sub-headings under the
    ``## API Versions`` section of the versioning guide (mirrored this run). This
    is date-based versioning, so a newly published version shows up as a new
    heading -> its spec is picked up automatically. The CDN 404s any date that is
    not an actual published version (the guide's SemVer *examples* have no spec),
    so guessing is self-correcting.
    """
    guide = DEST / "guide-versioning-api.md"
    if not guide.exists():
        print("\n(versioning guide not mirrored; skipping spec download)")
        return
    body = guide.read_text(encoding="utf-8").split("## API Versions", 1)[-1]
    versions = sorted(set(re.findall(r"^###\s+(\d{4}-\d{2}-\d{2})", body, re.M)))
    print(f"\nOpenAPI spec YAMLs for {len(versions)} published version(s): "
          f"{', '.join(versions) or '(none found)'}")
    for version in versions:
        status, spec = _get(CDN_SPEC.format(version=version))
        if status != 200 or not spec:
            print(f"  skip  openapi-spec-{version}.yaml (http {status})")
            continue
        (DEST / f"openapi-spec-{version}.yaml").write_bytes(spec)
        print(f"  spec  openapi-spec-{version}.yaml ({len(spec)}B)")
        time.sleep(PAUSE_SECONDS)


def _pinned_sdk_version(repo_root: Path) -> str | None:
    """The API version our vendored openapi-sdk sends by default, or None."""
    common = repo_root / "plugins/dnse/pynecore_dnse/_vendor/dnse/api/common.py"
    if not common.exists():
        return None
    match = re.search(r'DEFAULT_API_VERSION\s*=\s*"(\d{4}-\d{2}-\d{2})"',
                      common.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def verify_applied() -> None:
    """Verify our repo reflects the latest DNSE changelog / API version.

    Two checks, because they catch DIFFERENT drift:

    * **version** — the vendored openapi-sdk's default API version vs the latest
      PUBLISHED version. Catches *breaking* changes (which bump the date version).
    * **changelog** — the LIVE changelog vs our mirror, plus any entry dated after
      the applied version. Catches *non-breaking* additions: a new endpoint ships a
      changelog entry WITHOUT a version bump, so the version check alone is blind to
      it (e.g. 2026-08-06 "Get Expected Price" while the version stayed 2026-07-23).
    """
    print("\n=== verify: changelog vs applied ===")
    changelog = DEST / "changelog.md"
    entries = re.findall(r"^##\s+\[(\d{4}-\d{2}-\d{2})\]",
                         changelog.read_text(encoding="utf-8"), re.M) if changelog.exists() else []
    latest_changelog = max(entries) if entries else "?"

    guide = DEST / "guide-versioning-api.md"
    published = sorted(re.findall(
        r"^###\s+(\d{4}-\d{2}-\d{2})",
        guide.read_text(encoding="utf-8").split("## API Versions", 1)[-1], re.M)) if guide.exists() else []
    latest_version = published[-1] if published else "?"

    pinned = _pinned_sdk_version(DEST.parents[1]) or "?"

    print(f"  latest changelog entry:   {latest_changelog}")
    print(f"  latest published version: {latest_version}   (all published: {', '.join(published) or '?'})")
    print(f"  vendored SDK applies:     {pinned}")
    if pinned != "?" and pinned == latest_version:
        print("  ✓ version up to date — the vendored SDK sends the latest published API version")
    else:
        print("  ⚠ VERSION DRIFT — DNSE bumped the API version. Re-vendor the SDK:")
        print("    https://developers.dnse.com.vn/docs/changelog")

    # --- non-breaking additions: diff the LIVE changelog against our mirror ---
    print("  --- live changelog vs mirror ---")
    status, live = _get(f"{CONTENT_HOST}/docs/changelog.md")
    if status != 200 or not live:
        print(f"  (could not fetch live changelog: http {status} — skipping diff)")
        return
    live_entries = re.findall(r"^##\s+\[(\d{4}-\d{2}-\d{2})\]",
                              live.decode("utf-8", "replace"), re.M)
    unmirrored = [e for e in live_entries if e not in set(entries)]
    if unmirrored:
        print(f"  ⚠ {len(unmirrored)} live changelog entr(y/ies) NOT mirrored: "
              f"{', '.join(unmirrored)} — re-run to mirror, then review for new endpoints.")
    else:
        print("  ✓ mirror carries every published changelog entry")
    after_pin = sorted(e for e in set(live_entries) if pinned != "?" and e > pinned)
    if after_pin:
        print(f"  ⓘ {len(after_pin)} changelog entr(y/ies) dated AFTER the applied version "
              f"{pinned} — likely non-breaking additions; confirm we consume them: "
              f"{', '.join(after_pin)}")


def main() -> int:
    slugs = discover_doc_slugs()
    print(f"discovered {len(slugs)} /docs/ pages from the sitemap\n")

    new = updated = unchanged = skipped = 0
    for slug in slugs:
        url = f"{CONTENT_HOST}/docs/{slug}.md"
        status, body = _get(url)
        text = body.decode("utf-8", "replace")
        if status != 200 or not looks_like_markdown(text):
            print(f"  skip  {slug:45} (http {status}, no .md export)")
            skipped += 1
            continue

        target = DEST / (slug.replace("/", "-") + ".md")
        previous = target.read_text(encoding="utf-8") if target.exists() else None
        if previous is None:
            tag, new = "NEW", new + 1
        elif previous != text:
            tag, updated = "upd", updated + 1
        else:
            tag, unchanged = "==", unchanged + 1
        target.write_text(text, encoding="utf-8")
        print(f"  {tag:4} {target.name:48} ({len(body):>6}B)")
        time.sleep(PAUSE_SECONDS)

    print(f"\ndone: {new} new, {updated} updated, {unchanged} unchanged, "
          f"{skipped} skipped (landing pages)")
    fetch_specs()
    verify_applied()
    print(f"\nmirror: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
