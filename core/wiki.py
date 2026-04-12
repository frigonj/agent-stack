"""
core/wiki.py
────────────
Offline Wikipedia lookup using the multistream XML dump + index.

The dump lives at /workspace/wiki/ inside the container (mounted from
./workspace/wiki on the host).  Two files are required:

  enwiki-*-pages-articles-multistream.xml.bz2        (the dump)
  enwiki-*-pages-articles-multistream-index.txt.bz2  (byte-offset index)

Index format (one line per article):
  <byte_offset>:<page_id>:<article_title>

Usage
-----
  from core.wiki import wiki_lookup

  result = await wiki_lookup("Python (programming language)")
  # Returns WikiResult(title, summary, sections, url) or None if not found.

The index is loaded once at first call and held in a module-level dict.
Subsequent lookups are O(log N) via bisect on the sorted offset list.
"""

from __future__ import annotations

import asyncio
import bz2
import os
import re
import xml.etree.ElementTree as ET
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────

WIKI_DIR = Path(os.getenv("WIKI_DIR", "/workspace/wiki"))

# Max characters returned per article (keeps context manageable for the LLM)
WIKI_MAX_CHARS = int(os.getenv("WIKI_MAX_CHARS", "8000"))

# ── Index singleton ───────────────────────────────────────────────────────────

@dataclass
class _IndexEntry:
    offset: int       # byte offset into the bz2 dump
    page_id: int
    title: str


_index: list[_IndexEntry] = []          # sorted by title (lower-case) after load
_title_map: dict[str, _IndexEntry] = {} # lower-case title → entry
_offset_sorted: list[int] = []          # offsets in ascending order (for block lookup)
_offset_to_entry: dict[int, _IndexEntry] = {}
_index_loaded = False
_index_lock = asyncio.Lock()


def _find_dump_and_index() -> tuple[Path, Path]:
    """Locate the dump and index files regardless of exact date stamp."""
    dump = index = None
    for p in WIKI_DIR.iterdir():
        name = p.name
        if "pages-articles-multistream" in name and name.endswith(".xml.bz2") and "index" not in name:
            dump = p
        elif "pages-articles-multistream-index" in name and name.endswith(".bz2"):
            index = p
    if not dump:
        raise FileNotFoundError(f"No Wikipedia dump found in {WIKI_DIR}")
    if not index:
        raise FileNotFoundError(f"No Wikipedia index found in {WIKI_DIR}")
    return dump, index


def _load_index_sync() -> None:
    global _index, _title_map, _offset_sorted, _offset_to_entry, _index_loaded

    _, index_path = _find_dump_and_index()
    log.info("wiki.index_loading", path=str(index_path))

    entries: list[_IndexEntry] = []
    with bz2.open(index_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            # format: offset:page_id:title  (title may contain colons)
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            try:
                offset = int(parts[0])
                page_id = int(parts[1])
                title = parts[2]
            except ValueError:
                continue
            entries.append(_IndexEntry(offset=offset, page_id=page_id, title=title))

    # Build lookup structures
    for e in entries:
        _title_map[e.title.lower()] = e
        _offset_to_entry[e.offset] = e

    # Sorted unique offsets for bisect-based block range lookup
    _offset_sorted = sorted({e.offset for e in entries})
    _index = entries
    _index_loaded = True
    log.info("wiki.index_loaded", articles=len(entries), blocks=len(_offset_sorted))


async def _ensure_index() -> None:
    global _index_loaded
    if _index_loaded:
        return
    async with _index_lock:
        if _index_loaded:
            return
        await asyncio.get_event_loop().run_in_executor(None, _load_index_sync)


# ── Article extraction ────────────────────────────────────────────────────────

_NS = {"mw": "http://www.mediawiki.org/xml/DTD/mediawiki"}

# Strip wiki markup to readable plain text
_WIKI_STRIP = [
    (re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]"), r"\1"),  # [[link|text]] → text
    (re.compile(r"\{\{[^}]*\}\}", re.S), ""),                  # templates
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S | re.I), ""),    # refs
    (re.compile(r"<[^>]+>"), ""),                              # HTML tags
    (re.compile(r"'{2,}"), ""),                                # bold/italic
    (re.compile(r"==+\s*(.*?)\s*==+"), r"\n\1\n"),            # headings
    (re.compile(r"\[\[File:[^\]]+\]\]", re.I), ""),           # file embeds
    (re.compile(r"\[\[Image:[^\]]+\]\]", re.I), ""),          # image embeds
    (re.compile(r"\[https?://\S+(?:\s+[^\]]+)?\]"), ""),      # external links
    (re.compile(r"\n{3,}"), "\n\n"),                          # blank lines
]


def _strip_wikitext(text: str) -> str:
    for pattern, repl in _WIKI_STRIP:
        text = pattern.sub(repl, text)
    return text.strip()


def _extract_article_from_block(xml_block: bytes, target_title: str) -> Optional[str]:
    """
    Parse a decompressed XML block (containing up to ~100 articles) and return
    the wikitext of the article matching target_title.
    """
    # The block is a fragment — wrap it so ET can parse it
    try:
        root = ET.fromstring(b"<root>" + xml_block + b"</root>")
    except ET.ParseError:
        # Try stripping BOM / partial header
        try:
            clean = re.sub(rb"<\?xml[^?]*\?>", b"", xml_block)
            root = ET.fromstring(b"<root>" + clean + b"</root>")
        except ET.ParseError:
            return None

    target_lower = target_title.lower()
    for page in root.iter("page"):
        title_el = page.find("title")
        if title_el is None or title_el.text is None:
            continue
        if title_el.text.lower() != target_lower:
            continue
        # Found the page — extract wikitext
        text_el = page.find(".//revision/text")
        if text_el is not None and text_el.text:
            return text_el.text
    return None


def _read_block(dump_path: Path, block_offset: int, next_offset: Optional[int]) -> bytes:
    """
    Read and decompress one bz2 stream block from the multistream dump.
    Each block is an independent bz2 stream — we read exactly the bytes
    between block_offset and next_offset (or EOF).
    """
    with open(dump_path, "rb") as fh:
        fh.seek(block_offset)
        if next_offset is not None:
            raw = fh.read(next_offset - block_offset)
        else:
            raw = fh.read()
    return bz2.decompress(raw)


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class WikiResult:
    title: str
    text: str                          # clean plain text, truncated to WIKI_MAX_CHARS
    url: str
    found: bool = True
    redirect_followed: Optional[str] = None   # original title if this was a redirect


async def wiki_lookup(title: str, *, follow_redirects: bool = True) -> Optional[WikiResult]:
    """
    Look up a Wikipedia article by title and return cleaned plain text.

    Returns None if the article is not in the dump.
    Follows #REDIRECT automatically (one hop).
    """
    await _ensure_index()

    title_lower = title.lower()
    entry = _title_map.get(title_lower)
    if entry is None:
        # Try title-cased variant
        entry = _title_map.get(title.title().lower()) or _title_map.get(title.capitalize().lower())
    if entry is None:
        log.debug("wiki.not_found", title=title)
        return None

    dump_path, _ = _find_dump_and_index()

    # Find the next block offset so we read exactly one block
    idx = bisect_right(_offset_sorted, entry.offset) - 1
    block_offset = _offset_sorted[idx]
    next_offset = _offset_sorted[idx + 1] if idx + 1 < len(_offset_sorted) else None

    try:
        xml_block = await asyncio.get_event_loop().run_in_executor(
            None, _read_block, dump_path, block_offset, next_offset
        )
    except Exception as exc:
        log.warning("wiki.block_read_error", title=title, error=str(exc))
        return None

    wikitext = _extract_article_from_block(xml_block, entry.title)
    if wikitext is None:
        log.debug("wiki.article_not_in_block", title=title)
        return None

    # Follow redirect (one hop)
    redirect_match = re.match(r"#REDIRECT\s*\[\[([^\]]+)\]\]", wikitext, re.I)
    if redirect_match and follow_redirects:
        redirect_target = redirect_match.group(1).split("|")[0].strip()
        log.debug("wiki.redirect", from_title=title, to=redirect_target)
        result = await wiki_lookup(redirect_target, follow_redirects=False)
        if result:
            result.redirect_followed = title
        return result

    clean = _strip_wikitext(wikitext)
    return WikiResult(
        title=entry.title,
        text=clean[:WIKI_MAX_CHARS],
        url=f"https://en.wikipedia.org/wiki/{entry.title.replace(' ', '_')}",
    )


async def wiki_search_titles(query: str, max_results: int = 5) -> list[str]:
    """
    Fuzzy title search: return article titles that contain all words in the query.
    Fast string scan over the title map — no network call.
    """
    await _ensure_index()
    words = query.lower().split()
    results = []
    for title_lower, entry in _title_map.items():
        if all(w in title_lower for w in words):
            results.append(entry.title)
            if len(results) >= max_results:
                break
    return results
