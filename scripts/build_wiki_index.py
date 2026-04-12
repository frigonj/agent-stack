"""
scripts/build_wiki_index.py
───────────────────────────
Pre-build two artefacts from the compressed Wikipedia multistream index:

  1. /workspace/wiki/wiki_titles.bloom
     Bloom filter (bytearray + mmh3, ~26 MB) for O(1) membership checks.
     Used by agents to skip wiki lookups for queries that definitely aren't
     in the dump (zero false negatives, ~0.1% false positives).

  2. /workspace/wiki/wiki_manifest.json
     Category manifest — top-level coverage summary for agent metacognition.
     Tells agents *what the wiki knows* without loading the full index.

Run once after placing the dump files in /workspace/wiki/:

    python scripts/build_wiki_index.py

Both output files are checked into /workspace/wiki/ and loaded lazily by
core/wiki.py on first use.
"""

from __future__ import annotations

import bz2
import json
import math
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import mmh3

# ── Config ────────────────────────────────────────────────────────────────────

WIKI_DIR = Path(os.getenv("WIKI_DIR", "/workspace/wiki"))
BLOOM_PATH = WIKI_DIR / "wiki_titles.bloom"
MANIFEST_PATH = WIKI_DIR / "wiki_manifest.json"

# Bloom filter parameters
# 22M articles, 0.1% false positive rate → ~26 MB, 10 hash functions
BLOOM_N = 22_000_000          # expected number of items
BLOOM_FP = 0.001              # target false-positive rate
BLOOM_K = 10                  # number of hash functions

# Category manifest: top N categories by article count
MANIFEST_TOP_N = 300


# ── Bloom filter (pure Python, no dependencies beyond mmh3) ──────────────────

def _bloom_size(n: int, fp: float) -> int:
    """Optimal bit count for n items at fp false-positive rate."""
    return int(-n * math.log(fp) / (math.log(2) ** 2))


class BloomFilter:
    """
    Minimal bloom filter backed by a bytearray.
    Serialisation format (binary):
      - 8 bytes: magic b'WIKIBLM\x01'
      - 4 bytes: k (uint32 LE)
      - 8 bytes: m (uint64 LE, bit count)
      - m/8 bytes: raw bits (rounded up to byte boundary)
    """

    MAGIC = b"WIKIBLM\x01"

    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.bits = bytearray((m + 7) // 8)

    def _hashes(self, item: str) -> list[int]:
        h1 = mmh3.hash(item, 0, signed=False)
        h2 = mmh3.hash(item, 1, signed=False)
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: str) -> None:
        for pos in self._hashes(item):
            self.bits[pos >> 3] |= 1 << (pos & 7)

    def __contains__(self, item: str) -> bool:
        return all(self.bits[pos >> 3] & (1 << (pos & 7)) for pos in self._hashes(item))

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            fh.write(self.MAGIC)
            fh.write(struct.pack("<I", self.k))
            fh.write(struct.pack("<Q", self.m))
            fh.write(self.bits)

    @classmethod
    def load(cls, path: Path) -> "BloomFilter":
        with open(path, "rb") as fh:
            magic = fh.read(8)
            if magic != cls.MAGIC:
                raise ValueError(f"Bad magic: {magic!r}")
            k = struct.unpack("<I", fh.read(4))[0]
            m = struct.unpack("<Q", fh.read(8))[0]
            bits = bytearray(fh.read())
        bf = cls.__new__(cls)
        bf.m = m
        bf.k = k
        bf.bits = bits
        return bf


# ── Category extraction ───────────────────────────────────────────────────────

# Wikipedia top-level category prefixes we care about
# Derived from https://en.wikipedia.org/wiki/Wikipedia:Contents/Categories
_KNOWN_DOMAINS = [
    "Agriculture", "Arts", "Astronomy", "Biology", "Business", "Chemistry",
    "Computing", "Culture", "Economics", "Education", "Engineering",
    "Entertainment", "Environment", "Events", "Film", "Food", "Geography",
    "Government", "Health", "History", "Humanities", "Language", "Law",
    "Literature", "Mathematics", "Medicine", "Military", "Music", "Nature",
    "People", "Philosophy", "Physics", "Politics", "Psychology", "Religion",
    "Science", "Society", "Software", "Sports", "Technology", "Television",
    "Transport", "War",
]

def _classify_title(title: str) -> str | None:
    """
    Heuristic: assign a title to a domain based on namespace prefixes,
    parenthetical disambiguation, or keyword presence.
    Returns None for meta-pages (Wikipedia:, Talk:, etc.)
    """
    # Skip meta namespaces
    for ns in ("Wikipedia:", "Talk:", "User:", "File:", "Template:", "Help:",
               "Category:", "Portal:", "Draft:", "Module:"):
        if title.startswith(ns):
            return None

    tl = title.lower()
    for domain in _KNOWN_DOMAINS:
        if domain.lower() in tl:
            return domain
    return "General"


# ── Main ──────────────────────────────────────────────────────────────────────

def _find_index() -> Path:
    for p in WIKI_DIR.iterdir():
        if "pages-articles-multistream-index" in p.name and p.name.endswith(".bz2"):
            return p
    raise FileNotFoundError(f"No index file found in {WIKI_DIR}")


def build(dry_run: bool = False) -> None:
    index_path = _find_index()
    print(f"Index: {index_path} ({index_path.stat().st_size / 1e6:.0f} MB compressed)")

    m = _bloom_size(BLOOM_N, BLOOM_FP)
    print(f"Bloom: m={m:,} bits ({m // 8 / 1e6:.1f} MB), k={BLOOM_K}")

    bf = BloomFilter(m=m, k=BLOOM_K)
    domain_counts: dict[str, int] = defaultdict(int)
    total = 0
    skipped = 0
    t0 = time.time()

    print("Reading index…")
    with bz2.open(index_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            title = parts[2]

            domain = _classify_title(title)
            if domain is None:
                skipped += 1
                continue

            # Add both the exact title and a lower-cased variant
            bf.add(title)
            bf.add(title.lower())

            domain_counts[domain] += 1
            total += 1

            if total % 1_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  {total:>10,} articles  ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {total:,} articles in {elapsed:.1f}s (skipped {skipped:,} meta pages)")

    if not dry_run:
        print(f"Saving bloom filter → {BLOOM_PATH}")
        bf.save(BLOOM_PATH)
        print(f"  Size: {BLOOM_PATH.stat().st_size / 1e6:.1f} MB")

    # ── Category manifest ─────────────────────────────────────────────────────

    # Sort domains by count, build manifest
    sorted_domains = sorted(domain_counts.items(), key=lambda x: -x[1])
    top_domains = sorted_domains[:MANIFEST_TOP_N]

    # Compute coverage stats
    article_count = total
    meta_count = skipped

    manifest = {
        "version": 1,
        "article_count": article_count,
        "meta_page_count": meta_count,
        "dump_date": index_path.name.split("-")[1] if "-" in index_path.name else "unknown",
        "bloom_filter": str(BLOOM_PATH.name),
        "bloom_fp_rate": BLOOM_FP,
        "coverage_note": (
            "Offline Wikipedia dump. Covers encyclopaedic knowledge through dump date. "
            "Does NOT cover: very recent events, local/regional topics with few editors, "
            "primary sources, opinion pieces, or proprietary information."
        ),
        "strong_coverage": [d for d, _ in top_domains[:30]],
        "domain_counts": dict(top_domains),
    }

    if not dry_run:
        print(f"Saving manifest → {MANIFEST_PATH}")
        with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"  Domains: {len(domain_counts)}")
        print(f"  Top 5: {top_domains[:5]}")

    print("Build complete.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    build(dry_run=dry_run)
