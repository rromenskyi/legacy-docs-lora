"""Split extracted text into chunks sized for one QA-generation call.

Splits on paragraph boundaries (blank lines), merging small paragraphs
together and splitting oversized ones on sentence boundaries as a last
resort, so a chunk is never cut mid-thought if avoidable.
"""

from __future__ import annotations

import re

PARA_SPLIT = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def chunk_text(text: str, target_chars: int = 2000, max_chars: int = 3500, drop_low_value: bool = True) -> list[str]:
    paragraphs = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    if drop_low_value:
        # Filter BEFORE merging, not after: a short table-of-contents
        # paragraph sitting next to real prose would otherwise get
        # merged into the same chunk, and the merged chunk's TOC-heavy
        # line ratio would then drop the prose along with the TOC.
        paragraphs = [p for p in paragraphs if not is_low_value(p)]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            sentences = SENTENCE_SPLIT.split(para)
            buf = []
            buf_len = 0
            for s in sentences:
                if buf_len + len(s) > target_chars and buf:
                    chunks.append(" ".join(buf))
                    buf, buf_len = [], 0
                buf.append(s)
                buf_len += len(s)
            if buf:
                chunks.append(" ".join(buf))
            continue

        if current_len + len(para) > target_chars and current:
            flush()
        current.append(para)
        current_len += len(para)

    flush()
    return [c for c in chunks if len(c) > 100]  # drop scraps (page numbers, headers, OCR noise)


# --- quality filtering: skip chunks not worth an LLM call ------------------

LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)  # a "letter" in any script
DOTTED_LEADER_RE = re.compile(r"^.{2,80}[.…]{3,}\s*\d{1,4}\s*$")  # "Chapter 3 ..... 42"


def is_low_value(text: str, min_letter_ratio: float = 0.55, max_dotted_leader_ratio: float = 0.3) -> bool:
    """True for chunks unlikely to be worth generating QA pairs from:
    tables of contents (dotted leaders), indices, title pages, raw
    digit/number tables -- not prose. These cost a wasted LLM call and,
    worse, tend to produce QA pairs that look valid (well-formed JSON)
    but are hallucinated, since there's no real content to ground on."""
    stripped = text.strip()
    if len(stripped) < 100:
        return True

    non_space = [c for c in stripped if not c.isspace()]
    if not non_space:
        return True
    letter_ratio = sum(1 for c in non_space if LETTER_RE.match(c)) / len(non_space)
    if letter_ratio < min_letter_ratio:
        return True

    lines = [l for l in stripped.splitlines() if l.strip()]
    if lines:
        dotted_ratio = sum(1 for l in lines if DOTTED_LEADER_RE.match(l.strip())) / len(lines)
        if dotted_ratio > max_dotted_leader_ratio:
            return True

    return False


# --- corpus-wide near-duplicate filtering -----------------------------------
#
# The same passage (boilerplate front matter, a commonly-reprinted maxim,
# two editions of the same book) can show up near-verbatim in more than
# one source document. Deduping generated QA text after the fact (see
# generate.py's dedup()) doesn't catch this: two near-identical source
# chunks usually produce differently-WORDED but conceptually duplicate
# pairs. Catch it at the chunk level instead, across the whole corpus,
# before spending an LLM call on it.

def _shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + k]) for i in range(max(0, len(words) - k + 1))}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def dedup_chunks(chunks: list[tuple[str, int, str]], threshold: float = 0.8) -> list[tuple[str, int, str]]:
    """chunks: list of (source, chunk_index, text). Drops near-duplicates
    (by 5-word-shingle Jaccard similarity), keeping the first occurrence.
    O(n^2) in chunk count -- fine at the hundreds-of-books/thousands-of-
    chunks scale this pipeline targets; would need a real near-dup index
    (e.g. MinHash/LSH) well before that stops being true."""
    kept: list[tuple[str, int, str]] = []
    kept_shingles: list[set[str]] = []
    for src, idx, text in chunks:
        sig = _shingles(text)
        if any(_jaccard(sig, other) >= threshold for other in kept_shingles):
            continue
        kept.append((src, idx, text))
        kept_shingles.append(sig)
    return kept
