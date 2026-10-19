"""Split extracted text into chunks sized for one QA-generation call.

Splits on paragraph boundaries (blank lines), merging small paragraphs
together and splitting oversized ones on sentence boundaries as a last
resort, so a chunk is never cut mid-thought if avoidable.
"""

from __future__ import annotations

import re

PARA_SPLIT = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def chunk_text(text: str, target_chars: int = 2000, max_chars: int = 3500) -> list[str]:
    paragraphs = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]

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
