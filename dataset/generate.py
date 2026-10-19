#!/usr/bin/env python3
"""Generate a synthetic instruction/QA dataset from extracted text, for
LoRA fine-tuning. Reads the .txt output of ../ingest/extract.py, chunks
each document, and asks an LLM (any OpenAI-compatible endpoint -- a
paid API or a local server like LLMTray/Ollama/vLLM) to produce several
instruction/answer pairs grounded strictly in each chunk.

Usage:
    python generate.py <text_dir> <output.jsonl> \
        --base-url http://localhost:8765/v1 --model some-local-model \
        --languages ru,en --pairs-per-chunk 4 --workers 8

Auth: set OPENAI_API_KEY if the endpoint needs one (local servers
usually don't).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from chunker import chunk_text

SYSTEM_PROMPT = """You write training examples for fine-tuning a language model on a \
specific domain. Given a passage of source text, produce instruction/answer pairs that \
teach the model to explain, reason about, and apply the ideas in that passage -- not \
just recite facts. Favor "explain why X works", "what's the principle behind Y", \
"how would you apply this to situation Z" over simple fact lookups. Every pair must be \
fully grounded in the given passage: don't add outside knowledge, don't invent details \
the passage doesn't support. If the passage is too thin or degraded (OCR noise, a table \
of contents, a title page) to support good pairs, return an empty list -- don't force it."""

USER_TEMPLATE = """Passage:
---
{chunk}
---

Write {n} instruction/answer pairs grounded in this passage, in {lang_desc}. Respond with \
ONLY a JSON array, no other text, each item: {{"language": "...", "instruction": "...", "response": "..."}}"""

LANG_NAMES = {"ru": "Russian", "en": "English", "uk": "Ukrainian", "de": "German"}


def lang_description(languages: list[str]) -> str:
    names = [LANG_NAMES.get(code, code) for code in languages]
    if len(names) == 1:
        return names[0]
    return "a mix of " + " and ".join(names) + " (write different pairs in different languages, matching the source material's language where it fits naturally)"


def extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    # Strip markdown code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("instruction") and d.get("response")]


def call_llm(base_url: str, api_key: str | None, model: str, chunk: str, n: int, languages: list[str], timeout: int) -> list[dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(chunk=chunk, n=n, lang_desc=lang_description(languages))},
        ],
        "temperature": 0.7,
    }
    resp = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return extract_json_array(content)


def dedup(pairs: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for p in pairs:
        key = p["instruction"].strip().lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--base-url", default="http://localhost:8765/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--languages", default="en", help="comma-separated, e.g. ru,en")
    ap.add_argument("--pairs-per-chunk", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--limit-chunks", type=int, default=None, help="process at most N chunks total (for a cheap/quick test run)")
    args = ap.parse_args()

    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    api_key = os.environ.get("OPENAI_API_KEY")

    jobs = []  # (source, chunk_idx, chunk_text)
    for txt_file in sorted(args.text_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(text)):
            jobs.append((txt_file.name, i, chunk))
    if args.limit_chunks:
        jobs = jobs[: args.limit_chunks]

    print(f"{len(jobs)} chunks queued from {len(set(j[0] for j in jobs))} documents")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(call_llm, args.base_url, api_key, args.model, chunk, args.pairs_per_chunk, languages, args.timeout): (src, idx)
            for src, idx, chunk in jobs
        }
        for fut in as_completed(futs):
            src, idx = futs[fut]
            try:
                pairs = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[error] {src}#{idx}: {type(e).__name__}: {e}")
                continue
            print(f"[ok] {src}#{idx}: {len(pairs)} pairs")
            for p in pairs:
                p["_source"] = src
                p["_chunk"] = idx
                results.append(p)

    results = dedup(results)

    with args.output.open("w", encoding="utf-8") as f:
        for p in results:
            messages = [
                {"role": "user", "content": p["instruction"]},
                {"role": "assistant", "content": p["response"]},
            ]
            f.write(json.dumps({"messages": messages, "meta": {"source": p["_source"], "chunk": p["_chunk"], "language": p.get("language")}}, ensure_ascii=False) + "\n")

    print(f"\n{len(results)} pairs written to {args.output}")


if __name__ == "__main__":
    main()
