#!/usr/bin/env python3
"""Walk a directory of legacy documents, extract plain text from every
file it knows how to read, and write one .txt per source document plus
a manifest.jsonl describing what happened (method used, char count,
errors). Archives (zip/rar) are unpacked first and their contents fed
through the same pipeline.

Usage:
    python extract.py <input_dir> <output_dir> [--workers N] [--skip-ocr]

Output layout:
    <output_dir>/text/<relative-path-with-slashes-as-__>.txt
    <output_dir>/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from extractors import EXTRACTORS

ARCHIVE_EXTS = {".zip", ".rar"}
SKIP_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".ds_store"}


def iter_source_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in SKIP_EXTS or p.name == ".DS_Store":
            continue
        yield p


def unpack_archive(path: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest)
        except zipfile.BadZipFile:
            return []
    elif path.suffix.lower() == ".rar":
        if shutil.which("7z") is None:
            raise RuntimeError("7z not found (install p7zip) for .rar extraction")
        subprocess.run(["7z", "x", f"-o{dest}", str(path)], capture_output=True, check=False)
    return [p for p in dest.rglob("*") if p.is_file()]


def safe_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return str(rel).replace("/", "__")


def process_one(root: Path, path: Path, out_dir: Path, skip_ocr: bool, ocr_lang: str) -> dict:
    rec = {"source": str(path.relative_to(root)), "status": "skipped"}
    ext = path.suffix.lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        rec["status"] = "unsupported_format"
        return rec

    try:
        kwargs = {}
        if ext in (".pdf", ".djvu"):
            kwargs["ocr_lang"] = ocr_lang
            if skip_ocr:
                kwargs["ocr"] = False
        text, method = extractor(path, **kwargs)
        text = text.strip()
        if method.endswith("-no-text-layer"):
            rec["status"] = "skipped_ocr"
            return rec
        if not text:
            rec["status"] = "empty"
            return rec

        out_name = safe_name(root, path) + ".txt"
        out_path = out_dir / "text" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        rec.update(status="ok", method=method, chars=len(text), output=str(out_path))
    except Exception as e:  # noqa: BLE001 -- one bad file shouldn't kill the batch
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc(limit=3)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-ocr", action="store_true", help="skip pages/files with no text layer instead of OCR'ing them")
    ap.add_argument("--ocr-lang", default="eng", help="tesseract language code(s), e.g. 'eng', 'rus+eng' (needs the matching traineddata installed)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"

    with tempfile.TemporaryDirectory(prefix="legacy-docs-unpack-") as unpack_tmp:
        unpack_root = Path(unpack_tmp)
        all_files: list[tuple[Path, Path]] = []  # (root_for_relpath, file_path)

        for f in iter_source_files(args.input_dir):
            if f.suffix.lower() in ARCHIVE_EXTS:
                dest = unpack_root / f.stem
                try:
                    extracted = unpack_archive(f, dest)
                except Exception as e:  # noqa: BLE001
                    print(f"[archive-error] {f}: {e}", file=sys.stderr)
                    continue
                for ef in extracted:
                    if ef.suffix.lower() not in SKIP_EXTS:
                        all_files.append((dest, ef))
            else:
                all_files.append((args.input_dir, f))

        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(process_one, root, f, args.output_dir, args.skip_ocr, args.ocr_lang): f
                for root, f in all_files
            }
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
                print(f"[{rec['status']}] {rec['source']}")

    with manifest_path.open("w", encoding="utf-8") as mf:
        for rec in results:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} extracted. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
