"""Per-format text extraction. Each extractor takes a file path and
returns (text, method) where method says how the text was obtained
(e.g. "pdf-text", "pdf-ocr", "djvu-text", "djvu-ocr") so the caller can
flag low-confidence extractions (OCR) for later review.

External tools required on PATH, by format:
  - djvu:        djvutxt, ddjvu   (djvulibre)
  - doc/docx/rtf: libreoffice (headless)
  - ocr fallback: tesseract (via pytesseract), needs the target
                   language's traineddata installed (e.g. rus+eng)
  - chm:          7z (p7zip)

Nothing here reads a whole corpus into memory at once; each function
handles one file.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

MIN_CHARS_PER_PAGE_OK = 200  # below this, assume no real text layer -> OCR


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


def _ocr_image(image_path: Path, lang: str = "eng") -> str:
    import pytesseract
    from PIL import Image

    with Image.open(image_path) as img:
        return pytesseract.image_to_string(img, lang=lang)


def extract_pdf(path: Path, ocr_lang: str = "eng", ocr: bool = True) -> tuple[str, str]:
    import fitz  # pymupdf

    doc = fitz.open(path)
    try:
        texts = [page.get_text() for page in doc]
        text = "\n\n".join(texts)
        if doc.page_count and len(text) / doc.page_count >= MIN_CHARS_PER_PAGE_OK:
            return text, "pdf-text"
        if not ocr:
            return text, "pdf-no-text-layer"

        # No usable text layer -> render each page and OCR it.
        ocr_chunks = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img_path = Path(tmp) / f"page-{i}.png"
                pix.save(img_path)
                ocr_chunks.append(_ocr_image(img_path, lang=ocr_lang))
        return "\n\n".join(ocr_chunks), "pdf-ocr"
    finally:
        doc.close()


def extract_djvu(path: Path, ocr_lang: str = "eng", ocr: bool = True) -> tuple[str, str]:
    if shutil.which("djvutxt") is None:
        raise RuntimeError("djvutxt not found (install djvulibre)")

    res = _run(["djvutxt", str(path)])
    text = res.stdout
    if len(text.strip()) >= MIN_CHARS_PER_PAGE_OK:
        return text, "djvu-text"
    if not ocr:
        return text, "djvu-no-text-layer"

    if shutil.which("ddjvu") is None:
        raise RuntimeError("ddjvu not found (install djvulibre) for OCR fallback")

    page_count_res = _run(["djvused", "-e", "n", str(path)])
    try:
        n_pages = int(page_count_res.stdout.strip())
    except ValueError:
        n_pages = 1

    ocr_chunks = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, n_pages + 1):
            img_path = Path(tmp) / f"page-{i}.tiff"
            res = _run(["ddjvu", "-format=tiff", f"-page={i}", "-scale=300", str(path), str(img_path)])
            if not img_path.exists():
                print(f"[djvu-ocr] page {i}/{n_pages} render failed: {res.stderr.strip()}")
                continue
            ocr_chunks.append(_ocr_image(img_path, lang=ocr_lang))
    return "\n\n".join(ocr_chunks), "djvu-ocr"


def extract_office(path: Path) -> tuple[str, str]:
    """doc/docx/rtf/odt -> plain text. Tries pandoc first (handles
    docx/rtf/odt well, lighter dependency); falls back to headless
    LibreOffice, which is the only one of the two that reliably reads
    legacy binary .doc (Word 97-2003)."""
    if shutil.which("pandoc") is not None:
        res = _run(["pandoc", str(path), "-t", "plain"])
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout, "office-pandoc"

    binary = shutil.which("libreoffice") or shutil.which("soffice")
    if binary is None:
        raise RuntimeError("neither pandoc nor libreoffice/soffice available")

    with tempfile.TemporaryDirectory() as tmp:
        _run([binary, "--headless", "--convert-to", "txt:Text", "--outdir", tmp, str(path)])
        out = Path(tmp) / (path.stem + ".txt")
        if not out.exists():
            raise RuntimeError(f"libreoffice produced no output for {path}")
        return out.read_text(encoding="utf-8", errors="ignore"), "office-libreoffice"


def extract_txt(path: Path) -> tuple[str, str]:
    for enc in ("utf-8", "cp1251", "cp866", "latin-1"):
        try:
            return path.read_text(encoding=enc), f"txt-{enc}"
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore"), "txt-fallback"


def extract_fb2(path: Path) -> tuple[str, str]:
    data = path.read_text(encoding="utf-8", errors="ignore")
    # fb2 is XML but often has a bogus/missing encoding declaration; strip it.
    data = re.sub(r"^<\?xml[^>]*\?>", "<?xml version=\"1.0\"?>", data, count=1)
    root = ET.fromstring(data)
    chunks = [el.text for el in root.iter() if el.text and el.text.strip()]
    return "\n".join(chunks), "fb2-xml"


def extract_chm(path: Path) -> tuple[str, str]:
    if shutil.which("7z") is None:
        raise RuntimeError("7z not found (install p7zip) for .chm extraction")
    from bs4 import BeautifulSoup

    with tempfile.TemporaryDirectory() as tmp:
        _run(["7z", "x", f"-o{tmp}", str(path)])
        chunks = []
        for html_file in Path(tmp).rglob("*.htm*"):
            soup = BeautifulSoup(html_file.read_bytes(), "html.parser")
            chunks.append(soup.get_text("\n"))
        return "\n\n".join(chunks), "chm-html"


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".djvu": extract_djvu,
    ".doc": extract_office,
    ".docx": extract_office,
    ".rtf": extract_office,
    ".odt": extract_office,
    ".txt": extract_txt,
    ".fb2": extract_fb2,
    ".chm": extract_chm,
}
