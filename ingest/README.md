# ingest

Extracts plain text from a directory of mixed legacy documents.

## Setup

```bash
pip install -r requirements.txt
```

System tools (by format you actually have):

| format | tool | macOS (brew) | Debian/Ubuntu (apt) |
|---|---|---|---|
| djvu | djvulibre (`djvutxt`, `ddjvu`, `djvused`) | `brew install djvulibre` | `apt install djvulibre-bin` |
| doc/docx/rtf/odt | pandoc (preferred) | `brew install pandoc` | `apt install pandoc` |
| legacy .doc fallback | libreoffice | `brew install --cask libreoffice` | `apt install libreoffice` |
| zip/rar, chm | 7z | `brew install p7zip` | `apt install p7zip-full` |
| scanned pdf/djvu | tesseract + language data | `brew install tesseract tesseract-lang` | `apt install tesseract-ocr tesseract-ocr-<lang>` |

You only need the tools for the formats actually present in your input
directory; a missing tool makes that format's files fail individually
(see `manifest.jsonl`), it doesn't stop the batch.

## Usage

```bash
python extract.py <input_dir> <output_dir> --workers 4 --ocr-lang eng
```

- `--ocr-lang` — tesseract language code(s) for scanned PDFs/DJVUs with
  no text layer, e.g. `rus+eng`. Needs the matching `tesseract-lang`
  data installed. Defaults to `eng`.
- `--skip-ocr` — don't OCR scans at all, just skip them (OCR is the
  slow part: full-book OCR runs minutes per book on CPU).

Output: `<output_dir>/text/*.txt` (one per source document) and
`<output_dir>/manifest.jsonl` (per-file status: `ok`, `empty`,
`skipped_ocr`, `unsupported_format`, or `error` with the exception).
