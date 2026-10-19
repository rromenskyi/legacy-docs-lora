# legacy-docs-lora

Turn a pile of legacy documents — PDF, DJVU, DOC/DOCX, RTF, scans,
whatever — into a LoRA fine-tuning dataset. No RAG, no retrieval index:
the goal is a model that has absorbed a domain's terminology, style and
reasoning patterns, not a lookup tool over a fixed corpus.

This repo holds the pipeline only. It never contains source documents,
extracted text, or generated datasets — those stay local (see
`.gitignore`). Tested against a personal archive of several hundred
dog-training/genetics books as a stand-in for "any pile of old junk
documents"; nothing dog-specific is hardcoded.

## Pipeline

1. **Ingest** — unpack archives, route each file by format (PDF with a
   text layer vs scanned, DJVU, DOC/DOCX/RTF, ...), normalize everything
   to plain UTF-8 text with source metadata preserved. Drop anything
   that's images-only with no OCR value (e.g. photographed forms).
2. **Dataset generation** — chunk each document, generate synthetic
   instruction/QA pairs per chunk with a cheap LLM, grounded strictly in
   that chunk's content. Mix in a slice of pairs in other languages than
   the source material to avoid the model learning "topic X -> always
   answer in language Y" instead of mirroring the user's language.
3. **Train** — LoRA fine-tune a base model on the generated dataset.

## Status

Planning stage — no code yet.
