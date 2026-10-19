# dataset

Turns the plain-text output of `../ingest/` into a synthetic
instruction/QA dataset for LoRA fine-tuning. Chunks each document,
asks an LLM to write instruction/answer pairs grounded in each chunk
(favoring "explain the principle" over fact lookup -- see the prompt in
`generate.py`), and writes a chat-format JSONL.

Works against any OpenAI-compatible `/v1/chat/completions` endpoint --
a paid API (set `OPENAI_API_KEY`) or a local server such as LLMTray,
Ollama, or vLLM (no key needed).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# cheap/local test run, 2 chunks, against a local LLMTray model
python generate.py ../ingest-output/text test.jsonl \
  --base-url http://localhost:8765/v1 --model <local-model-id> \
  --languages ru,en --limit-chunks 2 --workers 1

# real run against a paid API
export OPENAI_API_KEY=...
python generate.py ../ingest-output/text dataset.jsonl \
  --base-url https://api.openai.com/v1 --model gpt-5-mini \
  --languages ru,en --pairs-per-chunk 4 --workers 8
```

`--languages` matters beyond just coverage: generating pairs in more
than one language on the *same* source material stops the model from
learning "topic X -> always answer in language Y" instead of mirroring
whatever language the user asked in.

Output: one JSON object per line, `{"messages": [...], "meta": {...}}`
-- standard chat format most LoRA trainers accept directly.
