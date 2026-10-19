# train

LoRA fine-tune on the dataset from `../dataset/`. CUDA-native
(transformers + peft + trl) -- this is for an NVIDIA box, not the
Apple Silicon/MLX stack used elsewhere in this account's other repos.

## Status

Written, **not yet run against real hardware**. Open questions before
a real run:

- Exact base checkpoint: which specific Nemotron release/size, and
  does it have a plain HF `transformers`-loadable checkpoint (as
  opposed to only NeMo-framework or GGUF/quantized-only releases)?
- If it's an MoE checkpoint (e.g. an "A3B"-style Nemotron), whether
  PEFT's `all-linear` auto-detect does something sane on it, or
  whether router/expert layers need an explicit target-module list
  instead -- untested.
- Where it runs: intended for a borrowed DGX Spark (GB10, 121GB
  unified memory) -- keep everything under `~/`, no `sudo`/system
  changes on that box.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train_lora.py \
  --model <hf-repo-id> \
  --dataset ../dataset/dataset.jsonl \
  --output-dir ./out \
  --lora-r 16 --epochs 2
```
