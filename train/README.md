# train

LoRA fine-tune on the dataset from `../dataset/`. Two paths, pick by
architecture:

- **Plain dense transformer** → `train_lora.py` (generic
  transformers+peft+trl, CUDA-native).
- **Nemotron-3.5-Lightning-30B-A3B** (or anything else with a hybrid
  Mamba/MoE mixer) → `axolotl_nemotron_30b_a3b.yaml` +
  `setup_spark.sh`. The generic script does **not** work on this
  architecture — see below.

## Why Nemotron needs Axolotl, not the generic script

Checked against this account's own prior work fine-tuning this exact
checkpoint for `ipsupport-code`
([quant-ternary/nemotron-extreme-quant/docs/FINETUNING.md](https://github.com/rromenskyi/quant-ternary/blob/main/nemotron-extreme-quant/docs/FINETUNING.md),
validated in real usage). NemotronH (hybrid Mamba2 + Attention + MoE, 52
layers):

- Attention lives in `NemotronHBlock.mixer`, not `layer.self_attn`.
- MoE's `up_proj`/`down_proj` are 3D `nn.Parameter` tensors
  (`[num_experts, out_dim, in_dim]`), not `nn.Linear` modules — a plain
  `peft.LoraConfig(target_modules=...)` can't reach them at all; PEFT's
  `all-linear` auto-detect (what `train_lora.py` falls back to) would
  silently miss the MoE experts entirely, which are ~93% of this
  model's parameters.
- Axolotl has ready-made NemotronH support for exactly this, including
  the `lora_target_parameters` path for the 3D expert tensors. Upstream
  example: `examples/nemotron-h/nemotron-3_5-lightning-30b-a3b-qlora.yaml`
  in [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl).

`axolotl_nemotron_30b_a3b.yaml` here is that upstream example, with
`lora_r`/`lora_alpha` and `lora_target_parameters` changed to the
config already validated working on this checkpoint (q/k/v/o_proj via
`lora_target_modules`, MoE expert up/down_proj via
`lora_target_parameters`, `lora_dropout: 0` -- required, the 3D-tensor
LoRA path doesn't support dropout). **Merging requires
`--merge_method legacy`** — Axolotl's default merge silently drops the
LoRA weights applied via `lora_target_parameters` on this architecture.

## Status

Written, **not yet run against real hardware**.

- Base checkpoint: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
  (confirmed real, transformers-loadable — this is the exact model
  already running quantized in the user's own LLMTray locally, just not
  the bf16 source used for training).
- Target: a borrowed DGX Spark (`ssh dgx`, GB10, 121GB unified, **Grace
  CPU = aarch64** — unlike the x86_64 RunPod pod the above recipe was
  originally validated on). `setup_spark.sh` adapts that pod's install
  order but can't reuse its x86_64 prebuilt-wheel shortcuts for
  flash-attn/causal-conv1d/mamba_ssm — expect a slower first-time
  source build, untested on this specific hardware.
- No sudo, nothing outside `$HOME` on the Spark — borrowed machine.

## Setup (generic path)

```bash
pip install -r requirements.txt
```

## Usage (generic path)

```bash
python train_lora.py \
  --model <hf-repo-id> \
  --dataset ../dataset/dataset.jsonl \
  --output-dir ./out \
  --lora-r 16 --epochs 2
```

## Usage (Nemotron / Axolotl path)

```bash
ssh dgx 'bash -s' < setup_spark.sh
scp axolotl_nemotron_30b_a3b.yaml dgx:~/
scp ../dataset/dataset.jsonl dgx:~/legacy-docs-lora-dataset.jsonl
ssh dgx
source ~/legacy-docs-lora-venv/bin/activate
axolotl train axolotl_nemotron_30b_a3b.yaml
axolotl merge-lora axolotl_nemotron_30b_a3b.yaml --merge_method legacy
```
