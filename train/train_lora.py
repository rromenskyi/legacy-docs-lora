#!/usr/bin/env python3
"""LoRA fine-tune a causal LM on the chat-format JSONL produced by
../dataset/generate.py. CUDA-native (transformers + peft + trl) --
meant for an NVIDIA box (e.g. a DGX Spark), not Apple Silicon/MLX.

Usage:
    python train_lora.py \
        --model nvidia/<some-nemotron-checkpoint> \
        --dataset ../dataset/dataset.jsonl \
        --output-dir ./out \
        --lora-r 16 --epochs 2

Status: written but not yet run against real hardware -- target_modules
defaults to PEFT's "all-linear" auto-detection specifically because
this hasn't been validated against the exact Nemotron checkpoint/
architecture yet (MoE naming can need a narrower target list; check
once a specific checkpoint is picked).
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--dataset", required=True, help="chat-format JSONL from dataset/generate.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-target-modules", default="all-linear", help="comma-separated module names, or 'all-linear' for PEFT's auto-detection")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="bfloat16")

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)

    target_modules = args.lora_target_modules if args.lora_target_modules == "all-linear" else args.lora_target_modules.split(",")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
