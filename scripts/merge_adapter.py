"""Merge the LoRA adapter into the base model in fp16 and save the full model."""

import argparse

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText

from receipt_vlm.model import BASE_MODEL

PROCESSOR_FILES = ["*.json", "merges.txt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="N-Mehdi/qwen3-vl-2b-receipts-lora")
    parser.add_argument("--out", default="models/merged")
    args = parser.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out)
    # Original config, tokenizer and processor files (same architecture), in the format read by
    # the older transformers of the GGUF converter.
    snapshot_download(BASE_MODEL, allow_patterns=PROCESSOR_FILES, local_dir=args.out)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
