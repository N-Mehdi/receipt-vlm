"""Write the model card of the GGUF files and, with --upload, publish them on the Hub."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

FILES = [
    "qwen3-vl-2b-receipts-Q8_0.gguf",
    "qwen3-vl-2b-receipts-Q4_K_M.gguf",
    "mmproj-qwen3-vl-2b-receipts-f16.gguf",
]

MODEL_CARD = """---
base_model: Qwen/Qwen3-VL-2B-Instruct
base_model_relation: finetune
license: apache-2.0
datasets:
- naver-clova-ix/cord-v2
- N-Mehdi/french-invoices-synth
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
---
# qwen3-vl-2b-receipts-gguf

GGUF files of [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
fine-tuned with QLoRA to extract receipts (CORD) and French invoices as JSON, for CPU inference
with llama.cpp. The [LoRA adapter](https://huggingface.co/N-Mehdi/qwen3-vl-2b-receipts-lora)
was merged into the base model in fp16, then converted with llama.cpp b12282.
Code, prompts and evaluation: [github.com/N-Mehdi/receipt-vlm](https://github.com/N-Mehdi/receipt-vlm).

## Files

| File | Content | Size |
|---|---|---|
{files}
The vision projector (`mmproj`) is needed with either language model.

## Licenses

- Model: Apache-2.0, like the base model and the adapter.
- Training data: [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC BY 4.0)
  and [synthetic French invoices](https://huggingface.co/datasets/N-Mehdi/french-invoices-synth)
  (Apache-2.0). No real invoice was used for training or testing.

## Usage

```bash
llama-server -m qwen3-vl-2b-receipts-Q8_0.gguf --mmproj mmproj-qwen3-vl-2b-receipts-f16.gguf \\
    -c 4096 --image-min-tokens 64 --image-max-tokens 1024
```

Send the image first, then the prompt from
[prompts.py](https://github.com/N-Mehdi/receipt-vlm/blob/main/src/receipt_vlm/prompts.py),
with `temperature: 0` (the default sampling stored in the files is the base model's).

## Results

100 test documents per set, greedy decoding. Exact match requires every field to be right;
F1 is the micro F1 over (field path, value) pairs. The fp16 reference runs in transformers.

| Test set | Model | Valid JSON | Exact match | Field F1 |
|---|---|---|---|---|
| Invoices | fp16 reference | 1.00 | 0.63 | 0.979 |
| | Q8_0 | 1.00 | 0.62 | 0.980 |
| | Q4_K_M | 1.00 | 0.50 | 0.969 |
| CORD | fp16 reference | 0.99 | 0.35 | 0.832 |
| | Q8_0 | 0.99 | 0.35 | 0.825 |
| | Q4_K_M | 1.00 | 0.23 | 0.807 |

No output was truncated at 1024 tokens. Q4_K_M mostly loses on long strings to copy exactly
(addresses, SIRET and VAT numbers).

## CPU latency

Batch size 1, 20 documents per set, on an Intel Core i7-1365U laptop (2 performance and
8 efficiency cores). llama.cpp uses 4 threads for generation and 12 for prompt processing,
transformers runs in fp32 with 10 threads. Tokens/s counts generated tokens over the total time.

| Test set | Model | Mean | p50 | p95 | Tokens/s |
|---|---|---|---|---|---|
| Invoices | transformers fp32 | 180 s | 171 s | 249 s | 1.6 |
| | Q8_0 | 50 s | 49 s | 66 s | 6.0 |
| | Q4_K_M | 43 s | 43 s | 57 s | 6.8 |
| CORD | transformers fp32 | 100 s | 102 s | 146 s | 1.0 |
| | Q8_0 | 60 s | 73 s | 83 s | 1.8 |
| | Q4_K_M | 54 s | 66 s | 75 s | 2.0 |

On this CPU, prompt processing dominates on receipts: CORD photos use 838 visual tokens on
average (60 of 100 reach the 1024 cap), and the vision encoder and prompt processing are
compute bound, so they do not speed up with quantization or more threads.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-dir", default="models/gguf")
    parser.add_argument("--repo", default="N-Mehdi/qwen3-vl-2b-receipts-gguf")
    parser.add_argument("--upload", action="store_true", help="publish (public repo)")
    args = parser.parse_args()

    folder = Path(args.gguf_dir)
    rows = []
    for name in FILES:
        if name.startswith("mmproj"):
            content = "vision projector, f16"
        else:
            content = "language model, " + name.split("-")[-1].removesuffix(".gguf")
        size = (folder / name).stat().st_size / 1e9
        rows.append(f"| `{name}` | {content} | {size:.2f} GB |\n")
    (folder / "README.md").write_text(MODEL_CARD.format(files="".join(rows)), encoding="utf-8")
    print("model card written to", folder / "README.md")
    if not args.upload:
        return
    api = HfApi()
    api.create_repo(args.repo, private=False, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=folder, allow_patterns=FILES + ["README.md"])
    print("uploaded to", args.repo)


if __name__ == "__main__":
    main()
