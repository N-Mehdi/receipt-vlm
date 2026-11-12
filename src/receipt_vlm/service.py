"""Extraction shared by the Gradio demo and the API.

BACKEND=transformers loads the model in this process; BACKEND=llamacpp calls llama-server
at LLAMACPP_URL and loads nothing.
"""

import os
from functools import lru_cache

from receipt_vlm import llamacpp
from receipt_vlm.parsing import parse_json

BACKEND = os.environ.get("BACKEND", "transformers")
ADAPTER = os.environ.get("ADAPTER", "N-Mehdi/qwen3-vl-2b-receipts-lora")
BACKENDS = ["transformers", "llamacpp"]
KINDS = ["invoice", "cord"]

if BACKEND not in BACKENDS:
    raise ValueError("BACKEND must be one of " + ", ".join(BACKENDS))


@lru_cache(maxsize=1)
def get_model():
    """Loaded once, on first use (transformers backend only)."""
    from receipt_vlm.model import load_model, load_processor

    return load_processor(), load_model(adapter=ADAPTER)


def run_model(doc):
    if BACKEND == "llamacpp":
        return llamacpp.generate([doc])[0]
    from receipt_vlm.model import generate

    processor, model = get_model()
    return generate(model, processor, [doc])[0]


def extract(image, kind="invoice"):
    """Return the parsed JSON (None if invalid) and the raw output."""
    if kind not in KINDS:
        raise ValueError("kind must be one of " + ", ".join(KINDS))
    out = run_model({"image": image.convert("RGB"), "kind": kind})
    return {
        "kind": kind,
        "data": parse_json(out["text"]),
        "raw": out["text"],
        "truncated": out["truncated"],
    }
