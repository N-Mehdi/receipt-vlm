"""Model loading and extraction shared by the Gradio demo and the API."""

import os
from functools import lru_cache

from receipt_vlm.model import generate, load_model, load_processor
from receipt_vlm.parsing import parse_json

ADAPTER = os.environ.get("ADAPTER", "N-Mehdi/qwen3-vl-2b-receipts-lora")
KINDS = ["invoice", "cord"]


@lru_cache(maxsize=1)
def get_model():
    """Loaded once, on first use."""
    return load_processor(), load_model(adapter=ADAPTER)


def extract(image, kind="invoice"):
    """Return the parsed JSON (None if invalid) and the raw output."""
    if kind not in KINDS:
        raise ValueError("kind must be one of " + ", ".join(KINDS))
    processor, model = get_model()
    doc = {"image": image.convert("RGB"), "kind": kind}
    out = generate(model, processor, [doc])[0]
    return {
        "kind": kind,
        "data": parse_json(out["text"]),
        "raw": out["text"],
        "truncated": out["truncated"],
    }
