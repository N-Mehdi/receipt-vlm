"""Gradio demo, usable as a Hugging Face Space entry point."""

import json
from pathlib import Path

import gradio as gr

from receipt_vlm.service import KINDS, extract

EXAMPLES = Path(__file__).parent / "examples"
INTRO = """# receipt-vlm
Qwen3-VL-2B fine-tuned with QLoRA to extract French invoices and receipts (CORD) as JSON.
Training invoices are synthetic: results on real documents are not measured.

**Latency.** On a T4 GPU: about 6 s per receipt and 14 s per invoice. On CPU it is much
slower: one to three minutes per document, plus a few minutes to load the model on the
first request. The examples below show results precomputed on a T4 and load instantly."""


def run(image, kind):
    if image is None:
        return {"error": "upload an image"}
    result = extract(image, kind)
    if result["data"] is None:
        return {"error": "the output is not valid JSON", "raw": result["raw"]}
    return result["data"]


def load_examples():
    rows = []
    for item in json.loads((EXAMPLES / "examples.json").read_text(encoding="utf-8")):
        rows.append([str(EXAMPLES / item["file"]), item["kind"], item["result"]])
    return rows


with gr.Blocks(title="receipt-vlm") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Receipt or invoice")
            kind = gr.Radio(KINDS, value="invoice", label="Document type")
            button = gr.Button("Extract", variant="primary")
        output = gr.JSON(label="Extracted fields")
    gr.Examples(
        load_examples(),
        inputs=[image, kind, output],
        label="Examples (result precomputed on a T4, no computation on click)",
    )
    button.click(run, inputs=[image, kind], outputs=output)

if __name__ == "__main__":
    demo.launch()
