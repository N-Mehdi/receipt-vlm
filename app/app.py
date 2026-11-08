"""Gradio demo, usable as a Hugging Face Space entry point."""

import gradio as gr

from app.service import KINDS, extract


def run(image, kind):
    if image is None:
        return {"error": "upload an image"}
    result = extract(image, kind)
    if result["data"] is None:
        return {"error": "the output is not valid JSON", "raw": result["raw"]}
    return result["data"]


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="pil", label="Receipt or invoice"),
        gr.Radio(KINDS, value="invoice", label="Document type"),
    ],
    outputs=gr.JSON(label="Extracted fields"),
    title="receipt-vlm",
    description=(
        "Qwen3-VL-2B fine-tuned with QLoRA to extract French invoices (synthetic training data) "
        "and receipts (CORD) as JSON. Runs on CPU here, expect about a minute per document."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
