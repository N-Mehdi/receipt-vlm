"""HTTP API: POST an image to /extract and get the JSON back. The Gradio demo is served at /demo."""

import io
from typing import Annotated
from urllib.error import URLError

import gradio as gr
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError

from app.app import demo
from receipt_vlm.service import BACKEND, KINDS, extract

api = FastAPI(
    title="receipt-vlm",
    description="Extract receipts and French invoices as JSON. Interactive demo: [/demo](/demo).",
)


@api.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@api.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND}


@api.post("/extract")
def extract_endpoint(
    file: Annotated[UploadFile, File()],
    kind: Annotated[str, Query(description="invoice or cord")] = "invoice",
):
    if kind not in KINDS:
        raise HTTPException(422, "kind must be one of " + ", ".join(KINDS))
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        image.load()
    except UnidentifiedImageError as error:
        raise HTTPException(400, "the file is not a readable image") from error
    try:
        return extract(image, kind)
    except URLError as error:
        raise HTTPException(503, "llama-server is not reachable") from error


app = gr.mount_gradio_app(api, demo, path="/demo")
