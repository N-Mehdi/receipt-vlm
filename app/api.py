"""HTTP API: POST an image to /extract and get the JSON back."""

import io
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app.service import KINDS, extract

app = FastAPI(title="receipt-vlm")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
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
    return extract(image, kind)
