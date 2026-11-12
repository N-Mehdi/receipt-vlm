"""Client for llama-server (OpenAI-compatible API), with the prompts of the transformers path."""

import base64
import io
import json
import os
import urllib.request

from receipt_vlm.prompts import build_messages

LLAMACPP_URL = os.environ.get("LLAMACPP_URL", "http://localhost:8080")


def image_part(image):
    """PNG keeps the pixels unchanged; the server resizes like the processor (32 px per token)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def call(url, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url + path, data=data)
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def generate(docs, max_new_tokens=1024, url=LLAMACPP_URL):
    """Greedy decoding, one request per document; same output as model.generate."""
    results = []
    for doc in docs:
        messages = build_messages(doc["kind"])
        messages[0]["content"][0] = image_part(doc["image"])
        body = {"messages": messages, "max_tokens": max_new_tokens, "temperature": 0}
        reply = call(url, "/v1/chat/completions", body)
        choice = reply["choices"][0]
        results.append(
            {
                "text": choice["message"]["content"],
                "n_tokens": reply["usage"]["completion_tokens"],
                "truncated": choice["finish_reason"] == "length",
            }
        )
    return results


def server_info(url=LLAMACPP_URL):
    """Model file and llama.cpp build, for the reports."""
    props = call(url, "/props", timeout=10)
    return {"model_path": props.get("model_path"), "build_info": props.get("build_info")}
