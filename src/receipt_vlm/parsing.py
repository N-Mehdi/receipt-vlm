"""Turn raw model output into a JSON object."""

import json


def strip_fences(text):
    """Remove a surrounding markdown code block if present."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline != -1 else ""
        end = text.rfind("```")
        if end != -1:
            text = text[:end]
    return text.strip()


def parse_json(text):
    """Return the first JSON object found in the text, or None."""
    text = strip_fences(text)
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def to_target_text(obj):
    """Compact serialization used as training target."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
