from receipt_vlm.parsing import parse_json, strip_fences, to_target_text


def test_plain_json():
    assert parse_json('{"a": "1"}') == {"a": "1"}


def test_markdown_fence():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert parse_json('```json\n{"a": {"b": [1, 2]}}\n```') == {"a": {"b": [1, 2]}}


def test_text_around_json():
    assert parse_json('Here is the result: {"a": 1} I hope it helps.') == {"a": 1}


def test_invalid_or_truncated_json():
    assert parse_json('{"a": [1, 2') is None
    assert parse_json("no json here") is None
    assert parse_json("") is None


def test_non_object_json():
    assert parse_json("[1, 2]") is None


def test_target_text_round_trip():
    obj = {"seller": {"name": "Société Générale"}, "total_ttc": 12.5}
    text = to_target_text(obj)
    assert "é" in text
    assert " " not in text.replace("Société Générale", "")
    assert parse_json(text) == obj
