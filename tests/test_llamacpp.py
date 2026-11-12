from PIL import Image

from receipt_vlm import llamacpp
from receipt_vlm.prompts import PROMPTS


def fake_server(requests, finish_reason):
    def call(url, path, body=None, timeout=600):
        requests.append(body)
        message = {"content": '{"total": {"total_price": "10"}}'}
        return {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"completion_tokens": 12},
        }

    return call


def test_request_uses_same_prompt_and_greedy_decoding(monkeypatch):
    requests = []
    monkeypatch.setattr(llamacpp, "call", fake_server(requests, "stop"))
    doc = {"image": Image.new("RGB", (64, 64), "white"), "kind": "cord"}
    out = llamacpp.generate([doc], max_new_tokens=256)[0]

    content = requests[0]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": PROMPTS["cord"]}
    assert requests[0]["temperature"] == 0
    assert requests[0]["max_tokens"] == 256
    assert out == {"text": '{"total": {"total_price": "10"}}', "n_tokens": 12, "truncated": False}


def test_length_stop_is_truncation(monkeypatch):
    monkeypatch.setattr(llamacpp, "call", fake_server([], "length"))
    doc = {"image": Image.new("RGB", (64, 64)), "kind": "invoice"}
    assert llamacpp.generate([doc])[0]["truncated"]
