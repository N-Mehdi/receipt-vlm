"""Model loading and batched generation."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from receipt_vlm.prompts import build_messages

BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# At most 1024 visual tokens per image (one token per 32x32 pixels).
MAX_PIXELS = 1024 * 32 * 32
MIN_PIXELS = 64 * 32 * 32


def load_processor(padding_side="left"):
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, size={"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}
    )
    processor.tokenizer.padding_side = padding_side
    return processor


def load_model(adapter=None, quantize=False):
    """fp16 on the GPU, fp32 on CPU (faster there); optionally 4-bit NF4 and a merged adapter."""
    if torch.cuda.is_available():
        kwargs = {"dtype": torch.float16, "device_map": {"": 0}}
    else:
        kwargs = {"dtype": torch.float32}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if not quantize:
            model = model.merge_and_unload()
    model.eval()
    return model


def prompt_texts(processor, docs):
    texts = []
    for doc in docs:
        texts.append(
            processor.apply_chat_template(
                build_messages(doc["kind"]), tokenize=False, add_generation_prompt=True
            )
        )
    return texts


@torch.no_grad()
def generate(model, processor, docs, max_new_tokens=1024):
    """Greedy decoding; returns one dict per document with text, length and truncation flag."""
    images = []
    for doc in docs:
        images.append(doc["image"])
    inputs = processor(
        text=prompt_texts(processor, docs), images=images, padding=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[:, inputs["input_ids"].shape[1] :].tolist()
    stop_ids = model.generation_config.eos_token_id
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    results = []
    for tokens in new_tokens:
        length = len(tokens)
        for position, token in enumerate(tokens):
            if token in stop_ids:
                length = position
                break
        text = processor.tokenizer.decode(tokens[:length], skip_special_tokens=True)
        results.append({"text": text, "n_tokens": length, "truncated": length == len(tokens)})
    return results
