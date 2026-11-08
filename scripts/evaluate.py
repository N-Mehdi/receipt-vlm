"""Evaluate the base model or a LoRA adapter on CORD or on the synthetic invoices."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from receipt_vlm.data import load_documents
from receipt_vlm.metrics import score
from receipt_vlm.model import generate, load_model, load_processor
from receipt_vlm.parsing import parse_json


def run_predictions(model, processor, docs, batch_size, max_new_tokens):
    rows = []
    for start in range(0, len(docs), batch_size):
        batch = []
        for i in range(start, min(start + batch_size, len(docs))):
            batch.append(docs[i])
        outputs = generate(model, processor, batch, max_new_tokens)
        for doc, out in zip(batch, outputs, strict=True):
            rows.append(
                {
                    "id": doc["id"],
                    "group": doc["group"],
                    "target": doc["target"],
                    "prediction": parse_json(out["text"]),
                    **out,
                }
            )
        print(f"{len(rows)}/{len(docs)} documents", flush=True)
    return rows


def summarize(rows):
    preds = []
    gts = []
    truncated = 0
    for row in rows:
        preds.append(row["prediction"])
        gts.append(row["target"])
        truncated += row["truncated"]
    result = score(preds, gts)
    result["truncated_rate"] = truncated / len(rows)
    return result


def measure_latency(model, processor, docs, n_docs, max_new_tokens):
    """Batch size 1: seconds per document, generated tokens per second and peak memory."""
    generate(model, processor, [docs[0]], max_new_tokens)
    torch.cuda.reset_peak_memory_stats()
    seconds = []
    tokens = 0
    for i in range(min(n_docs, len(docs))):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = generate(model, processor, [docs[i]], max_new_tokens)[0]
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
        tokens += out["n_tokens"]
    return {
        "n_docs": len(seconds),
        "mean_s": float(np.mean(seconds)),
        "p50_s": float(np.percentile(seconds, 50)),
        "p95_s": float(np.percentile(seconds, 95)),
        "tokens_per_s": tokens / sum(seconds),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cord", "invoice"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--invoice-source", help="local folder or Hub dataset id")
    parser.add_argument("--adapter", help="LoRA adapter folder or Hub repo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--latency-docs", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docs = load_documents(args.dataset, args.split, args.invoice_source)
    if args.limit:
        docs.dataset = docs.dataset.select(range(min(args.limit, len(docs))))
    processor = load_processor()
    model = load_model(adapter=args.adapter)

    start = time.perf_counter()
    rows = run_predictions(model, processor, docs, args.batch_size, args.max_new_tokens)
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "adapter": args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "seconds": time.perf_counter() - start,
        "all": summarize(rows),
    }
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    if len(groups) > 1:
        for name, group_rows in groups.items():
            report[name] = summarize(group_rows)
    if args.latency_docs:
        report["latency"] = measure_latency(
            model, processor, docs, args.latency_docs, args.max_new_tokens
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with open(out.with_suffix(".predictions.jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {}
    for key in ["valid_json", "exact_match", "f1", "truncated_rate"]:
        summary[key] = round(report["all"][key], 4)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
