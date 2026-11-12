"""Evaluate the base model, a LoRA adapter or a llama-server GGUF on CORD or the invoices."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import transformers

from receipt_vlm import llamacpp
from receipt_vlm.data import load_documents
from receipt_vlm.metrics import score
from receipt_vlm.model import generate, load_model, load_processor
from receipt_vlm.parsing import parse_json


def run_predictions(predict, docs, batch_size):
    rows = []
    for start in range(0, len(docs), batch_size):
        batch = []
        for i in range(start, min(start + batch_size, len(docs))):
            batch.append(docs[i])
        outputs = predict(batch)
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


def cpu_name():
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return None


def measure_latency(predict, docs, n_docs):
    """Batch size 1, after one warmup document: seconds per document and generated tokens/s."""
    cuda = torch.cuda.is_available()
    predict([docs[0]])
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    seconds = []
    tokens = 0
    for i in range(min(n_docs, len(docs))):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = predict([docs[i]])[0]
        if cuda:
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
        tokens += out["n_tokens"]
        print(f"latency {i + 1}/{n_docs}: {seconds[-1]:.1f} s", flush=True)
    result = {
        "n_docs": len(seconds),
        "mean_s": float(np.mean(seconds)),
        "p50_s": float(np.percentile(seconds, 50)),
        "p95_s": float(np.percentile(seconds, 95)),
        "tokens_per_s": tokens / sum(seconds),
    }
    if cuda:
        result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    else:
        result["cpu"] = cpu_name()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cord", "invoice"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--invoice-source", help="local folder or Hub dataset id")
    parser.add_argument("--backend", choices=["transformers", "llamacpp"], default="transformers")
    parser.add_argument("--adapter", help="LoRA adapter folder or Hub repo (transformers)")
    parser.add_argument("--llamacpp-url", default=llamacpp.LLAMACPP_URL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--latency-docs", type=int, default=0)
    parser.add_argument("--latency-only", action="store_true", help="skip the quality run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docs = load_documents(args.dataset, args.split, args.invoice_source)
    if args.limit:
        docs.dataset = docs.dataset.select(range(min(args.limit, len(docs))))
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "backend": args.backend,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.backend == "llamacpp":
        predict = partial(
            llamacpp.generate, max_new_tokens=args.max_new_tokens, url=args.llamacpp_url
        )
        report.update(llamacpp.server_info(args.llamacpp_url))
    else:
        processor = load_processor()
        model = load_model(adapter=args.adapter)
        predict = partial(generate, model, processor, max_new_tokens=args.max_new_tokens)
        report["adapter"] = args.adapter
        report["torch"] = torch.__version__
        report["transformers"] = transformers.__version__
        report["torch_threads"] = torch.get_num_threads()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.latency_only:
        start = time.perf_counter()
        rows = run_predictions(predict, docs, args.batch_size)
        report["seconds"] = time.perf_counter() - start
        report["all"] = summarize(rows)
        groups = {}
        for row in rows:
            groups.setdefault(row["group"], []).append(row)
        if len(groups) > 1:
            for name, group_rows in groups.items():
                report[name] = summarize(group_rows)
        with open(out.with_suffix(".predictions.jsonl"), "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {}
        for key in ["valid_json", "exact_match", "f1", "truncated_rate"]:
            summary[key] = round(report["all"][key], 4)
        print(json.dumps(summary))
    if args.latency_docs:
        report["latency"] = measure_latency(predict, docs, args.latency_docs)
        print(json.dumps(report["latency"]))
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
