"""Field level F1, exact match and bootstrap confidence intervals."""

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

import numpy as np

NUMBER = re.compile(r"^-?\d+([.,]\d+)?$")


def normalize_value(value):
    """Lowercase, collapse spaces and write numbers in a canonical form."""
    text = " ".join(str(value).lower().split())
    compact = text.replace(" ", "")
    if NUMBER.match(compact):
        try:
            number = Decimal(compact.replace(",", ".")).normalize()
            return format(number, "f")
        except InvalidOperation:
            return text
    return text


def flatten(obj, prefix=""):
    """List of (field path, normalized value); list indices are dropped."""
    pairs = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = prefix + "." + str(key) if prefix else str(key)
            pairs.extend(flatten(value, path))
    elif isinstance(obj, list):
        for value in obj:
            pairs.extend(flatten(value, prefix))
    elif obj is not None and str(obj).strip() != "":
        pairs.append((prefix, normalize_value(obj)))
    return pairs


def field_counts(pred, gt):
    """Per field: [true positives, predicted, expected]."""
    pred_pairs = Counter(flatten(pred)) if isinstance(pred, dict) else Counter()
    gt_pairs = Counter(flatten(gt))
    counts = {}
    for (field, _), n in pred_pairs.items():
        counts.setdefault(field, [0, 0, 0])[1] += n
    for (field, _), n in gt_pairs.items():
        counts.setdefault(field, [0, 0, 0])[2] += n
    for pair, n in (pred_pairs & gt_pairs).items():
        counts[pair[0]][0] += n
    return counts


def f1(tp, n_pred, n_gt):
    if n_pred + n_gt == 0:
        return 1.0
    return 2.0 * tp / (n_pred + n_gt)


def bootstrap_ci(values_fn, n_docs, n_boot=1000, seed=0):
    """95% percentile interval of a statistic computed on resampled documents."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        stats.append(values_fn(rng.integers(0, n_docs, n_docs)))
    low, high = np.percentile(stats, [2.5, 97.5])
    return [float(low), float(high)]


def score(preds, gts, n_boot=1000):
    """Aggregate metrics; preds are dicts or None when the output was not valid JSON."""
    n = len(gts)
    tp = np.zeros(n)
    n_pred = np.zeros(n)
    n_gt = np.zeros(n)
    exact = np.zeros(n)
    valid = np.zeros(n)
    totals = {}
    for i in range(n):
        counts = field_counts(preds[i], gts[i])
        for field, (a, b, c) in counts.items():
            row = totals.setdefault(field, [0, 0, 0])
            row[0] += a
            row[1] += b
            row[2] += c
            tp[i] += a
            n_pred[i] += b
            n_gt[i] += c
        valid[i] = isinstance(preds[i], dict)
        exact[i] = valid[i] and Counter(flatten(preds[i])) == Counter(flatten(gts[i]))

    def micro_f1(idx):
        return f1(tp[idx].sum(), n_pred[idx].sum(), n_gt[idx].sum())

    per_field = {}
    for field in sorted(totals):
        a, b, c = totals[field]
        if c > 0:
            per_field[field] = {"f1": f1(a, b, c), "support": c}
    all_idx = np.arange(n)
    return {
        "n": n,
        "valid_json": float(valid.mean()),
        "valid_json_ci": bootstrap_ci(lambda idx: valid[idx].mean(), n, n_boot),
        "exact_match": float(exact.mean()),
        "exact_match_ci": bootstrap_ci(lambda idx: exact[idx].mean(), n, n_boot),
        "f1": micro_f1(all_idx),
        "f1_ci": bootstrap_ci(micro_f1, n, n_boot),
        "per_field": per_field,
    }
