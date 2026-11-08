"""Loaders returning documents as {id, image, target, kind, group}."""

import json
from pathlib import Path

CORD_REPO = "naver-clova-ix/cord-v2"
LIST_KEYS = ["menu", "sub", "void_menu"]


def normalize_cord(obj):
    """Make item containers always lists so the target format is stable."""
    if isinstance(obj, list):
        out = []
        for value in obj:
            out.append(normalize_cord(value))
        return out
    if not isinstance(obj, dict):
        return obj
    out = {}
    for key, value in obj.items():
        if key in LIST_KEYS and isinstance(value, dict):
            value = [value]
        out[key] = normalize_cord(value)
    return out


class Documents:
    """Wraps a Hugging Face dataset with an image and a ground_truth column."""

    def __init__(self, dataset, kind):
        self.dataset = dataset
        self.kind = kind

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        if self.kind == "cord":
            target = normalize_cord(json.loads(row["ground_truth"])["gt_parse"])
            doc_id = "cord_" + str(index)
            group = "cord"
        else:
            target = json.loads(row["ground_truth"])
            doc_id = row["id"]
            group = "seen" if row["seen"] else "unseen"
        image = row["image"].convert("RGB")
        return {"id": doc_id, "image": image, "target": target, "kind": self.kind, "group": group}


def load_cord(split):
    from datasets import load_dataset

    return Documents(load_dataset(CORD_REPO, split=split), "cord")


def invoices_from_folder(folder):
    """Build a dataset from the files written by the invoice generator."""
    from datasets import Dataset, Image

    folder = Path(folder)
    columns = {"image": [], "id": [], "template": [], "seen": [], "ground_truth": []}
    with open(folder / "annotations.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            columns["image"].append(str(folder / row["file"]))
            for key in ["id", "template", "seen", "ground_truth"]:
                columns[key].append(row[key])
    return Dataset.from_dict(columns).cast_column("image", Image())


def load_invoices(source, split):
    """Source is a local folder from the generator or a Hub dataset id."""
    if Path(source).is_dir():
        return Documents(invoices_from_folder(Path(source) / split), "invoice")
    from datasets import load_dataset

    return Documents(load_dataset(source, split=split), "invoice")


def load_documents(name, split, invoice_source=None):
    if name == "cord":
        return load_cord(split)
    return load_invoices(invoice_source, split)
