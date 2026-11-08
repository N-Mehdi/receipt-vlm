"""Generate the synthetic invoice splits and optionally publish them as a private Hub dataset."""

import argparse
import time

from receipt_vlm.data import invoices_from_folder
from receipt_vlm.invoices import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/invoices")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test-seen", type=int, default=50)
    parser.add_argument("--test-unseen", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push-to-hub", help="private Hub dataset id")
    args = parser.parse_args()

    start = time.perf_counter()
    generate(args.out, args.train, args.val, args.test_seen, args.test_unseen, args.seed)
    print(f"generated in {time.perf_counter() - start:.0f} s", flush=True)

    if args.push_to_hub:
        from datasets import DatasetDict

        splits = DatasetDict()
        for split in ["train", "validation", "test"]:
            splits[split] = invoices_from_folder(f"{args.out}/{split}")
        splits.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed to {args.push_to_hub}", flush=True)


if __name__ == "__main__":
    main()
