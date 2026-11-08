"""QLoRA fine-tuning on CORD train and the synthetic invoices."""

import argparse
import math
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from torch.utils.data import ConcatDataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from receipt_vlm.data import load_cord, load_invoices
from receipt_vlm.model import BASE_MODEL, load_model, load_processor
from receipt_vlm.parsing import to_target_text
from receipt_vlm.prompts import build_messages

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


class Collator:
    """Tokenizes prompt and answer; only answer tokens contribute to the loss."""

    def __init__(self, processor):
        self.processor = processor
        self.marker = processor.tokenizer.encode("<|im_start|>assistant\n")

    def answer_start(self, ids):
        size = len(self.marker)
        for i in range(len(ids) - size, -1, -1):
            if ids[i : i + size] == self.marker:
                return i + size
        raise ValueError("assistant marker not found")

    def __call__(self, docs):
        texts = []
        images = []
        for doc in docs:
            messages = build_messages(doc["kind"], to_target_text(doc["target"]))
            texts.append(self.processor.apply_chat_template(messages, tokenize=False))
            images.append(doc["image"])
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for row in range(labels.shape[0]):
            labels[row, : self.answer_start(batch["input_ids"][row].tolist())] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class Guard(TrainerCallback):
    """Stops on a NaN loss or when the projected run time exceeds the budget."""

    def __init__(self, max_minutes):
        self.max_seconds = max_minutes * 60
        self.start = time.time()
        self.reason = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and math.isnan(logs["loss"]):
            self.reason = f"NaN loss at step {state.global_step}"
            control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 10:
            projected = (time.time() - self.start) / 10 * state.max_steps
            print(f"projected training time: {projected / 60:.1f} min", flush=True)
            if projected > self.max_seconds:
                self.reason = f"projected {projected / 60:.0f} min exceeds the budget"
                control.should_training_stop = True


MODEL_CARD = """---
base_model: {base}
library_name: peft
license: apache-2.0
datasets:
- naver-clova-ix/cord-v2
---
# {repo}

LoRA adapter (QLoRA NF4, r=16) for {base} that extracts receipts and French invoices as JSON.
Trained for one epoch on CORD v2 train and {n_invoices} synthetic French invoices.
No real invoices were used for training or evaluation.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--invoice-source", required=True, help="local folder or Hub dataset id")
    parser.add_argument("--output-dir", default="outputs/adapter")
    parser.add_argument("--hub-repo", help="private Hub repo for the adapter")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-minutes", type=float, default=100)
    parser.add_argument("--limit", type=int, help="documents per source, for smoke tests")
    args = parser.parse_args()

    cord = load_cord("train")
    invoices = load_invoices(args.invoice_source, "train")
    if args.limit:
        cord.dataset = cord.dataset.select(range(args.limit))
        invoices.dataset = invoices.dataset.select(range(args.limit))
    train_set = ConcatDataset([cord, invoices])
    print(f"training documents: {len(cord)} CORD + {len(invoices)} invoices", flush=True)

    processor = load_processor(padding_side="right")
    model = load_model(quantize=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="no",
        report_to=["wandb"],
        run_name="qlora-r16",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=0,
    )
    guard = Guard(args.max_minutes)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        data_collator=Collator(processor),
        callbacks=[guard],
    )
    trainer.train()
    if guard.reason:
        print(f"training stopped: {guard.reason}", flush=True)
        sys.exit(1)

    model.save_pretrained(args.output_dir)
    if args.hub_repo:
        card = MODEL_CARD.format(base=BASE_MODEL, repo=args.hub_repo, n_invoices=len(invoices))
        Path(args.output_dir, "README.md").write_text(card)
        api = HfApi()
        api.create_repo(args.hub_repo, private=True, exist_ok=True)
        api.upload_folder(
            repo_id=args.hub_repo,
            folder_path=args.output_dir,
            allow_patterns=["adapter_*", "README.md"],
        )
        print(f"adapter uploaded to {args.hub_repo}", flush=True)


if __name__ == "__main__":
    main()
