"""
Dataset Loading and Preprocessing for PubMedQA

This script:
- Loads the domain-specific Q&A dataset: `qiaojin/PubMedQA` with subset `pqa_labeled`.
- Splits into train/validation/test with a fixed seed for reproducibility.
- Converts the short answer `final_decision` (yes/no/maybe) into numeric labels.
- Creates a causal LM-friendly text field: "Question... Context... Answer..." (answer is final_decision).
- Tokenizes using Hugging Face `AutoTokenizer`.
- Saves a small processed and tokenized sample to `data/processed_dataset.json` (or .csv) for review.

Usage (examples):
  python -m src.data_prep \
    --model_name gpt2 \
    --output_sample_path data/processed_dataset.json \
    --sample_size 100 \
    --max_length 512

Notes:
- For causal LM models like `gpt2`, we set `pad_token` to `eos_token` if needed to avoid padding issues.
- Adjust `model_name` as desired (e.g., a domain-specific LLM tokenizer).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

# Import the datasets package as a module to avoid fragile symbol imports
import datasets as ds
from transformers import AutoTokenizer


LABEL_MAP: Dict[str, int] = {"yes": 0, "no": 1, "maybe": 2}


def _normalize_context(ctx):
    """Ensure context is a single string (some datasets store lists)."""
    if ctx is None:
        return ""
    if isinstance(ctx, list):
        return " ".join(str(x) for x in ctx)
    return str(ctx)


def load_and_split(seed: int = 42) -> Tuple["ds.Dataset", "ds.Dataset", "ds.Dataset"]:
    """Load PubMedQA (pqa_labeled) and split into train/val/test.

    Returns (train, val, test) datasets.
    """
    ds_dict = ds.load_dataset("qiaojin/PubMedQA", "pqa_labeled")

    # Some datasets provide only one split (e.g., 'train'). Use what's available as the base.
    if isinstance(ds_dict, ds.DatasetDict):
        if "train" in ds_dict:
            base = ds_dict["train"]
        else:
            # Fallback: concatenate all splits if train isn't present
            parts = [split for split in ds_dict.values()]
            base = parts[0]
            for p in parts[1:]:
                base = base.concatenate(p)
    else:
        base = ds_dict

    # 80/20 initial split
    split_1 = base.train_test_split(test_size=0.2, seed=seed)
    train = split_1["train"]
    remainder = split_1["test"]

    # Split the 20% remainder into 10% val and 10% test
    split_2 = remainder.train_test_split(test_size=0.5, seed=seed)
    val = split_2["train"]
    test = split_2["test"]

    return train, val, test


def preprocess_labels(dataset: "ds.Dataset") -> "ds.Dataset":
    """Map final_decision yes/no/maybe -> numeric labels in new column 'label'."""
    def _map_label(ex):
        decision = (ex.get("final_decision") or "").strip().lower()
        ex["label"] = LABEL_MAP.get(decision, -1)
        return ex

    return dataset.map(_map_label, desc="Mapping final_decision -> label")


def select_columns(dataset: "ds.Dataset") -> "ds.Dataset":
    """Keep only the columns needed for training/review: question, context, label, final_decision."""
    keep = ["question", "context", "label", "final_decision"]

    # Ensure context is normalized to a string
    def _norm_context(ex):
        ex["context"] = _normalize_context(ex.get("context"))
        return ex

    ds = dataset.map(_norm_context, desc="Normalizing context text")

    # Remove extra columns if present
    drop_cols = [c for c in ds.column_names if c not in keep]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    return ds


def add_causal_lm_text(dataset: "ds.Dataset") -> "ds.Dataset":
    """Create a causal LM-friendly 'text' field combining question, context, and the short answer.

    The target here is the short answer `final_decision`. For full generative training,
    replace with long answers when available.
    """
    def _compose(ex):
        q = ex.get("question") or ""
        c = ex.get("context") or ""
        a = ex.get("final_decision") or ""
        ex["text"] = f"Question: {q}\nContext: {c}\nAnswer: {a}"
        return ex

    return dataset.map(_compose, desc="Composing causal LM text")


def tokenize_text(dataset: "ds.Dataset", tokenizer: AutoTokenizer, max_length: int = 512) -> "ds.Dataset":
    """Tokenize the 'text' field with truncation/padding.

    Returns a dataset with 'input_ids' and 'attention_mask'.
    """
    def _tok(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(_tok, batched=True, desc="Tokenizing text")


def save_sample(sample: "ds.Dataset", path: str, preview_token_count: int = 32) -> None:
    """Save a small processed sample to JSON or CSV for human review.

    Includes: question, context, final_decision, label, text, and the first N token ids.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows: List[Dict] = []
    for ex in sample:
        input_ids = ex.get("input_ids") or []
        rows.append(
            {
                "question": ex.get("question"),
                "context": ex.get("context"),
                "final_decision": ex.get("final_decision"),
                "label": ex.get("label"),
                "text": ex.get("text"),
                "input_ids_preview": input_ids[:preview_token_count],
            }
        )

    if path.lower().endswith(".csv"):
        fieldnames = list(rows[0].keys()) if rows else [
            "question",
            "context",
            "final_decision",
            "label",
            "text",
            "input_ids_preview",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                # Convert list to JSON string for CSV cell
                r = dict(r)
                if isinstance(r.get("input_ids_preview"), list):
                    r["input_ids_preview"] = json.dumps(r["input_ids_preview"], ensure_ascii=False)
                writer.writerow(r)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


def build_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Many causal models (e.g., gpt2) lack a pad token by default; set it to eos for batching.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Preprocess PubMedQA for LLM fine-tuning")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Tokenizer model name")
    parser.add_argument("--output_sample_path", type=str, default="data/processed_dataset.json", help="Path to save a small processed sample (.json or .csv)")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of examples to save for review")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    # 1) Load and split dataset
    train, val, test = load_and_split(seed=args.seed)

    # 2) Map labels and reduce columns
    train = preprocess_labels(train)
    val = preprocess_labels(val)
    test = preprocess_labels(test)

    train = select_columns(train)
    val = select_columns(val)
    test = select_columns(test)

    # 3) Create causal LM-friendly text field
    train = add_causal_lm_text(train)
    val = add_causal_lm_text(val)
    test = add_causal_lm_text(test)

    # 4) Tokenize with AutoTokenizer
    tokenizer = build_tokenizer(args.model_name)
    train_tok = tokenize_text(train, tokenizer, max_length=args.max_length)
    val_tok = tokenize_text(val, tokenizer, max_length=args.max_length)
    test_tok = tokenize_text(test, tokenizer, max_length=args.max_length)

    # 5) Save a small sample for review
    sample = train_tok.select(range(min(args.sample_size, len(train_tok))))
    save_sample(sample, args.output_sample_path)

    # 6) Print a brief summary to stdout
    print("Preprocessing complete.")
    print(f"Train/Val/Test sizes: {len(train_tok)}/{len(val_tok)}/{len(test_tok)}")
    print(f"Sample saved to: {args.output_sample_path}")
    # Optional: Save full tokenized datasets to disk for later loading
    # train_tok.save_to_disk("data/train_tok")
    # val_tok.save_to_disk("data/val_tok")
    # test_tok.save_to_disk("data/test_tok")


if __name__ == "__main__":
    main()
