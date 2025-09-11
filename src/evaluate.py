"""
Evaluate fine-tuned vs. base model on PubMedQA test set (classification).

This script:
- Loads the fine-tuned model from models/domain-llm and the base model checkpoint.
- Uses the same tokenizer for both (prefers the fine-tuned tokenizer if available).
- Prepares the test split via src.data_prep helpers.
- Runs batched inference on GPU if available, else CPU.
- Computes Accuracy and Macro-F1 for both models.
- Prints a simple before-vs-after comparison with colored output.

Run:
  python -m src.evaluate \
    --finetuned_dir models/domain-llm \
    --base_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --batch_size 16 --max_length 512
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from colorama import Fore, Style, init as colorama_init

from src import data_prep


LABEL_MAP: Dict[str, int] = {"yes": 0, "no": 1, "maybe": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_tokenizer(finetuned_dir: str, base_model: str) -> AutoTokenizer:
    """Prefer tokenizer from finetuned_dir, fallback to base_model.
    Ensures fast tokenizer when available.
    """
    try:
        return AutoTokenizer.from_pretrained(finetuned_dir, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(base_model, use_fast=True)


def prepare_test_dataset(tokenizer: AutoTokenizer, max_length: int, seed: int = 42):
    """Load PubMedQA, preprocess labels/columns, tokenize test split, return tokenized Dataset.
    Also renames label->labels for Trainer/model compatibility and sets PyTorch format.
    """
    _, _, test = data_prep.load_and_split(seed=seed)
    test = data_prep.preprocess_labels(test)
    test = data_prep.select_columns(test)

    def tok(ex):
        return tokenizer(
            ex["question"],
            ex["context"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    test_tok = test.map(tok, batched=True, desc="Tokenizing test")
    if "label" in test_tok.column_names:
        test_tok = test_tok.rename_column("label", "labels")

    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return test_tok


def build_loader(dataset, tokenizer: AutoTokenizer, batch_size: int, use_fp16: bool) -> DataLoader:
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if use_fp16 else None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True)


@torch.no_grad()
def evaluate_model(model: AutoModelForSequenceClassification, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return acc, f1


def main():
    colorama_init(autoreset=True)

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned vs base model on PubMedQA test set")
    parser.add_argument("--finetuned_dir", type=str, default="models/domain-llm", help="Path to fine-tuned model directory")
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Base pretrained model to compare against",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
    print(f"Device: {device} | GPU: {gpu_name}")

    # Tokenizer
    tokenizer = load_tokenizer(args.finetuned_dir, args.base_model)

    # Test dataset + loader
    test_tok = prepare_test_dataset(tokenizer, args.max_length, seed=args.seed)
    loader = build_loader(test_tok, tokenizer, args.batch_size, use_fp16)

    # Load base model (fresh head)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    # Load fine-tuned model
    ft_model = AutoModelForSequenceClassification.from_pretrained(args.finetuned_dir)

    # Evaluate both
    print("Evaluating base model...")
    base_acc, base_f1 = evaluate_model(base_model, loader, device)
    print("Evaluating fine-tuned model...")
    ft_acc, ft_f1 = evaluate_model(ft_model, loader, device)

    # Pretty print results
    print()
    print(Fore.CYAN + "Base Model" + Style.RESET_ALL + f":  accuracy={base_acc:.4f}  f1_macro={base_f1:.4f}")
    print(Fore.GREEN + "Fine-tuned" + Style.RESET_ALL + f":  accuracy={ft_acc:.4f}  f1_macro={ft_f1:.4f}")

    d_acc = ft_acc - base_acc
    d_f1 = ft_f1 - base_f1
    arrow = lambda x: (Fore.GREEN + "↑" if x > 0 else (Fore.RED + "↓" if x < 0 else Fore.YELLOW + "→")) + Style.RESET_ALL
    print()
    print(f"Delta accuracy: {d_acc:+.4f} {arrow(d_acc)}")
    print(f"Delta f1_macro: {d_f1:+.4f} {arrow(d_f1)}")


if __name__ == "__main__":
    main()

