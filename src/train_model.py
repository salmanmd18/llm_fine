"""
Fine-tune a biomedical model (PubMedBERT by default) on PubMedQA
for 3-way classification (yes/no/maybe) using Hugging Face Trainer.

Best practices:
- Uses `AutoTokenizer` and `AutoModelForSequenceClassification` with num_labels=3.
- Tokenizes (question, context) pairs with truncation and padding.
- Evaluates each epoch and saves best checkpoint; prints metrics.
- Compatible with single-GPU or Colab; Trainer leverages Accelerate under the hood.

Run:
  python -m src.train_model \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --output_dir models/domain-llm \
    --epochs 3 \
    --batch_size 8 \
    --max_length 512
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import datasets as ds
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Reuse preprocessing helpers where possible
from src import data_prep


# Consistent label mapping with preprocessing
LABEL_MAP: Dict[str, int] = {"yes": 0, "no": 1, "maybe": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


def build_tokenizer(name: str) -> AutoTokenizer:
    """Load tokenizer; for encoder models (e.g., BERT) pad token exists."""
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    return tok


def tokenize_pairs(example, tokenizer: AutoTokenizer, max_length: int):
    """Tokenize question-context pair for sequence classification."""
    return tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def compute_metrics_builder():
    """Compute accuracy and macro-F1 using scikit-learn (no Hub downloads)."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PubMedBERT on PubMedQA (3-way classification)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Base model checkpoint",
    )
    parser.add_argument("--output_dir", type=str, default="models/domain-llm", help="Where to save checkpoints and final model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load and preprocess dataset (reuse data_prep helpers)
    train, val, test = data_prep.load_and_split(seed=args.seed)
    train = data_prep.preprocess_labels(train)
    val = data_prep.preprocess_labels(val)
    test = data_prep.preprocess_labels(test)

    train = data_prep.select_columns(train)
    val = data_prep.select_columns(val)
    test = data_prep.select_columns(test)

    # 2) Tokenizer and tokenization
    tokenizer = build_tokenizer(args.model_name)
    tok_fn = lambda ex: tokenize_pairs(ex, tokenizer, args.max_length)
    train_tok = train.map(tok_fn, batched=True, desc="Tokenizing train")
    val_tok = val.map(tok_fn, batched=True, desc="Tokenizing val")
    test_tok = test.map(tok_fn, batched=True, desc="Tokenizing test")

    # Ensure the label column is named 'labels' for Trainer compatibility
    if "label" in train_tok.column_names:
        train_tok = train_tok.rename_column("label", "labels")
    if "label" in val_tok.column_names:
        val_tok = val_tok.rename_column("label", "labels")
    if "label" in test_tok.column_names:
        test_tok = test_tok.rename_column("label", "labels")

    # 3) Model for 3-class classification with label mapping
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
    )

    # 4) Training configuration (single-GPU friendly); Trainer uses Accelerate
    use_fp16 = torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
    print(f"Device: {device} | GPU: {gpu_name} | fp16: {use_fp16}")
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if use_fp16 else None)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=args.seed,
        fp16=use_fp16,
        dataloader_pin_memory=True,
        report_to=["none"],  # set to ["tensorboard"] to log
        save_total_limit=2,
    )

    # 5) Metrics and Trainer
    compute_metrics = compute_metrics_builder()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # 6) Train + evaluate each epoch; Trainer prints loss/metrics
    trainer.train()

    # 7) Evaluate on validation and test sets for completeness
    val_metrics = trainer.evaluate(eval_dataset=val_tok)
    print("Validation:", val_metrics)
    test_metrics = trainer.evaluate(eval_dataset=test_tok)
    print("Test:", test_metrics)

    # 8) Save final/best model to output_dir
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
