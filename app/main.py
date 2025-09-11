import os
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# FastAPI application
app = FastAPI(title="LLM Fine-tune API", version="0.2.0")


# Environment/setup tweaks for faster, quieter startup
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


# Paths and defaults
MODEL_DIR = os.environ.get("MODEL_DIR", "models/domain-llm")
BASE_MODEL = os.environ.get(
    "BASE_MODEL",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
)
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))


# Load tokenizer/model once at startup for low-latency requests
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
except Exception:
    # Fallback to a base model with 3 labels if fine-tuned weights are missing
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label={0: "yes", 1: "no", 2: "maybe"},
        label2id={"yes": 0, "no": 1, "maybe": 2},
    )

model.eval()
model.to(device)


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    context: Optional[str] = Field("", description="Optional supporting context/abstract")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": device,
        "model_dir": MODEL_DIR,
    }


@app.post("/predict")
def predict(req: QARequest) -> dict:
    """Classify the question+context into yes/no/maybe using the fine-tuned model.

    This uses torch.inference_mode() for speed and low memory.
    """
    question = (req.question or "").strip()
    context = (req.context or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Field 'question' is required and cannot be empty.")

    # Tokenize input pair similar to training
    enc = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        logits = model(**enc).logits
        pred_id = int(torch.argmax(logits, dim=-1).item())

    # Map index to label using the model config when available
    id2label = getattr(model.config, "id2label", None) or {0: "yes", 1: "no", 2: "maybe"}
    answer = id2label.get(pred_id, str(pred_id))
    return {"answer": answer}


if __name__ == "__main__":
    # Allows: python app/main.py (useful for quick local dev)
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
