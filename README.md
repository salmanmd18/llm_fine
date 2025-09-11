Fine-tuning a domain-specific LLM for Q&A use cases

Overview
- Goal: Provide a clean, reproducible scaffold for fine-tuning an open-source LLM using the Hugging Face ecosystem and serving it via a minimal FastAPI API.
- Python: 3.10+

Project Structure
- data/ — datasets (raw and processed)
- models/ — saved checkpoints and final models
- src/ — training/evaluation code (Python package)
- src/api/ — API-related source (shared utils, schemas)
- app/ — FastAPI server entrypoint
- notebooks/ — exploratory notebooks
- scripts/ — automation scripts (e.g., setup.sh)

Getting Started
1) Create and activate a virtual environment
   - Unix/macOS: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows PowerShell: `python -m venv .venv; .venv\\Scripts\\Activate.ps1`

2) Install dependencies
   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`
   - Note (PyTorch): Depending on your CUDA/CPU setup, you may prefer installing from the official PyTorch index. Example: `pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121`

3) Run the API server
   - `uvicorn app.main:app --reload`
   - Browse: http://127.0.0.1:8000/docs

4) Typical workflow
- Place raw data in `data/raw/`, generate processed splits into `data/processed/`.
- Implement training logic under `src/` (e.g., `src/train.py`). Save checkpoints to `models/`.
- Update `app/main.py` to load your fine-tuned model for inference.

Notes
- requirements.txt pins versions for reproducibility. Adjust `torch` install per your platform/CUDA.
- `.gitignore` excludes large artifacts; `.gitkeep` files preserve folder structure.
- This scaffold is intentionally minimal—extend training scripts, configs, and evaluation as needed.
Fine-tuning a domain-specific LLM for Q&A use cases

Overview
- Goal: Fine-tune and serve an open-source model for biomedical Q&A using the Hugging Face ecosystem, with a clean, reproducible setup and FastAPI deployment.
- Python: 3.10+

Project Structure
- data/ — datasets (raw and processed)
- models/ — saved checkpoints and final models (e.g., models/domain-llm)
- src/ — training, preprocessing, evaluation code (Python package)
- src/api/ — API-related source (reserved for shared API utilities)
- app/ — FastAPI server entrypoint
- notebooks/ — exploratory notebooks
- scripts/ — automation scripts (e.g., setup.sh)

Quickstart
1) Create and activate a virtual environment
   - Windows PowerShell
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
   - macOS/Linux
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`

2) Install dependencies
   - Always prefer module form on Windows to avoid file locks:
     - `python -m pip install --upgrade pip setuptools wheel`
     - `python -m pip install -r requirements.txt`
   - PyTorch GPU (optional): install the CUDA wheel that matches your drivers
     - CUDA 11.8: `python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio`
     - CUDA 12.1: `python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`

3) Verify CUDA
   - `python - << 'PY'
import torch
print('cuda:', torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')
PY`

Data Preparation (PubMedQA)
- Script: `src/data_prep.py`
- Loads `qiaojin/PubMedQA` (subset `pqa_labeled`), splits 80/10/10, maps labels {yes:0,no:1,maybe:2}, normalizes text, composes causal-LM-friendly text, tokenizes, and saves a small preview.

Examples
- JSON preview: `python -m src.data_prep --model_name gpt2 --output_sample_path data/processed_dataset.json --sample_size 100 --max_length 512`
- CSV preview: `python -m src.data_prep --output_sample_path data/processed_dataset.csv`

Fine-tuning (Sequence Classification)
- Script: `src/train_model.py`
- Default model: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (PubMedBERT). Trains a 3-class head for yes/no/maybe. Uses Trainer with evaluation each epoch and mixed precision on CUDA.

Run
- Standard: `python -m src.train_model --output_dir models/domain-llm --epochs 3 --batch_size 8 --max_length 512`
- If VRAM is tight (e.g., GTX 1650): `--batch_size 4` or `--batch_size 2`
- Smaller model for quick tests: `--model_name distilbert-base-uncased`

Output
- Saves the best/final model and tokenizer under `models/domain-llm` (configurable via `--output_dir`).
- Prints device, GPU name, training loss, and validation metrics (accuracy, macro-F1) each epoch.

Evaluation (Before vs After)
- Script: `src/evaluate.py`
- Compares the base model vs fine-tuned model on the test set and prints accuracy and macro-F1 with colored deltas.

Run
- `python -m src.evaluate --finetuned_dir models/domain-llm --base_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --batch_size 16 --max_length 512`

API (FastAPI)
- Entrypoint: `app/main.py`
- Loads the fine-tuned classifier at startup and exposes:
  - `GET /health` — health and device info
  - `POST /predict` — accepts `{ "question": "...", "context": "..." }` and returns `{ "answer": "yes|no|maybe" }`

Run (local)
- `uvicorn app.main:app --reload`
- Docs: http://127.0.0.1:8000/docs
- Example request (curl):
  - `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"question\":\"Is aspirin effective for migraine?\",\"context\":\"Randomized trials suggest benefit in acute treatment.\"}"`

Postman/Browser
- Open Swagger UI at http://127.0.0.1:8000/docs and use “Try it out”.
- In Postman, send POST `http://127.0.0.1:8000/predict` with JSON body and `Content-Type: application/json`.

Docker
- Dockerfile provided. It copies the repo and installs from `requirements.txt`, then runs Uvicorn.

Build
- `docker build -t llm-finetune-api .`

Run (CPU)
- `docker run --rm -p 8000:8000 llm-finetune-api`
- Health: http://127.0.0.1:8000/health

GPU container (optional)
- Requires a CUDA base image and `--gpus all`. Ask if you want a CUDA Dockerfile variant.

Environment Variables
- `MODEL_DIR` — path to fine-tuned model (default: `models/domain-llm`)
- `BASE_MODEL` — fallback base model for tokenizer/weights
- `MAX_LENGTH` — max sequence length for API tokenization (default: 512)
- `TOKENIZERS_PARALLELISM=false` is set to reduce warnings

Troubleshooting
- Windows pip upgrade error (pip.exe locked): use `python -m pip ...` instead of `pip ...`.
- Missing `importlib_metadata`: run `python -m pip install importlib-metadata` (also pinned in `requirements.txt`).
- PyTorch GPU wheels: install from the PyTorch index (`cu118`/`cu121`) for CUDA support.
- Hugging Face cache symlink warning on Windows: enable Developer Mode or run as admin to allow symlinks, or ignore (cache falls back to copies).
- OOM on GPU: lower `--batch_size`, reduce `--max_length`, or use a smaller base model.

Notes
- Classification setup uses accuracy and macro-F1. If you need generative fine-tuning/evaluation (BLEU/ROUGE on `long_answer`), we can add a causal-LM training/eval path alongside the classifier.
