from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="LLM Fine-tune API", version="0.1.0")


class Query(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Query) -> dict:
    # TODO: Load your fine-tuned model from ./models and run inference.
    # This is a placeholder implementation.
    return {"answer": f"Not implemented. Received: {payload.question}"}


if __name__ == "__main__":
    # Allows: python app/main.py (useful for quick local dev)
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
