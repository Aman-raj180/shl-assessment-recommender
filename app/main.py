from fastapi import FastAPI, HTTPException
from .schemas import ChatRequest, ChatResponse, HealthResponse
from .agent import handle_chat

app = FastAPI(title="SHL Assessment Recommender")


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = handle_chat(request.messages)
        return result
    except RuntimeError as e:
        # e.g. missing GEMINI_API_KEY -- surfaces as 500 with a clear message
        raise HTTPException(status_code=500, detail=str(e))
