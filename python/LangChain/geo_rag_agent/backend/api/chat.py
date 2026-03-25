
from fastapi import APIRouter
from core.rag_pipeline import ask

router = APIRouter()

@router.post("/")
def chat(query: str, user_id: str = "default"):
    return {"answer": ask(query, user_id)}
