
from fastapi import FastAPI
from pydantic import BaseModel
from agent.agent import run_agent

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    result = run_agent(q.question)
    return {"answer": result}
