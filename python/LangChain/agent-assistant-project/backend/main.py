
from fastapi import FastAPI
from agent.agent import run_agent

app = FastAPI()

@app.post("/chat")
def chat(query: str):
    return {"answer": run_agent(query)}
