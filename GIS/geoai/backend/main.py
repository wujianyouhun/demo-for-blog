from fastapi import FastAPI
from pydantic import BaseModel
from agent import create_agent

app = FastAPI()
agent = create_agent()

class Query(BaseModel):
    text: str

@app.post("/query")
def run_query(q: Query):
    result = agent.invoke({"input": q.text})
    return {"result": result}
