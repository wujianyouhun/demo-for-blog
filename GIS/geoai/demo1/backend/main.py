from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/query")
def run_query(q: Query):
    from agent import create_agent
    agent = create_agent()
    result = agent.invoke({"input": q.text})
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
