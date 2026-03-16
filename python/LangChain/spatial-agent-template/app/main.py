from fastapi import FastAPI
from agent import agent

app = FastAPI()

@app.get("/query")
def query(q:str):

    result = agent.run(q)

    return {"result":result}
