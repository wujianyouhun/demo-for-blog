from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="AtomicGIS")

app.include_router(router)

@app.get("/")
def root():
    return {"message": "AtomicGIS Runtime Running"}