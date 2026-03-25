
from fastapi import FastAPI
from api.chat import router as chat_router
from api.upload import router as upload_router

app = FastAPI()

app.include_router(chat_router, prefix="/chat")
app.include_router(upload_router, prefix="/upload")
