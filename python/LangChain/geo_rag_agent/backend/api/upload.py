
from fastapi import APIRouter, UploadFile, File
from core.ingest import ingest_file

router = APIRouter()

@router.post("/")
async def upload(file: UploadFile = File(...), user_id: str = "default"):
    await ingest_file(file, user_id)
    return {"msg": "uploaded"}
