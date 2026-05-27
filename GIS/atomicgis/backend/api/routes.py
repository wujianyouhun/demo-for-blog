from fastapi import APIRouter
from core.executor import execute_workflow

router = APIRouter()

@router.post("/workflow/run")
def run_workflow(payload: dict):
    return execute_workflow(payload)