from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
GEOAI_ROOT=ROOT.parent/"geoai"
sys.path[:0]=[str(ROOT),str(GEOAI_ROOT)]

from fastapi import FastAPI,File,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field

from web_common import TaskRegistry,list_files
from train import DEFAULT_IMAGES,DEFAULT_MASKS,MODEL_ROOT,train
from predict import predict

DATA_DIR=ROOT/"data"; OUTPUT_DIR=ROOT/"outputs"; UPLOAD_DIR=DATA_DIR/"uploads"
for directory in (OUTPUT_DIR,UPLOAD_DIR,MODEL_ROOT): directory.mkdir(parents=True,exist_ok=True)
tasks=TaskRegistry(max_workers=1)
app=FastAPI(title="GeoAI Segmentation Baselines",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])


class TrainRequest(BaseModel):
    images:str=str(DEFAULT_IMAGES); masks:str=str(DEFAULT_MASKS); model:str="unetpp"
    epochs:int=Field(default=20,ge=1,le=200); batch_size:int=Field(default=2,ge=1,le=32)
    size:int=Field(default=256,ge=64,le=1024); base_channels:int=Field(default=32,ge=8,le=64)
    lr:float=3e-4; val_ratio:float=.25; seed:int=42; output:str|None=None


class PredictRequest(BaseModel):
    input_path:str; checkpoint:str


def train_job(request:TrainRequest,*,update,cancel_event,**_):
    namespace=argparse.Namespace(**request.model_dump())
    return train(namespace,lambda p,m:update(progress=p,stage="train",message=f"epoch {m['epoch']} IoU={m['iou']:.4f}",metrics=m),cancel_event)


def predict_job(request:PredictRequest,*,task_id,update,**_):
    update(progress=10,stage="predict",message="加载模型与影像")
    result=predict(Path(request.input_path).expanduser().resolve(),Path(request.checkpoint).expanduser().resolve(),OUTPUT_DIR/f"task_{task_id}")
    update(progress=95,stage="vectorize",message="保存栅格和矢量成果")
    return result


@app.get("/api/health")
def health():
    import torch
    return {"status":"ok","cuda":torch.cuda.is_available(),"models":["unetpp","deeplabv3plus"]}


@app.get("/api/config")
def config(): return {"default_images":str(DEFAULT_IMAGES),"default_masks":str(DEFAULT_MASKS),"ports":{"backend":8027,"frontend":5187}}


@app.get("/api/models")
def models(): return {"architectures":["unetpp","deeplabv3plus"],"checkpoints":list_files(MODEL_ROOT,{".pth",".pt"})}


@app.post("/api/uploads")
async def upload(file:UploadFile=File(...)):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in {".tif",".tiff",".png",".jpg",".jpeg"}: raise HTTPException(400,"不支持的影像格式")
    target=UPLOAD_DIR/Path(file.filename or "upload.tif").name; target.write_bytes(await file.read())
    return {"path":str(target),"size":target.stat().st_size}


@app.post("/api/train")
def start_train(request:TrainRequest):
    if request.model not in {"unetpp","deeplabv3plus"}: raise HTTPException(400,"未知模型")
    return tasks.submit("train",train_job,request)


@app.post("/api/predict")
def start_predict(request:PredictRequest): return tasks.submit("predict",predict_job,request)


@app.get("/api/tasks/{task_id}")
def task(task_id:str):
    try:return tasks.public(task_id)
    except KeyError:raise HTTPException(404,"任务不存在")


@app.post("/api/tasks/{task_id}/cancel")
def cancel(task_id:str):
    try:return tasks.cancel(task_id)
    except KeyError:raise HTTPException(404,"任务不存在")


@app.get("/api/download")
def download(path:str):
    candidate=Path(path).resolve(); roots=[OUTPUT_DIR.resolve(),MODEL_ROOT.resolve()]
    if not candidate.is_file() or not any(root in candidate.parents for root in roots): raise HTTPException(404,"文件不存在")
    return FileResponse(candidate,filename=candidate.name)
