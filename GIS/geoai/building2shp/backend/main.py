from __future__ import annotations
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;GEOAI_ROOT=ROOT.parent;sys.path[:0]=[str(ROOT),str(GEOAI_ROOT)]
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from web_common import TaskRegistry,install_common_routes,resolve_user_path,list_files
DATA_DIR=ROOT/'data';OUTPUT_DIR=ROOT/'outputs';MODEL_DIR=GEOAI_ROOT/'models'/'building2shp'/'trained';tasks=TaskRegistry(max_workers=1)
app=FastAPI(title='Building2SHP',version='1.0.0');app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
install_common_routes(app,tasks,'building2shp',DATA_DIR,OUTPUT_DIR,{'ports':{'backend':8024,'frontend':5184},'model_dir':str(MODEL_DIR)})
class TrainRequest(BaseModel):samples:int=Field(200,ge=20,le=5000);epochs:int=Field(20,ge=1,le=200)
class PredictRequest(BaseModel):image_path:str;model_path:str='';tile_size:int=256;overlap:int=32
def run_command(command,*,update,cancel_event,stage):
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    lines=[]
    while True:
        line=process.stdout.readline() if process.stdout else ''
        if line: lines.append(line.rstrip());update(stage=stage,message=line.rstrip()[-300:])
        if cancel_event.is_set():process.terminate();raise RuntimeError('任务已取消')
        if process.poll() is not None:break
    if process.returncode:raise RuntimeError('\n'.join(lines[-20:]))
    return lines
def train_job(request:TrainRequest,**kwargs):
    run_command([sys.executable,'train_model.py','--samples',str(request.samples),'--epochs',str(request.epochs)],stage='train',**kwargs)
    return {'checkpoint':str(MODEL_DIR/'best_model.pth'),'files':list_files(MODEL_DIR)}
def predict_job(request:PredictRequest,**kwargs):
    image=resolve_user_path(request.image_path,DATA_DIR,{'.tif','.tiff'})
    model=Path(request.model_path).expanduser().resolve() if request.model_path else MODEL_DIR/'best_model.pth'
    command=[sys.executable,'building_extractor.py','--image',str(image),'--model',str(model),'--tile-size',str(request.tile_size),'--overlap',str(request.overlap),'--output-dir',str(OUTPUT_DIR)]
    run_command(command,stage='predict',**kwargs);return {'files':list_files(OUTPUT_DIR)}
@app.post('/api/train')
def train_api(request:TrainRequest):return tasks.submit('train',train_job,request)
@app.post('/api/predict')
def predict_api(request:PredictRequest):return tasks.submit('predict',predict_job,request)
