from __future__ import annotations
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;GEOAI_ROOT=ROOT.parent;sys.path[:0]=[str(ROOT),str(GEOAI_ROOT)]
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from web_common import TaskRegistry,install_common_routes,resolve_user_path,list_files
DATA_DIR=ROOT/'data';OUTPUT_DIR=ROOT/'outputs';tasks=TaskRegistry(max_workers=1)
app=FastAPI(title='GeoAI Tiled Building Extraction',version='1.0.0');app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
install_common_routes(app,tasks,'building-extraction',DATA_DIR,OUTPUT_DIR,{'ports':{'backend':8025,'frontend':5185},'models_dir':str(GEOAI_ROOT/'models')})
class ExtractRequest(BaseModel):
 image_path:str;method:str='geoai';tile_size:int=Field(2048,ge=256,le=4096);overlap:int=Field(256,ge=0,le=1024)
def command_job(command,*,update,cancel_event,stage):
 process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace');lines=[]
 while True:
  line=process.stdout.readline() if process.stdout else ''
  if line:lines.append(line.rstrip());update(stage=stage,message=line.rstrip()[-400:])
  if cancel_event.is_set():process.terminate();raise RuntimeError('任务已取消')
  if process.poll() is not None:break
 if process.returncode:raise RuntimeError('\n'.join(lines[-30:]))
 return lines
def check_job(**kwargs):
 lines=command_job([sys.executable,'extract_buildings.py','--check-env'],stage='check',**kwargs)
 return {'ok':True,'log':lines[-20:],'models':list_files(GEOAI_ROOT/'models')}
def extract_job(request:ExtractRequest,**kwargs):
 image=resolve_user_path(request.image_path,DATA_DIR,{'.tif','.tiff'});models='geoai' if request.method=='geoai' else 'sam_h'
 command=[sys.executable,'extract_buildings.py','--source-tif',str(image),'--models',models,'--tile-size',str(request.tile_size),'--overlap',str(request.overlap),'--device','auto','--tile-full-image','--no-download-models']
 command_job(command,stage='extract',**kwargs)
 return {'source':str(image),'files':list_files(OUTPUT_DIR)}
@app.post('/api/check')
def check_api():return tasks.submit('check',check_job)
@app.post('/api/extract')
def extract_api(request:ExtractRequest):return tasks.submit('extract',extract_job,request)
