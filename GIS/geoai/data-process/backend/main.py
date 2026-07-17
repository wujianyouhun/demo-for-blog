from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;GEOAI_ROOT=ROOT.parent;sys.path[:0]=[str(ROOT),str(GEOAI_ROOT)]
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from web_common import TaskRegistry,install_common_routes
from core import inspect,preview
DATA_DIR=ROOT/'data';OUTPUT_DIR=ROOT/'outputs';tasks=TaskRegistry(max_workers=2)
app=FastAPI(title='GeoAI Data Inspector',version='1.0.0');app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
install_common_routes(app,tasks,'data-process',DATA_DIR,OUTPUT_DIR,{"ports":{"backend":8022,"frontend":5182}})
class InspectRequest(BaseModel):path:str;preview:bool=True
def inspect_job(request:InspectRequest,*,task_id,update,**_):
    path=Path(request.path).expanduser().resolve();update(progress=20,stage='inspect',message='读取元数据');result=inspect(path)
    if request.preview:result['preview']=str(preview(path,OUTPUT_DIR/f'{task_id}_preview.png'))
    report=OUTPUT_DIR/f'{task_id}_metadata.json';report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');result['report']=str(report);update(progress=95,stage='report',message='保存报告');return result
@app.post('/api/inspect')
def inspect_api(request:InspectRequest):return tasks.submit('inspect',inspect_job,request)
@app.post('/api/compare')
def compare(request:InspectRequest):
    path=Path(request.path).expanduser().resolve();raw=inspect(path);geoai_result=None
    try:
        import geoai
        value=geoai.get_raster_info(str(path)) if raw['kind']=='raster' else geoai.get_vector_info(str(path));geoai_result=json.loads(json.dumps(value,default=str))
    except Exception as exc:geoai_result={"available":False,"error":str(exc)}
    return {"raw":raw,"geoai":geoai_result}
