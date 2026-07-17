from __future__ import annotations
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;GEOAI_ROOT=ROOT.parent;sys.path[:0]=[str(ROOT),str(GEOAI_ROOT)]
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from web_common import TaskRegistry,install_common_routes,resolve_user_path
from backend.agent import run_agent
from backend.tools import ensure_demo_boundary
DATA_DIR=ROOT/'data';OUTPUT_DIR=ROOT/'outputs';DEMO=ensure_demo_boundary(DATA_DIR/'demo_boundary.geojson');tasks=TaskRegistry(max_workers=2)
app=FastAPI(title='GeoAI Spatial Agent',version='1.0.0');app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
install_common_routes(app,tasks,'demo1',DATA_DIR,OUTPUT_DIR,{'ports':{'backend':8026,'frontend':5186},'modes':{'local':True,'openai':bool(os.getenv('OPENAI_API_KEY'))},'demo_boundary':str(DEMO)})
class Query(BaseModel):query:str='请在演示边界内生成 30 个风电站候选点';boundary_path:str='';mode:str='local'
def query_job(request:Query,*,task_id,update,**_):
    boundary=resolve_user_path(request.boundary_path,DATA_DIR,{'.geojson','.json','.gpkg','.shp','.zip'}) if request.boundary_path else DEMO
    update(progress=25,stage='parse',message='解析自然语言和边界')
    result,explanation=run_agent(request.query,str(boundary),request.mode)
    update(progress=75,stage='spatial-analysis',message=f'已生成 {len(result)} 个候选点')
    path=OUTPUT_DIR/f'{task_id}_wind_sites.geojson';result.to_file(path,driver='GeoJSON')
    return {'message':explanation,'count':len(result),'geojson':result.__geo_interface__,'files':[str(path)]}
@app.post('/api/query-task')
def query_task(request:Query):return tasks.submit('agent',query_job,request)
@app.post('/api/query')
def query_sync(request:Query):
    result,explanation=run_agent(request.query,str(resolve_user_path(request.boundary_path,DATA_DIR) if request.boundary_path else DEMO),request.mode)
    return {'message':explanation,'geojson':result.__geo_interface__}
