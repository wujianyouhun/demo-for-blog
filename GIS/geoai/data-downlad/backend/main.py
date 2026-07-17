from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;GEOAI_ROOT=ROOT.parent;sys.path.insert(0,str(GEOAI_ROOT))
import requests
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from web_common import TaskRegistry,install_common_routes
DATA_DIR=ROOT/"data";OUTPUT_DIR=DATA_DIR/"downloads";tasks=TaskRegistry(max_workers=2)
app=FastAPI(title="GeoAI Multi-source Download",version="1.0.0");app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
install_common_routes(app,tasks,"data-downlad",DATA_DIR,OUTPUT_DIR,{"ports":{"backend":8020,"frontend":5180},"sources":["planetary","naip","overture"]})
class SearchRequest(BaseModel):
    source:str="planetary";bbox:str="108.85,34.15,109.10,34.35";date_start:str="2024-01-01";date_end:str="2024-12-31";max_items:int=Field(default=10,ge=1,le=100)
class DownloadRequest(BaseModel):url:str;filename:str="asset.tif"
def parse_bbox(value:str):
    result=[float(item.strip()) for item in value.split(',')]
    if len(result)!=4 or result[0]>=result[2] or result[1]>=result[3]:raise ValueError("BBox 必须为 minx,miny,maxx,maxy")
    return result
@app.post('/api/search')
def search(request:SearchRequest):
    bounds=parse_bbox(request.bbox)
    if request.source=='overture':return {"source":"overture","bbox":bounds,"items":[{"id":"overture-buildings","assets":{"data":{"type":"overture-stream"}}}],"message":"Overture 数据按 bbox 流式读取，提交下载任务时执行。"}
    try:
        from pystac_client import Client
        collection='naip' if request.source=='naip' else 'sentinel-2-l2a'
        items=list(Client.open('https://planetarycomputer.microsoft.com/api/stac/v1').search(collections=[collection],bbox=bounds,datetime=f'{request.date_start}/{request.date_end}',max_items=request.max_items).items())
        return {"source":request.source,"collection":collection,"items":[{"id":item.id,"datetime":item.datetime.isoformat() if item.datetime else None,"bbox":item.bbox,"assets":{key:{"href":asset.href,"type":asset.media_type} for key,asset in item.assets.items()}} for item in items]}
    except Exception as exc:raise HTTPException(502,f"STAC 检索失败: {exc}")
def download_job(request:DownloadRequest,*,update,cancel_event,**_):
    if not request.url.startswith(('http://','https://')):raise ValueError('仅允许 HTTP(S) URL')
    target=OUTPUT_DIR/Path(request.filename).name;target.parent.mkdir(parents=True,exist_ok=True)
    with requests.get(request.url,stream=True,timeout=60) as response:
        response.raise_for_status();total=int(response.headers.get('content-length',0));written=0
        with target.open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                if cancel_event.is_set():break
                if chunk:stream.write(chunk);written+=len(chunk);update(progress=int(written/total*100) if total else 0,stage='download',message=f'已下载 {written/1024/1024:.1f} MB')
    return {"path":str(target),"size":target.stat().st_size}
@app.post('/api/download-task')
def download_task(request:DownloadRequest):return tasks.submit('download',download_job,request)
