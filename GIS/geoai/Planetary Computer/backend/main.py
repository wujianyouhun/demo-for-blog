from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;GEOAI_ROOT=ROOT.parent;sys.path.insert(0,str(GEOAI_ROOT))
import requests
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from web_common import TaskRegistry,install_common_routes
DATA_DIR=ROOT/'data';OUTPUT_DIR=DATA_DIR/'downloads';tasks=TaskRegistry(max_workers=2)
app=FastAPI(title='Planetary Computer Assets',version='1.0.0');app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
install_common_routes(app,tasks,'planetary-computer',DATA_DIR,OUTPUT_DIR,{"ports":{"backend":8021,"frontend":5181}})
class ItemRequest(BaseModel):item_url:str
class DownloadRequest(BaseModel):item_url:str;assets:str='B04,B08'
def load_item(url:str):
    if not url.startswith('https://planetarycomputer.microsoft.com/'):raise ValueError('仅允许 Planetary Computer Item URL')
    import pystac,planetary_computer
    return planetary_computer.sign(pystac.Item.from_file(url))
@app.post('/api/inspect')
def inspect(request:ItemRequest):
    try:item=load_item(request.item_url);return {"id":item.id,"collection":item.collection_id,"datetime":item.datetime.isoformat() if item.datetime else None,"bbox":item.bbox,"properties":item.properties,"assets":{key:{"href":asset.href,"title":asset.title,"type":asset.media_type,"roles":asset.roles} for key,asset in item.assets.items()}}
    except Exception as exc:raise HTTPException(502,f'读取 Item 失败: {exc}')
def download_job(request:DownloadRequest,*,update,cancel_event,**_):
    item=load_item(request.item_url);selected=[key.strip() for key in request.assets.split(',') if key.strip()];results={}
    for index,key in enumerate(selected):
        if cancel_event.is_set():break
        if key not in item.assets:raise ValueError(f'资产不存在: {key}')
        url=item.assets[key].href;extension=Path(url.split('?')[0]).suffix or '.tif';target=OUTPUT_DIR/f'{item.id}_{key}{extension}'
        with requests.get(url,stream=True,timeout=90) as response:
            response.raise_for_status();total=int(response.headers.get('content-length',0));written=0
            with target.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    if cancel_event.is_set():break
                    if chunk:stream.write(chunk);written+=len(chunk);local=written/total if total else 0;update(progress=int((index+local)/len(selected)*100),stage=f'download:{key}',message=f'{key} {written/1024/1024:.1f} MB')
        results[key]=str(target)
    return {"item":item.id,"assets":results}
@app.post('/api/download-task')
def download(request:DownloadRequest):return tasks.submit('download',download_job,request)
