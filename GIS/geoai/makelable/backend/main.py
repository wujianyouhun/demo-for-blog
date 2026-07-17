from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;GEOAI_ROOT=ROOT.parent;sys.path[:0]=[str(ROOT),str(GEOAI_ROOT)]
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from web_common import TaskRegistry,install_common_routes,list_files
from dataprocess import align_data
DATA_DIR=ROOT/'data';OUTPUT_DIR=DATA_DIR/'web_output';tasks=TaskRegistry(max_workers=1)
app=FastAPI(title='GeoAI Sample Maker',version='1.0.0');app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
install_common_routes(app,tasks,'makelable',DATA_DIR,OUTPUT_DIR,{"ports":{"backend":8023,"frontend":5183},"stac_api":"http://localhost:8010"})
class AlignRequest(BaseModel):raster_path:str;vector_path:str
class SampleRequest(AlignRequest):tile_size:int=Field(default=512,ge=64,le=2048);stride:int=Field(default=256,ge=32,le=2048);augmentation_count:int=Field(default=2,ge=0,le=10)
def align_job(request:AlignRequest,*,task_id,update,**_):
    update(progress=10,stage='align',message='检查 CRS 与范围');result=align_data(Path(request.raster_path).resolve(),Path(request.vector_path).resolve(),OUTPUT_DIR/f'align_{task_id}');update(progress=95,stage='preview',message='生成叠加预览');return {key:str(value) for key,value in result.items()}
@app.post('/api/align')
def align(request:AlignRequest):return tasks.submit('align',align_job,request)
def sample_job(request:SampleRequest,*,task_id,update,**_):
    import geoai
    output=OUTPUT_DIR/f'samples_{task_id}';output.mkdir(parents=True,exist_ok=True);update(progress=10,stage='tiles',message='切分影像和标签')
    geoai.export_geotiff_tiles(in_raster=str(Path(request.raster_path).resolve()),out_folder=str(output),in_class_data=str(Path(request.vector_path).resolve()),tile_size=request.tile_size,stride=request.stride,skip_empty_tiles=True,buffer_radius=0,all_touched=True,create_overview=True,apply_augmentation=request.augmentation_count>0,augmentation_count=request.augmentation_count,quiet=True)
    images=list_files(output/'images');masks=list_files(output/'labels');manifest={"tile_size":request.tile_size,"stride":request.stride,"images":images,"masks":masks,"paired":len({Path(x['name']).stem for x in images}&{Path(x['name']).stem for x in masks})};manifest_path=output/'manifest.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');update(progress=95,stage='manifest',message='写入样本清单');return {"output":str(output),"manifest":str(manifest_path),"images":len(images),"masks":len(masks),"overview":str(output/'overview.png') if (output/'overview.png').exists() else None}
@app.post('/api/samples')
def samples(request:SampleRequest):return tasks.submit('samples',sample_job,request)
