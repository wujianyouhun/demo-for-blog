from __future__ import annotations
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
RASTER={'.tif','.tiff','.img','.vrt'};VECTOR={'.geojson','.json','.gpkg','.shp','.parquet','.zip'}
def raster_info(path:Path)->dict:
    with rasterio.open(path) as source:
        sample=source.read(out_shape=(source.count,min(source.height,512),min(source.width,512)),masked=True)
        bands=[]
        for index in range(source.count):
            values=sample[index].compressed();bands.append({"band":index+1,"min":float(values.min()) if values.size else None,"max":float(values.max()) if values.size else None,"mean":float(values.mean()) if values.size else None,"std":float(values.std()) if values.size else None})
        return {"kind":"raster","path":str(path),"driver":source.driver,"width":source.width,"height":source.height,"count":source.count,"dtypes":list(source.dtypes),"nodata":source.nodata,"crs":str(source.crs),"transform":list(source.transform)[:6],"bounds":list(source.bounds),"resolution":list(source.res),"band_stats":bands}
def vector_info(path:Path)->dict:
    source=f'zip://{path}' if path.suffix.lower()=='.zip' else path
    frame=gpd.read_parquet(source) if path.suffix.lower()=='.parquet' else gpd.read_file(source)
    numeric={}
    for column in frame.select_dtypes(include='number').columns:
        series=frame[column];numeric[column]={"min":float(series.min()) if len(series) else None,"max":float(series.max()) if len(series) else None,"mean":float(series.mean()) if len(series) else None,"nulls":int(series.isna().sum())}
    return {"kind":"vector","path":str(path),"features":len(frame),"crs":str(frame.crs),"bounds":frame.total_bounds.tolist(),"geometry_types":frame.geom_type.value_counts().to_dict(),"columns":[str(column) for column in frame.columns if column!='geometry'],"numeric_stats":numeric}
def inspect(path:Path)->dict:
    suffix=path.suffix.lower()
    if suffix in RASTER:return raster_info(path)
    if suffix in VECTOR:return vector_info(path)
    raise ValueError(f'不支持的格式: {suffix}')
def preview(path:Path,output:Path)->Path:
    output.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower() in RASTER:
        with rasterio.open(path) as source:
            indexes=list(range(1,min(source.count,3)+1));array=source.read(indexes,out_shape=(len(indexes),512,512)).astype(np.float32)
        if len(array)==1:array=np.repeat(array,3,axis=0)
        array=np.moveaxis(array[:3],0,-1);valid=np.isfinite(array);low,high=np.percentile(array[valid],[2,98]);image=np.clip((array-low)/max(high-low,1e-6)*255,0,255).astype(np.uint8);Image.fromarray(image).save(output)
    else:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        source=f'zip://{path}' if path.suffix.lower()=='.zip' else path;frame=gpd.read_file(source);figure,axis=plt.subplots(figsize=(7,6));frame.plot(ax=axis,color='#4aa58b',edgecolor='#173b4c');axis.set_axis_off();figure.tight_layout();figure.savefig(output,dpi=140);plt.close(figure)
    return output
