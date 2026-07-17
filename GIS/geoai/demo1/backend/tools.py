from __future__ import annotations
import math,re
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import Point,box

def ensure_demo_boundary(path: Path) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        gpd.GeoDataFrame({'name':['演示风电选址区']},geometry=[box(108.75,34.05,109.20,34.40)],crs='EPSG:4326').to_file(path,driver='GeoJSON')
    return path

def requested_count(query: str, default: int=30) -> int:
    match=re.search(r'(\d+)\s*(?:个|处|座|点)',query)
    return min(500,max(1,int(match.group(1)))) if match else default

def find_wind_sites(boundary_path: str|Path, count: int=30) -> gpd.GeoDataFrame:
    boundary=gpd.read_file(boundary_path)
    if boundary.empty:raise ValueError('边界数据为空')
    area=boundary.to_crs(boundary.estimate_utm_crs())
    geom=area.geometry.union_all()
    minx,miny,maxx,maxy=geom.bounds
    side=max(4,math.ceil(math.sqrt(count*2.2)))
    points=[]
    for y in np.linspace(miny,maxy,side+2)[1:-1]:
        for x in np.linspace(minx,maxx,side+2)[1:-1]:
            point=Point(float(x),float(y))
            if geom.contains(point):
                score=0.55+0.4*((math.sin(x/1700)+math.cos(y/1900)+2)/4)
                points.append((point,round(score,4)))
    points=sorted(points,key=lambda item:item[1],reverse=True)[:count]
    result=gpd.GeoDataFrame({'site_id':range(1,len(points)+1),'suitability':[v for _,v in points],'source':['local-deterministic']*len(points)},geometry=[p for p,_ in points],crs=area.crs)
    return result.to_crs(boundary.crs)
