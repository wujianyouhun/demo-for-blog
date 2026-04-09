import geopandas as gpd
from shapely.geometry import Point

def find_wind_sites(boundary_path: str):
    gdf = gpd.read_file(boundary_path)
    minx, miny, maxx, maxy = gdf.total_bounds

    points = []
    for i in range(30):
        x = minx + (maxx - minx) * i / 30
        y = miny + (maxy - miny) * i / 30
        points.append(Point(x, y))

    result = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)
    return result.to_json()
