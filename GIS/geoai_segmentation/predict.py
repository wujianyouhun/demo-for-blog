import torch
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from models.unet_plus_plus import UNetPlusPlus

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNetPlusPlus().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# 读取图像
img = cv2.imread('test.png')
img = cv2.resize(img, (512,512))
input = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()/255.0

# 推理
with torch.no_grad():
    pred = model(input.to(device))[0][0].cpu().numpy()
mask = (pred > 0.5).astype(np.uint8)*255

# 轮廓转矢量
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
polygons = [Polygon(c[:,0,:]) for c in contours if cv2.contourArea(c) > 100]

# 保存 shp
gdf = gpd.GeoDataFrame({'geometry': polygons})
gdf.to_file('result.shp', encoding='utf-8')
print("已生成：result.shp（可直接在 ArcGIS/QGIS 打开）")