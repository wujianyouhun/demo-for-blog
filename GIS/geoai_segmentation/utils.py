import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from shapely.geometry import Polygon
import geopandas as gpd

# 数据集加载
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        self.mask_paths = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir))]
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.resize(img, (self.size, self.size))
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        mask = cv2.imread(self.mask_paths[idx], 0)
        mask = cv2.resize(mask, (self.size, self.size))
        mask = mask / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

# 预测后处理：转二值图
def post_process(pred, threshold=0.5):
    pred = (pred > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    return pred

# 轮廓转SHP矢量
def mask_to_shp(mask, save_path="result.shp", min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        poly = Polygon(cnt[:, 0, :])
        polygons.append(poly)
    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf.to_file(save_path, encoding='utf-8')
    print(f"SHP保存完成：{save_path}")

# 计算IoU指标
def calculate_iou(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()