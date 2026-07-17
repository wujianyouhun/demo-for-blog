# GeoAI 语义分割基线

本项目补全原有未完成实现，提供完整 U-Net++ 与无需下载预训练权重的轻量 DeepLabV3+。训练脚本严格按文件名配对影像和标签，训练增强不会进入验证集，并按验证 IoU 保存最佳权重。

```powershell
conda activate geoai
python train.py --model unetpp --epochs 20
python predict.py --input image.tif --checkpoint path/to/best.pth
python start.py --no-install
```

默认读取相邻 `GIS/geoai/makelable/data/output` 的真实建筑样本。Web 为 `http://127.0.0.1:5187`，API 文档为 `http://127.0.0.1:8027/docs`。模型默认保存到共享 `GIS/geoai/models/geoai_segmentation`。
