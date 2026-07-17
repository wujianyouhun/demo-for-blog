# GeoAI 样本制作

支持本地文件或环境变量配置的 STAC Item，将矢量重投影并裁剪到影像范围，再通过 `geoai.export_geotiff_tiles` 生成同步影像、标签、增强样本、预览和 manifest。

```powershell
$env:STAC_API_BASE='http://localhost:8010'
python start.py --no-install
```

Web：`http://127.0.0.1:5183`；API：`http://127.0.0.1:8023/docs`。原 `dataprocess.py` 和 `geoai-process.py` 继续保留为 CLI/文章示例。
