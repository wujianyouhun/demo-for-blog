# GeoAI 栅格与矢量元数据检查

已移除原脚本中的固定 `E:\` 示例路径，改为标准 CLI 和独立 Web。支持栅格波段统计、矢量几何/属性统计、预览和 JSON 报告。

```powershell
python metaInfo.py path/to/data.tif --json metadata.json --preview preview.png
python start.py --no-install
```

Web：`http://127.0.0.1:5182`；API：`http://127.0.0.1:8022/docs`。
