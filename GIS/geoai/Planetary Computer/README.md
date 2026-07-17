# Planetary Computer 资产下载

读取公开 STAC Item、检查资产与元数据，并对选定资产签名后流式下载。原 `download_sentinel2.py` 仍可直接运行；Web 入口：

```powershell
conda activate geoai
python start.py --no-install
```

Web：`http://127.0.0.1:5181`；API：`http://127.0.0.1:8021/docs`。自动测试只模拟 STAC 响应，真实下载依赖网络。
