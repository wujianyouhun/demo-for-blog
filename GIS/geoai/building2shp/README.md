# Building2SHP

离线合成多类遥感样本，训练 DeepLabV3+ 并将建筑分割结果正则化为 GIS 矢量。训练与验证按固定索引互斥，增强只作用于训练集，保存验证建筑 F1 最佳权重。

```powershell
conda activate geoai
python start.py --no-install
```

默认前后端为 `5184 / 8024`；CLI 可运行 `python train_model.py --help` 和 `python building_extractor.py --help`。
