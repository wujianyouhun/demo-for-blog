# 祁连遥感影像 U-Net 五分类推理

本目录包含经典 U-Net、ResUNet、Attention U-Net、U-Net++、TransUNet、Swin-UNet 六个推理入口。当前流程读取 `Unet/Data/Qilian.tif`，加载本项目训练得到的五分类权重，输出道路、居民地、耕地、裸地、光伏场分类结果。

## 快速运行

```powershell
conda activate python12
python Unet/resunet_example.py --device cpu
```

其他模型入口：

```powershell
python Unet/unet_example.py --device cpu
python Unet/attention_unet_example.py --device cpu
python Unet/unetpp_example.py --device cpu
python Unet/transunet_example.py --device cpu
python Unet/swin_unet_example.py --device cpu
```

权重默认读取：

```text
models/Unet/qilian_multiclass_5class_refined/<模型>/best.pth
```

五分类结果默认输出：

```text
Unet/out/qilian_multiclass_5class_refined/<模型>/classification.tif
```

分类编码为：`0=道路`、`1=居民地`、`2=耕地`、`3=裸地`、`4=光伏场`。

## 建筑与植被/耕地提取

建筑提取：

```powershell
python Unet/resunet_example.py --task building --device cpu
```

植被/耕地提取：

```powershell
python Unet/resunet_example.py --task vegetation --device cpu
```

输出为二值掩膜和概率图：

```text
building/building_mask.tif
building/building_probability.tif
vegetation/vegetation_mask.tif
vegetation/vegetation_probability.tif
```

当前样本中没有单独的自然植被类，`vegetation` 目前对应 `cropland / 耕地`。如需真正植被提取，需要补充植被样本并重新训练。
